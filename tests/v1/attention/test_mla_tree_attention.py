# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.mla_tree_attention import (
    MAX_TREE_NODES,
    build_tree_ancestor_masks,
    mla_tree_suffix_attention,
    mla_tree_suffix_attention_reference,
)


def test_build_tree_ancestor_masks() -> None:
    #            0
    #          /   \
    #         1     2
    #        / \     \
    #       3   4     5
    parents = [-1, 0, 0, 1, 1, 2]
    assert build_tree_ancestor_masks(parents) == [
        0b000001,
        0b000011,
        0b000101,
        0b001011,
        0b010011,
        0b100101,
    ]


@pytest.mark.parametrize(
    "parents, match",
    [
        ([], "contain its root"),
        ([0], "root parent"),
        ([-1, -1], "topologically ordered"),
        ([-1, 1], "topologically ordered"),
        ([-1, 0, 3], "topologically ordered"),
        ([-1] + [0] * MAX_TREE_NODES, "at most"),
    ],
)
def test_build_tree_ancestor_masks_rejects_invalid_trees(
    parents: list[int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_tree_ancestor_masks(parents)


def test_mla_tree_suffix_attention_reference_matches_explicit_paths() -> None:
    torch.manual_seed(7)
    parents = [-1, 0, 0, 1, 2, 3]
    masks = torch.tensor(build_tree_ancestor_masks(parents), dtype=torch.int64)
    num_nodes = len(parents)
    num_heads = 3
    kv_lora_rank = 8
    rope_head_dim = 4
    query = torch.randn(num_nodes, num_heads, kv_lora_rank + rope_head_dim)
    current_kv = torch.randn(num_nodes, kv_lora_rank + rope_head_dim)
    logit_scale = 0.37
    value_scale = 0.83

    output, lse = mla_tree_suffix_attention_reference(
        query,
        current_kv,
        masks,
        kv_lora_rank=kv_lora_rank,
        logit_scale=logit_scale,
        value_scale=value_scale,
    )

    expected_output = torch.empty_like(output)
    expected_lse = torch.empty_like(lse)
    for node_idx, mask in enumerate(masks.tolist()):
        path = [idx for idx in range(num_nodes) if mask & (1 << idx)]
        path_kv = current_kv[path]
        node_logits = torch.einsum("hd,sd->hs", query[node_idx], path_kv).mul(
            logit_scale
        )
        expected_lse[node_idx] = torch.logsumexp(node_logits, dim=-1)
        expected_output[node_idx] = torch.einsum(
            "hs,sd->hd",
            torch.softmax(node_logits, dim=-1),
            path_kv[:, :kv_lora_rank],
        ).mul(value_scale)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(lse, expected_lse)


def test_prefix_suffix_lse_merge_matches_monolithic_attention() -> None:
    torch.manual_seed(11)
    parents = [-1, 0, 0, 1, 3]
    masks = torch.tensor(build_tree_ancestor_masks(parents), dtype=torch.int64)
    num_nodes = len(parents)
    num_heads = 2
    kv_lora_rank = 8
    rope_head_dim = 4
    prefix_len = 13
    scale = 0.21

    query = torch.randn(num_nodes, num_heads, kv_lora_rank + rope_head_dim)
    prefix_kv = torch.randn(prefix_len, kv_lora_rank + rope_head_dim)
    current_kv = torch.randn(num_nodes, kv_lora_rank + rope_head_dim)

    suffix_output, suffix_lse = mla_tree_suffix_attention_reference(
        query,
        current_kv,
        masks,
        kv_lora_rank=kv_lora_rank,
        logit_scale=scale,
    )

    prefix_logits = torch.matmul(query, prefix_kv.T).mul(scale)
    prefix_lse = torch.logsumexp(prefix_logits, dim=-1)
    prefix_output = torch.matmul(
        torch.softmax(prefix_logits, dim=-1), prefix_kv[:, :kv_lora_rank]
    )

    max_lse = torch.maximum(prefix_lse, suffix_lse)
    prefix_weight = torch.exp(prefix_lse - max_lse)
    suffix_weight = torch.exp(suffix_lse - max_lse)
    merged = (
        prefix_output * prefix_weight[..., None]
        + suffix_output * suffix_weight[..., None]
    ) / (prefix_weight + suffix_weight)[..., None]

    expected = torch.empty_like(merged)
    for node_idx, mask in enumerate(masks.tolist()):
        path = [idx for idx in range(num_nodes) if mask & (1 << idx)]
        all_kv = torch.cat([prefix_kv, current_kv[path]])
        logits = torch.einsum("hd,sd->hs", query[node_idx], all_kv).mul(scale)
        expected[node_idx] = torch.einsum(
            "hs,sd->hd",
            torch.softmax(logits, dim=-1),
            all_kv[:, :kv_lora_rank],
        )

    torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("dtype", "logit_scale", "value_scale", "rtol", "atol"),
    [
        (torch.bfloat16, 0.04, 1.0, 3e-2, 3e-2),
        (torch.float8_e4m3fn, 0.02, 0.37, 8e-2, 8e-2),
    ],
)
def test_mla_tree_suffix_attention_cuda_matches_reference(
    dtype: torch.dtype,
    logit_scale: float,
    value_scale: float,
    rtol: float,
    atol: float,
) -> None:
    """Compile the Triton kernel with Kimi-shaped paged MLA inputs."""

    torch.manual_seed(19)
    device = torch.device("cuda")
    num_nodes = MAX_TREE_NODES
    num_heads = 16
    kv_lora_rank = 512
    rope_head_dim = 64
    head_dim = kv_lora_rank + rope_head_dim
    block_size = 32
    num_blocks = 41

    parents = [-1, *[(node_idx - 1) // 2 for node_idx in range(1, num_nodes)]]
    ancestor_masks = torch.tensor(
        build_tree_ancestor_masks(parents), dtype=torch.int64, device=device
    )
    # Cross page boundaries and avoid accidentally testing contiguous tokens.
    slot_mapping = (torch.arange(num_nodes, device=device) * 37 + 29) % (
        num_blocks * block_size
    )
    query = torch.randn(num_nodes, num_heads, head_dim, device=device).to(dtype)
    current_kv = torch.randn(num_nodes, head_dim, device=device).to(dtype)
    kv_cache = torch.zeros(num_blocks, block_size, head_dim, dtype=dtype, device=device)
    kv_cache.view(-1, head_dim)[slot_mapping] = current_kv

    output, lse = mla_tree_suffix_attention(
        query,
        kv_cache,
        slot_mapping,
        ancestor_masks,
        kv_lora_rank=kv_lora_rank,
        rope_head_dim=rope_head_dim,
        logit_scale=logit_scale,
        value_scale=value_scale,
    )
    expected_output, expected_lse = mla_tree_suffix_attention_reference(
        query,
        current_kv,
        ancestor_masks,
        kv_lora_rank=kv_lora_rank,
        logit_scale=logit_scale,
        value_scale=value_scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=rtol, atol=atol)
    torch.testing.assert_close(lse, expected_lse, rtol=rtol, atol=atol)
