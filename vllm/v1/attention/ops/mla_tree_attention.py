# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small-suffix MLA attention for speculative proposal trees.

Tree verification is split into two exact partial-attention problems:

* the existing paged decode kernel attends every query to the committed prefix;
* this module attends every query to its root-to-node path in the current tree.

The two partial states can then be combined with ``merge_attn_states`` using
their log-sum-exp values.  Keeping the tree suffix separate lets the target use
the tuned paged MLA kernel without materializing a full prefix-by-tree mask.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

MAX_TREE_NODES = 32


def build_tree_ancestor_masks(parent_indices: list[int]) -> list[int]:
    """Build one root-inclusive ancestor bit mask per tree node.

    Nodes must be in topological order.  Bit ``j`` in result ``i`` is set iff
    node ``j`` is node ``i`` itself or one of its ancestors.
    """

    num_nodes = len(parent_indices)
    if num_nodes == 0:
        raise ValueError("a proposal tree must contain its root")
    if num_nodes > MAX_TREE_NODES:
        raise ValueError(
            f"proposal trees support at most {MAX_TREE_NODES} nodes; got {num_nodes}"
        )
    if parent_indices[0] != -1:
        raise ValueError("the root parent must be -1")

    masks = [1]
    for node_idx, parent_idx in enumerate(parent_indices[1:], start=1):
        if parent_idx < 0 or parent_idx >= node_idx:
            raise ValueError(
                "tree nodes must be topologically ordered: "
                f"node {node_idx} has parent {parent_idx}"
            )
        masks.append(masks[parent_idx] | (1 << node_idx))
    return masks


def mla_tree_suffix_attention_reference(
    query: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    current_kv: torch.Tensor,
    ancestor_masks: torch.Tensor,
    *,
    kv_lora_rank: int,
    logit_scale: float,
    value_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Framework reference for root-to-node latent attention.

    ``query`` is ``[T, H, L + R]`` and ``current_kv`` is ``[T, L + R]``.
    Inputs may contain raw FP8 values; callers express their dequantization in
    ``logit_scale`` and ``value_scale`` just like the TRT-LLM MLA API.

    Returns latent output ``[T, H, L]`` and natural-log LSE ``[T, H]``.
    """

    if isinstance(query, tuple):
        query = torch.cat(query, dim=-1)

    if query.ndim != 3:
        raise ValueError(f"query must have rank 3, got shape {tuple(query.shape)}")
    if current_kv.ndim != 2:
        raise ValueError(
            f"current_kv must have rank 2, got shape {tuple(current_kv.shape)}"
        )

    num_nodes = query.shape[0]
    if current_kv.shape[0] != num_nodes:
        raise ValueError(
            "query and current_kv must contain the same number of tree nodes"
        )
    if ancestor_masks.shape != (num_nodes,):
        raise ValueError(
            f"ancestor_masks must have shape ({num_nodes},), "
            f"got {tuple(ancestor_masks.shape)}"
        )
    if num_nodes > MAX_TREE_NODES:
        raise ValueError(
            f"proposal trees support at most {MAX_TREE_NODES} nodes; got {num_nodes}"
        )
    if query.shape[-1] != current_kv.shape[-1]:
        raise ValueError("query and current_kv head dimensions must match")
    if not 0 < kv_lora_rank <= query.shape[-1]:
        raise ValueError(f"invalid kv_lora_rank {kv_lora_rank}")

    # Convert even FP8 inputs to FP32 before einsum.  The caller-provided
    # scales preserve the exact raw-value convention used by TRT-LLM MLA.
    q = query.float()
    kv = current_kv.float()
    logits = torch.matmul(q, kv.T)
    logits.mul_(logit_scale)

    key_indices = torch.arange(num_nodes, device=ancestor_masks.device)
    allowed = ((ancestor_masks.to(torch.int64)[:, None] >> key_indices) & 1).bool()
    logits.masked_fill_(~allowed[:, None, :], float("-inf"))

    lse = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, kv[:, :kv_lora_rank]).mul_(value_scale)
    output_dtype = query.dtype if query.element_size() > 1 else torch.bfloat16
    return output.to(output_dtype), lse


@triton.jit
def _mla_tree_suffix_attention_kernel(
    query_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    ancestor_masks_ptr,
    output_ptr,
    lse_ptr,
    query_stride_t,
    query_stride_h,
    kv_cache_stride_b,
    kv_cache_stride_t,
    output_stride_t,
    output_stride_h,
    lse_stride_t,
    logit_scale,
    value_scale,
    num_nodes: tl.constexpr,
    num_heads: tl.constexpr,
    block_size: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    rope_head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    latent_offsets = tl.arange(0, BLOCK_L)
    rope_offsets = tl.arange(0, BLOCK_R)
    node_offsets = tl.arange(0, BLOCK_N)
    node_mask = node_offsets < num_nodes

    q_latent_offsets = (
        query_idx * query_stride_t
        + head_offsets[:, None] * query_stride_h
        + latent_offsets[None, :]
    )
    q_latent = tl.load(
        query_ptr + q_latent_offsets,
        mask=head_mask[:, None] & (latent_offsets[None, :] < kv_lora_rank),
        other=0.0,
    ).to(tl.bfloat16)
    q_rope_offsets = (
        query_idx * query_stride_t
        + head_offsets[:, None] * query_stride_h
        + kv_lora_rank
        + rope_offsets[None, :]
    )
    q_rope = tl.load(
        query_ptr + q_rope_offsets,
        mask=head_mask[:, None] & (rope_offsets[None, :] < rope_head_dim),
        other=0.0,
    ).to(tl.bfloat16)

    slots = tl.load(slot_mapping_ptr + node_offsets, mask=node_mask, other=0)
    cache_blocks = slots // block_size
    cache_offsets = slots % block_size
    cache_token_offsets = (
        cache_blocks * kv_cache_stride_b + cache_offsets * kv_cache_stride_t
    )

    k_latent_offsets = latent_offsets[:, None] + cache_token_offsets[None, :]
    k_latent = tl.load(
        kv_cache_ptr + k_latent_offsets,
        mask=(latent_offsets[:, None] < kv_lora_rank) & node_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k_rope_offsets = kv_lora_rank + rope_offsets[:, None] + cache_token_offsets[None, :]
    k_rope = tl.load(
        kv_cache_ptr + k_rope_offsets,
        mask=(rope_offsets[:, None] < rope_head_dim) & node_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    logits = tl.dot(q_latent, k_latent)
    logits += tl.dot(q_rope, k_rope)
    logits *= logit_scale

    ancestor_bits = tl.load(ancestor_masks_ptr + query_idx)
    allowed = ((ancestor_bits >> node_offsets) & 1) != 0
    logits = tl.where(
        head_mask[:, None] & node_mask[None, :] & allowed[None, :],
        logits,
        float("-inf"),
    )

    max_logits = tl.max(logits, axis=1)
    probabilities = tl.exp(logits - max_logits[:, None])
    probability_sum = tl.sum(probabilities, axis=1)
    probabilities /= probability_sum[:, None]

    output = tl.dot(probabilities.to(tl.bfloat16), tl.trans(k_latent))
    output *= value_scale
    output_offsets = (
        query_idx * output_stride_t
        + head_offsets[:, None] * output_stride_h
        + latent_offsets[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=head_mask[:, None] & (latent_offsets[None, :] < kv_lora_rank),
    )
    tl.store(
        lse_ptr + query_idx * lse_stride_t + head_offsets,
        max_logits + tl.log(probability_sum),
        mask=head_mask,
    )


def mla_tree_suffix_attention(
    query: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    ancestor_masks: torch.Tensor,
    *,
    kv_lora_rank: int,
    rope_head_dim: int,
    logit_scale: float,
    value_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused small-suffix tree attention kernel.

    The paged cache is addressed by one physical slot per root-inclusive tree
    node.  Returned LSE uses ``[T, H]`` layout, matching FlashInfer's TRT-LLM
    decode API.
    """

    if isinstance(query, tuple):
        query = torch.cat(query, dim=-1)

    if not query.is_cuda:
        raise ValueError("mla_tree_suffix_attention requires CUDA tensors")
    if query.ndim != 3:
        raise ValueError(f"query must have rank 3, got shape {tuple(query.shape)}")
    if kv_cache.ndim != 3:
        raise ValueError(
            f"kv_cache must have rank 3, got shape {tuple(kv_cache.shape)}"
        )
    num_nodes, num_heads, query_dim = query.shape
    if num_nodes == 0 or num_nodes > MAX_TREE_NODES:
        raise ValueError(
            f"tree node count must be in [1, {MAX_TREE_NODES}], got {num_nodes}"
        )
    if query_dim != kv_lora_rank + rope_head_dim:
        raise ValueError(
            f"query head dimension must be {kv_lora_rank + rope_head_dim}, "
            f"got {query_dim}"
        )
    if kv_cache.shape[-1] != query_dim:
        raise ValueError("query and KV cache head dimensions must match")
    if slot_mapping.shape != (num_nodes,):
        raise ValueError(
            f"slot_mapping must have shape ({num_nodes},), "
            f"got {tuple(slot_mapping.shape)}"
        )
    if ancestor_masks.shape != (num_nodes,):
        raise ValueError(
            f"ancestor_masks must have shape ({num_nodes},), "
            f"got {tuple(ancestor_masks.shape)}"
        )

    output = torch.empty(
        (num_nodes, num_heads, kv_lora_rank),
        dtype=torch.bfloat16,
        device=query.device,
    )
    lse = torch.empty((num_nodes, num_heads), dtype=torch.float32, device=query.device)

    block_n = triton.next_power_of_2(num_nodes)
    block_h = min(16, triton.next_power_of_2(num_heads))
    block_l = triton.next_power_of_2(kv_lora_rank)
    block_r = triton.next_power_of_2(rope_head_dim)
    grid = (num_nodes, triton.cdiv(num_heads, block_h))
    _mla_tree_suffix_attention_kernel[grid](
        query,
        kv_cache,
        slot_mapping,
        ancestor_masks,
        output,
        lse,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        logit_scale,
        value_scale,
        num_nodes=num_nodes,
        num_heads=num_heads,
        block_size=kv_cache.shape[1],
        kv_lora_rank=kv_lora_rank,
        rope_head_dim=rope_head_dim,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        BLOCK_L=block_l,
        BLOCK_R=block_r,
        num_warps=4,
        num_stages=2,
    )
    return output, lse
