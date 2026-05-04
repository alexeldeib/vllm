# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    _mla_tree_attention_ref,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.ddtree import (
    DDTreeRequestMetadata,
    DDTreeProposer,
    _make_ddtree_drafter_token_indices,
    build_ddtree_tree,
    ddtree_greedy_sample,
    ddtree_can_use_tree_verifier,
    make_batched_ddtree_attention_bias,
    make_ddtree_attention_bias,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _sampling_metadata() -> SamplingMetadata:
    return _sampling_metadata_for_batch(1)


def _sampling_metadata_for_batch(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=[],
    )


def _random_sampling_metadata_for_batch(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=[],
    )


def test_ddtree_verifier_is_greedy_only():
    greedy = _sampling_metadata_for_batch(2)
    random = _random_sampling_metadata_for_batch(2)
    with_logprobs = _sampling_metadata_for_batch(2)
    with_logprobs.max_num_logprobs = 1

    assert ddtree_can_use_tree_verifier(greedy)
    assert not ddtree_can_use_tree_verifier(random)
    assert not ddtree_can_use_tree_verifier(with_logprobs)


def test_ddtree_proposer_falls_back_to_linear_dflash_for_non_greedy():
    proposer = DDTreeProposer.__new__(DDTreeProposer)
    proposer.num_speculative_tokens = 2
    proposer.tree_budget = 4
    proposer.min_path_probability = None
    proposer.min_branch_gain_per_node = None
    proposer._last_ddtree_metadata = []
    proposer._last_ddtree_debug_metrics = None
    proposer._debug_metrics = True

    logits = torch.full((4, 8), -10.0)
    logits[0, 3] = 1.0
    logits[1, 4] = 1.0
    logits[2, 5] = 1.0
    logits[3, 6] = 1.0

    tokens = DDTreeProposer.propose_ddtree_from_logits(
        proposer,
        logits,
        batch_size=2,
        sampling_metadata=_random_sampling_metadata_for_batch(2),
    )

    assert tokens == [[3, 4], [5, 6]]
    assert proposer.take_last_ddtree_metadata() is None
    metrics = proposer.take_last_ddtree_debug_metrics()
    assert metrics is not None
    assert metrics["metadata_bypass"]
    assert metrics["fallback_reason"] == "non_greedy"
    assert metrics["tree_modes"] == ["linear_fallback", "linear_fallback"]


def test_build_ddtree_tree_is_prefix_closed():
    logits = torch.full((3, 8), -10.0)
    logits[0, 1] = 5.0
    logits[0, 2] = 4.0
    logits[1, 3] = 5.0
    logits[1, 4] = 4.0
    logits[2, 5] = 5.0
    logits[2, 6] = 4.0

    draft = build_ddtree_tree(logits, budget=5)

    assert len(draft.token_ids) == 5
    assert len(draft.metadata.parents) == 5
    assert draft.metadata.node_depths == sorted(draft.metadata.node_depths)
    for node_index, parent in enumerate(draft.metadata.parents, start=1):
        assert parent < node_index
    assert draft.metadata.max_depth <= logits.shape[0]


def test_build_ddtree_tree_prunes_low_probability_paths():
    logits = torch.full((3, 8), -4.0)
    logits[:, 1] = 0.0

    draft = build_ddtree_tree(
        logits,
        budget=6,
        min_path_probability=0.8,
    )

    assert draft.metadata.max_depth == 1
    assert len(draft.token_ids) == 1


def test_build_ddtree_tree_can_fallback_to_linear_chain():
    logits = torch.full((3, 8), -10.0)
    logits[0, 1] = 5.0
    logits[1, 2] = 5.0
    logits[2, 3] = 5.0

    draft = build_ddtree_tree(
        logits,
        budget=6,
        min_branch_gain_per_node=1.0,
    )

    assert draft.metadata.mode == "linear"
    assert draft.token_ids == [1, 2, 3]
    assert draft.metadata.node_depths == [1, 2, 3]
    assert draft.metadata.parents == [0, 1, 2]


def test_make_ddtree_attention_bias():
    metadata = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )

    visible = torch.isfinite(make_ddtree_attention_bias(metadata, device="cpu")).int()

    assert torch.equal(
        visible,
        torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
            dtype=torch.int32,
        ),
    )


def test_make_batched_ddtree_attention_bias_supports_variable_trees():
    metadata_a = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )
    metadata_b = DDTreeRequestMetadata(
        node_depths=[1],
        parents=[0],
        max_depth=1,
    )

    bias = make_batched_ddtree_attention_bias(
        [metadata_a, None, metadata_b], device="cpu"
    )
    visible = torch.isfinite(bias).int()

    assert bias.shape == (3, 4, 4)
    assert torch.equal(
        visible[0],
        torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(visible[1], torch.ones((4, 4), dtype=torch.int32))
    assert torch.equal(
        visible[2, :2, :2],
        torch.tensor(
            [
                [1, 0],
                [1, 1],
            ],
            dtype=torch.int32,
        ),
    )


def test_ddtree_greedy_sample_walks_tree():
    metadata = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )
    spec_metadata = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([10, 20, 11], dtype=torch.int32),
        num_draft_tokens=[3],
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([4], dtype=torch.int32),
        target_logits_indices=torch.arange(4, dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        logits_indices=torch.arange(4, dtype=torch.int32),
        ddtree_metadata=[metadata],
    )
    logits = torch.full((4, 32), -100.0)
    logits[0, 10] = 1.0
    logits[1, 11] = 1.0
    logits[3, 7] = 1.0

    out = ddtree_greedy_sample(spec_metadata, logits, _sampling_metadata())

    assert out.sampled_token_ids.tolist() == [[10, 11, 7]]
    assert out.ddtree_accepted_node_indices == [[1, 3]]


def test_ddtree_greedy_sample_handles_batched_variable_trees():
    metadata_a = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )
    metadata_b = DDTreeRequestMetadata(
        node_depths=[1],
        parents=[0],
        max_depth=1,
    )
    spec_metadata = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([10, 20, 11, 30], dtype=torch.int32),
        num_draft_tokens=[3, 1],
        cu_num_draft_tokens=torch.tensor([3, 4], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([4, 6], dtype=torch.int32),
        target_logits_indices=torch.arange(6, dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 5], dtype=torch.int32),
        logits_indices=torch.arange(6, dtype=torch.int32),
        ddtree_metadata=[metadata_a, metadata_b],
    )
    logits = torch.full((6, 64), -100.0)
    logits[0, 10] = 1.0
    logits[1, 11] = 1.0
    logits[3, 7] = 1.0
    logits[4, 30] = 1.0
    logits[5, 31] = 1.0

    out = ddtree_greedy_sample(
        spec_metadata, logits, _sampling_metadata_for_batch(2)
    )

    assert out.sampled_token_ids.tolist() == [[10, 11, 7], [30, 31, -1]]
    assert out.ddtree_accepted_node_indices == [[1, 3], [1]]


def test_make_ddtree_drafter_token_indices_follow_accepted_path():
    metadata_a = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )
    metadata_b = DDTreeRequestMetadata(
        node_depths=[1],
        parents=[0],
        max_depth=1,
    )
    num_rejected, token_indices = _make_ddtree_drafter_token_indices(
        query_start_loc_cpu=torch.tensor([0, 4, 6], dtype=torch.int32),
        sampled_token_ids=[[10, 11, 7], [30, 31]],
        num_draft_tokens=[3, 1],
        accepted_node_indices=[[1, 3], [1]],
        ddtree_metadata=[metadata_a, metadata_b],
    )

    assert num_rejected.tolist() == [1, 0]
    assert token_indices.tolist() == [0, 1, 3, 4, 5]


def test_make_ddtree_drafter_token_indices_keeps_non_tree_rows_contiguous():
    metadata = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )
    num_rejected, token_indices = _make_ddtree_drafter_token_indices(
        query_start_loc_cpu=torch.tensor([0, 4, 11], dtype=torch.int32),
        sampled_token_ids=[[10, 11, 7], [30]],
        num_draft_tokens=[3, 0],
        accepted_node_indices=[[1, 3], []],
        ddtree_metadata=[metadata, None],
    )

    assert num_rejected.tolist() == [1, 0]
    assert token_indices.tolist() == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]


def test_mla_tree_attention_ref_applies_tree_visibility_to_mla_cache():
    torch.manual_seed(0)

    kv_lora_rank = 3
    rope_dim = 2
    num_heads = 2
    head_size = kv_lora_rank + rope_dim
    block_size = 4

    tree_a = DDTreeRequestMetadata(
        node_depths=[1, 1, 2],
        parents=[0, 0, 1],
        max_depth=2,
    )
    tree_b = DDTreeRequestMetadata(
        node_depths=[1],
        parents=[0],
        max_depth=1,
    )
    tree_bias = make_batched_ddtree_attention_bias([tree_a, tree_b], device="cpu")

    query_start_loc_cpu = torch.tensor([0, 4, 6], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([7, 4], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [2, 4],
            [1, 3],
        ],
        dtype=torch.int32,
    )
    slot_mapping = torch.tensor(
        [
            4 * block_size + 3,
            2 * block_size + 0,
            2 * block_size + 1,
            2 * block_size + 2,
            3 * block_size + 2,
            3 * block_size + 3,
        ],
        dtype=torch.int64,
    )
    kv_cache = torch.randn(5, block_size, head_size)
    ql_nope = torch.randn(6, num_heads, kv_lora_rank)
    q_pe = torch.randn(6, num_heads, rope_dim)
    scale = head_size**-0.5

    metadata = MLACommonMetadata(
        num_reqs=2,
        max_query_len=4,
        max_seq_len=7,
        num_actual_tokens=6,
        query_start_loc=query_start_loc_cpu,
        slot_mapping=slot_mapping,
        num_decodes=2,
        num_decode_tokens=6,
        num_prefills=0,
        decode=MLACommonDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens_cpu,
            dcp_tot_seq_lens=None,
        ),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        tree_attn_bias=tree_bias,
    )

    actual = _mla_tree_attention_ref(
        ql_nope,
        q_pe,
        kv_cache,
        metadata,
        scale,
        kv_lora_rank,
    )

    expected_parts = []
    kv_cache_flat = kv_cache.flatten(0, 1)
    for req_index, q_len in enumerate([4, 2]):
        req_start = int(query_start_loc_cpu[req_index])
        context_len = int(seq_lens_cpu[req_index]) - q_len
        context_positions = torch.arange(context_len, dtype=torch.long)
        context_slots = (
            block_table[req_index, context_positions // block_size].to(torch.long)
            * block_size
            + context_positions % block_size
        )
        tree_slots = slot_mapping[req_start : req_start + q_len]
        kv = kv_cache_flat[torch.cat((context_slots, tree_slots))]
        q = torch.cat(
            (
                ql_nope[req_start : req_start + q_len],
                q_pe[req_start : req_start + q_len],
            ),
            dim=-1,
        )
        k = kv.unsqueeze(1).expand(-1, num_heads, -1)
        v = kv[:, :kv_lora_rank].unsqueeze(1).expand(-1, num_heads, -1)
        context_mask = torch.ones(q_len, context_len, dtype=torch.bool)
        tree_mask = torch.isfinite(tree_bias[req_index, :q_len, :q_len])
        attn_mask = torch.cat((context_mask, tree_mask), dim=-1)
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
            scale=scale,
        )
        expected_parts.append(expected.squeeze(0).transpose(0, 1))

    torch.testing.assert_close(actual, torch.cat(expected_parts, dim=0))
