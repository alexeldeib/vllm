# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.ddtree import (
    DDTreeRequestMetadata,
    _make_ddtree_drafter_token_indices,
    build_ddtree_tree,
    ddtree_greedy_sample,
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
    num_rejected, token_indices = _make_ddtree_drafter_token_indices(
        query_start_loc_cpu=torch.tensor([0, 4, 6], dtype=torch.int32),
        sampled_token_ids=[[10, 11, 7], [30, 31]],
        num_draft_tokens=[3, 1],
        accepted_node_indices=[[1, 3], [1]],
    )

    assert num_rejected.tolist() == [1, 0]
    assert token_indices.tolist() == [0, 1, 3, 4, 5]
