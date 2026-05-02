# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.ddtree import (
    DDTreeRequestMetadata,
    build_ddtree_tree,
    ddtree_greedy_sample,
    make_ddtree_attention_bias,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _sampling_metadata() -> SamplingMetadata:
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
        output_token_ids=[[]],
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
