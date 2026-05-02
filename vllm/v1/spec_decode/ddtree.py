# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1


@dataclass
class DDTreeRequestMetadata:
    """CPU-side metadata for one dynamic DDTree draft."""

    node_depths: list[int]
    parents: list[int]
    max_depth: int


@dataclass
class DDTreeDraft:
    token_ids: list[int]
    metadata: DDTreeRequestMetadata


def _order_nodes_by_depth(
    token_ids: list[int],
    node_depths: list[int],
    parents: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Put dynamic tree nodes in verifier-friendly depth order."""
    order = sorted(range(len(token_ids)), key=lambda i: (node_depths[i], i))

    old_to_new = {0: 0}
    for new_index, old_index in enumerate(order, start=1):
        old_to_new[old_index + 1] = new_index

    return (
        [token_ids[i] for i in order],
        [node_depths[i] for i in order],
        [old_to_new[parents[i]] for i in order],
    )


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> DDTreeDraft:
    """Build a prefix-closed DDTree from one request's DFlash logits.

    The heap expansion mirrors the reference DDTree prototype. Parent indices
    are 1-based for tree nodes and use 0 for the root. The final node order is
    normalized by depth for the verifier's tree attention path.
    """
    if draft_logits.ndim != 2:
        raise ValueError(f"Expected [horizon, vocab] logits, got {draft_logits.shape}")
    if budget <= 0:
        return DDTreeDraft([], DDTreeRequestMetadata([], [], 0))

    depth_limit, vocab_size = draft_logits.shape
    if depth_limit == 0 or vocab_size == 0:
        return DDTreeDraft([], DDTreeRequestMetadata([], [], 0))

    topk = min(budget, vocab_size)
    logprobs = torch.log_softmax(draft_logits.float(), dim=-1)
    top_logprobs, top_token_ids = torch.topk(logprobs, k=topk, dim=-1)

    top_logprobs_cpu = top_logprobs.detach().cpu()
    top_token_ids_cpu = top_token_ids.detach().cpu()

    node_token_ids: list[int] = []
    node_depths: list[int] = []
    parents: list[int] = []

    # Max heap via negative cumulative logprob.
    # Entries are (-score, depth, parent_index, rank_at_depth).
    heap: list[tuple[float, int, int, int]] = [
        (-float(top_logprobs_cpu[0, 0]), 0, 0, 0)
    ]

    while heap and len(node_token_ids) < budget:
        neg_score, depth, parent_index, rank = heapq.heappop(heap)
        token_id = int(top_token_ids_cpu[depth, rank])

        node_index = len(node_token_ids) + 1
        node_token_ids.append(token_id)
        node_depths.append(depth + 1)
        parents.append(parent_index)

        sibling_rank = rank + 1
        if sibling_rank < topk:
            sibling_score = -neg_score - float(top_logprobs_cpu[depth, rank])
            sibling_score += float(top_logprobs_cpu[depth, sibling_rank])
            heapq.heappush(
                heap, (-sibling_score, depth, parent_index, sibling_rank)
            )

        child_depth = depth + 1
        if child_depth < depth_limit:
            child_score = -neg_score + float(top_logprobs_cpu[child_depth, 0])
            heapq.heappush(heap, (-child_score, child_depth, node_index, 0))

    node_token_ids, node_depths, parents = _order_nodes_by_depth(
        node_token_ids, node_depths, parents
    )
    max_depth = max(node_depths, default=0)
    return DDTreeDraft(
        token_ids=node_token_ids,
        metadata=DDTreeRequestMetadata(
            node_depths=node_depths,
            parents=parents,
            max_depth=max_depth,
        ),
    )


def make_ddtree_attention_bias(
    metadata: DDTreeRequestMetadata,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the dense root/ancestor/self attention bias for one DDTree."""
    tree_len = len(metadata.node_depths) + 1
    bias = torch.full((tree_len, tree_len), -torch.inf, device=device, dtype=dtype)
    bias[:, 0] = 0
    bias.fill_diagonal_(0)

    for node_index, parent in enumerate(metadata.parents, start=1):
        while parent:
            bias[node_index, parent] = 0
            parent = metadata.parents[parent - 1]
    return bias


def ddtree_greedy_sample(
    metadata: SpecDecodeMetadata,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    """Greedy target walk for DDTree verification.

    This mirrors the reference DDTree walk: target argmax at the current tree
    node is accepted only when it names one of the current node's children. The
    first target token that leaves the tree is emitted as the fallback token.
    """
    if not sampling_metadata.all_greedy:
        raise NotImplementedError("DDTree currently supports greedy sampling only.")
    if sampling_metadata.max_num_logprobs is not None:
        raise NotImplementedError("DDTree logprobs are not implemented yet.")

    ddtree_metadata = metadata.ddtree_metadata
    if ddtree_metadata is None:
        raise ValueError("DDTree sampler requires metadata.ddtree_metadata.")

    batch_size = len(ddtree_metadata)
    max_output_len = max((m.max_depth for m in ddtree_metadata), default=0) + 1
    output_token_ids = torch.full(
        (batch_size, max_output_len),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=logits.device,
    )

    target_argmax = logits.argmax(dim=-1).detach().cpu().tolist()
    draft_token_ids = metadata.draft_token_ids.detach().cpu().tolist()

    logit_offset = 0
    draft_offset = 0
    for req_index, tree in enumerate(ddtree_metadata):
        num_nodes = len(tree.node_depths)
        children: dict[int, dict[int, int]] = {}
        req_draft_tokens = draft_token_ids[draft_offset : draft_offset + num_nodes]
        for node_index, (parent, token_id) in enumerate(
            zip(tree.parents, req_draft_tokens), start=1
        ):
            children.setdefault(parent, {})[int(token_id)] = node_index

        current_node = 0
        emitted: list[int] = []
        while True:
            target_token_id = int(target_argmax[logit_offset + current_node])
            emitted.append(target_token_id)
            child = children.get(current_node, {}).get(target_token_id)
            if child is None:
                break
            current_node = child

        output_token_ids[req_index, : len(emitted)] = torch.tensor(
            emitted,
            dtype=torch.int32,
            device=logits.device,
        )
        logit_offset += num_nodes + 1
        draft_offset += num_nodes

    return SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=None)


class DDTreeProposer(DFlashProposer):
    """DFlash proposer that turns one parallel draft pass into DDTree nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = self.vllm_config.speculative_config
        assert spec_config is not None
        self.tree_budget = spec_config.get_ddtree_tree_budget()
        self._last_ddtree_metadata: list[DDTreeRequestMetadata] | None = None

    def propose_ddtree_from_logits(
        self,
        logits: torch.Tensor,
        batch_size: int,
    ) -> list[list[int]]:
        draft_logits = logits.view(batch_size, self.num_speculative_tokens, -1)
        all_token_ids: list[list[int]] = []
        all_metadata: list[DDTreeRequestMetadata] = []
        for req_index in range(batch_size):
            draft = build_ddtree_tree(draft_logits[req_index], self.tree_budget)
            all_token_ids.append(draft.token_ids)
            all_metadata.append(draft.metadata)
        self._last_ddtree_metadata = all_metadata
        return all_token_ids

    def take_last_ddtree_metadata(self) -> list[DDTreeRequestMetadata] | None:
        metadata = self._last_ddtree_metadata
        self._last_ddtree_metadata = None
        return metadata
