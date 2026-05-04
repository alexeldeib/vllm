# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata
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


def _make_ddtree_drafter_token_indices(
    query_start_loc_cpu: torch.Tensor,
    sampled_token_ids: list[list[int]],
    num_draft_tokens: list[int],
    accepted_node_indices: list[list[int]],
    ddtree_metadata: list[DDTreeRequestMetadata | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map DDTree accepted paths back to target-forward token indices.

    Generic spec decode can keep the first N target positions for each request
    because draft verification is linear. DDTree verification is depth ordered,
    so the accepted path can be non-contiguous in the flattened tree window.
    """
    batch_size = len(num_draft_tokens)
    if (
        len(sampled_token_ids) != batch_size
        or len(accepted_node_indices) != batch_size
        or len(ddtree_metadata) != batch_size
        or query_start_loc_cpu.numel() != batch_size + 1
    ):
        raise ValueError(
            "DDTree drafter inputs are not batch-aligned: "
            f"{len(sampled_token_ids)=}, {len(num_draft_tokens)=}, "
            f"{len(accepted_node_indices)=}, {len(ddtree_metadata)=}, "
            f"{query_start_loc_cpu.numel()=}"
        )

    token_indices: list[int] = []
    num_rejected_tokens: list[int] = []
    query_start_np = query_start_loc_cpu.numpy()
    for req_index, (tokens, num_draft, accepted_nodes, tree_metadata) in enumerate(
        zip(
            sampled_token_ids,
            num_draft_tokens,
            accepted_node_indices,
            ddtree_metadata,
        )
    ):
        req_start = int(query_start_np[req_index])
        req_end = int(query_start_np[req_index + 1])
        query_len = req_end - req_start

        if tree_metadata is None:
            if accepted_nodes:
                raise ValueError(
                    "Accepted DDTree nodes were reported for a non-DDTree "
                    f"request row {req_index}: {accepted_nodes}"
                )
            token_indices.extend(range(req_start, req_end))
            num_rejected_tokens.append(0)
            continue

        if len(tokens) != len(accepted_nodes) + 1:
            raise ValueError(
                "DDTree sampled token count must equal accepted nodes plus "
                f"bonus token for request {req_index}: {len(tokens)} vs "
                f"{len(accepted_nodes)}"
            )

        tree_len = num_draft + 1
        if query_len < tree_len:
            raise ValueError(
                "DDTree query window does not match draft token count for "
                f"request {req_index}: {query_len} < {tree_len}"
            )

        tree_offset = query_len - tree_len
        token_indices.extend(range(req_start, req_start + tree_offset))
        path = [0, *accepted_nodes]
        for node_index in path:
            if node_index < 0 or node_index >= tree_len:
                raise ValueError(
                    "DDTree accepted node index is outside the scheduled tree "
                    f"window for request {req_index}: {node_index}"
                )
            token_indices.append(req_start + tree_offset + node_index)
        num_rejected_tokens.append(tree_len - len(path))

    return (
        torch.tensor(num_rejected_tokens, dtype=torch.int32),
        torch.tensor(token_indices, dtype=torch.int64),
    )


def _build_ddtree_tree_from_topk(
    top_logprobs_cpu: torch.Tensor,
    top_token_ids_cpu: torch.Tensor,
    budget: int,
) -> DDTreeDraft:
    if budget <= 0:
        return DDTreeDraft([], DDTreeRequestMetadata([], [], 0))

    if top_logprobs_cpu.ndim != 2 or top_token_ids_cpu.ndim != 2:
        raise ValueError(
            "Expected [horizon, topk] tensors, got "
            f"{top_logprobs_cpu.shape} and {top_token_ids_cpu.shape}"
        )
    if top_logprobs_cpu.shape != top_token_ids_cpu.shape:
        raise ValueError(
            "top_logprobs and top_token_ids must have the same shape, got "
            f"{top_logprobs_cpu.shape} and {top_token_ids_cpu.shape}"
        )

    depth_limit, topk = top_logprobs_cpu.shape
    if depth_limit == 0 or topk == 0:
        return DDTreeDraft([], DDTreeRequestMetadata([], [], 0))

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
    return _build_ddtree_tree_from_topk(
        top_logprobs.detach().cpu(),
        top_token_ids.detach().cpu(),
        budget,
    )


def make_ddtree_attention_bias(
    metadata: DDTreeRequestMetadata,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the dense root/ancestor/self attention bias for one DDTree."""
    tree_len = len(metadata.node_depths) + 1
    build_on_cpu = device.type == "cuda"
    bias_device = torch.device("cpu") if build_on_cpu else device
    bias = torch.full((tree_len, tree_len), -torch.inf, device=bias_device, dtype=dtype)
    bias[:, 0] = 0
    bias.fill_diagonal_(0)

    for node_index, parent in enumerate(metadata.parents, start=1):
        while parent:
            bias[node_index, parent] = 0
            parent = metadata.parents[parent - 1]
    if build_on_cpu:
        return bias.to(device, non_blocking=True)
    return bias


def make_batched_ddtree_attention_bias(
    metadata: list[DDTreeRequestMetadata | None],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build per-request dense tree attention biases for a DDTree batch.

    Requests without DDTree metadata use a zero bias, which preserves normal
    causal attention behavior. Requests with metadata get the DDTree
    root/ancestor/self visibility mask in their active tree window.
    """
    if not metadata:
        return torch.empty((0, 0, 0), device=device, dtype=dtype)

    max_tree_len = max(
        (len(m.node_depths) + 1 if m is not None else 1) for m in metadata
    )
    build_on_cpu = device.type == "cuda"
    bias_device = torch.device("cpu") if build_on_cpu else device
    pin_memory = build_on_cpu and is_pin_memory_available()
    batched_bias = torch.zeros(
        (len(metadata), max_tree_len, max_tree_len),
        device=bias_device,
        dtype=dtype,
        pin_memory=pin_memory,
    )

    for req_index, tree_metadata in enumerate(metadata):
        if tree_metadata is None:
            continue
        tree_len = len(tree_metadata.node_depths) + 1
        req_bias = batched_bias[req_index, :tree_len, :tree_len]
        req_bias.fill_(-torch.inf)
        req_bias[:, 0] = 0
        req_bias.fill_diagonal_(0)
        for node_index, parent in enumerate(tree_metadata.parents, start=1):
            while parent:
                req_bias[node_index, parent] = 0
                parent = tree_metadata.parents[parent - 1]
    if build_on_cpu:
        return batched_bias.to(device, non_blocking=True)
    return batched_bias


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
    max_output_len = (
        max((m.max_depth if m is not None else 0 for m in ddtree_metadata), default=0)
        + 1
    )
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
    accepted_node_indices_by_req: list[list[int]] = []
    empty_metadata = DDTreeRequestMetadata([], [], 0)
    for req_index, tree in enumerate(ddtree_metadata):
        tree = tree or empty_metadata
        num_nodes = len(tree.node_depths)
        children: dict[int, dict[int, int]] = {}
        req_draft_tokens = draft_token_ids[draft_offset : draft_offset + num_nodes]
        for node_index, (parent, token_id) in enumerate(
            zip(tree.parents, req_draft_tokens), start=1
        ):
            children.setdefault(parent, {})[int(token_id)] = node_index

        current_node = 0
        emitted: list[int] = []
        accepted_node_indices: list[int] = []
        while True:
            target_token_id = int(target_argmax[logit_offset + current_node])
            emitted.append(target_token_id)
            child = children.get(current_node, {}).get(target_token_id)
            if child is None:
                break
            accepted_node_indices.append(child)
            current_node = child

        output_token_ids[req_index, : len(emitted)] = torch.tensor(
            emitted,
            dtype=torch.int32,
            device=logits.device,
        )
        accepted_node_indices_by_req.append(accepted_node_indices)
        logit_offset += num_nodes + 1
        draft_offset += num_nodes

    return SamplerOutput(
        sampled_token_ids=output_token_ids,
        logprobs_tensors=None,
        ddtree_accepted_node_indices=accepted_node_indices_by_req,
    )


class DDTreeProposer(DFlashProposer):
    """DFlash proposer that turns one parallel draft pass into DDTree nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = self.vllm_config.speculative_config
        assert spec_config is not None
        self.tree_budget = spec_config.get_ddtree_tree_budget()
        self._last_ddtree_metadata: list[DDTreeRequestMetadata] | None = None
        self._last_ddtree_debug_metrics: dict[str, Any] | None = None
        self._debug_metrics = os.environ.get("SPEC_DECODE_DEBUG_METRICS") == "1"

    def _debug_sync(self) -> None:
        if self._debug_metrics and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def propose_ddtree_from_logits(
        self,
        logits: torch.Tensor,
        batch_size: int,
    ) -> list[list[int]]:
        draft_logits = logits.view(batch_size, self.num_speculative_tokens, -1)
        self._last_ddtree_debug_metrics = None
        if draft_logits.shape[-1] == 0 or self.tree_budget <= 0:
            empty = DDTreeRequestMetadata([], [], 0)
            self._last_ddtree_metadata = [empty for _ in range(batch_size)]
            return [[] for _ in range(batch_size)]

        topk = min(self.tree_budget, draft_logits.shape[-1])

        self._debug_sync()
        topk_start = time.perf_counter()
        logprobs = torch.log_softmax(draft_logits.float(), dim=-1)
        top_logprobs, top_token_ids = torch.topk(logprobs, k=topk, dim=-1)
        self._debug_sync()
        topk_ms = (time.perf_counter() - topk_start) * 1000

        transfer_start = time.perf_counter()
        top_logprobs_cpu = top_logprobs.detach().cpu()
        top_token_ids_cpu = top_token_ids.detach().cpu()
        transfer_ms = (time.perf_counter() - transfer_start) * 1000

        build_start = time.perf_counter()
        all_token_ids: list[list[int]] = []
        all_metadata: list[DDTreeRequestMetadata] = []
        for req_index in range(batch_size):
            draft = _build_ddtree_tree_from_topk(
                top_logprobs_cpu[req_index],
                top_token_ids_cpu[req_index],
                self.tree_budget,
            )
            all_token_ids.append(draft.token_ids)
            all_metadata.append(draft.metadata)
        build_ms = (time.perf_counter() - build_start) * 1000
        self._last_ddtree_metadata = all_metadata
        if self._debug_metrics:
            self._last_ddtree_debug_metrics = {
                "tree_budget": self.tree_budget,
                "batch_size": batch_size,
                "horizon": self.num_speculative_tokens,
                "topk": topk,
                "tree_nodes": [len(tokens) for tokens in all_token_ids],
                "tree_max_depths": [metadata.max_depth for metadata in all_metadata],
                "tree_topk_ms": topk_ms,
                "tree_transfer_ms": transfer_ms,
                "tree_cpu_build_ms": build_ms,
            }
        return all_token_ids

    def take_last_ddtree_metadata(self) -> list[DDTreeRequestMetadata] | None:
        metadata = self._last_ddtree_metadata
        self._last_ddtree_metadata = None
        return metadata

    def take_last_ddtree_debug_metrics(self) -> dict[str, Any] | None:
        metrics = self._last_ddtree_debug_metrics
        self._last_ddtree_debug_metrics = None
        return metrics

    def prepare_ddtree_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
        accepted_node_indices: list[list[int]],
        ddtree_metadata: list[DDTreeRequestMetadata | None],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """Prepare DFlash drafter context after a DDTree verification step."""
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_rejected_tokens, token_indices_cpu = _make_ddtree_drafter_token_indices(
            query_start_loc_cpu,
            sampled_token_ids,
            num_draft_tokens,
            accepted_node_indices,
            ddtree_metadata,
        )

        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu_upper_bound - num_rejected_tokens
        )

        new_query_len_per_req = (
            query_start_loc_cpu[1:] - query_start_loc_cpu[:-1] - num_rejected_tokens
        )
        new_query_len_per_req_np = new_query_len_per_req.numpy()

        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_query_len_per_req_np, out=new_query_start_loc_np[1:])

        device = common_attn_metadata.query_start_loc.device
        token_indices = token_indices_cpu.to(device, non_blocking=True)
        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=int(new_query_start_loc_np[-1]),
            max_query_len=int(new_query_len_per_req.max().item()),
            max_seq_len=int(new_seq_lens_cpu.max().item()),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices
