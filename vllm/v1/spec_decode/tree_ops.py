# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-resident operations for bounded speculative proposal trees.

The first production experiment is intentionally narrow: one request, greedy
sampling, and at most 31 proposal nodes (32 root-inclusive target rows).  The
operations in this module keep selection and rollback on device so a branch
hit does not introduce a CPU synchronization into the decode loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.mla_tree_attention import MAX_TREE_NODES

MAX_PROPOSAL_NODES = MAX_TREE_NODES - 1


@dataclass(frozen=True)
class DeviceProposalTree:
    """Device metadata for one topologically packed proposal tree."""

    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    query_ancestor_masks: torch.Tensor
    request_id: str

    @property
    def num_nodes(self) -> int:
        return self.parent_indices.numel()

    def validate(self) -> DeviceProposalTree:
        if self.token_ids.ndim != 2 or self.token_ids.shape[0] != 1:
            raise ValueError("proposal-tree token_ids must have shape [1, N]")
        if (
            self.parent_indices.ndim != 1
            or self.depths.ndim != 1
            or self.query_ancestor_masks.ndim != 1
        ):
            raise ValueError("proposal-tree metadata tensors must be one-dimensional")
        if self.token_ids.shape[1] != self.num_nodes:
            raise ValueError("proposal-tree token and parent counts differ")
        if self.depths.numel() != self.num_nodes:
            raise ValueError("proposal-tree depth and parent counts differ")
        if self.query_ancestor_masks.numel() != self.num_nodes + 1:
            raise ValueError(
                "proposal-tree ancestor masks must include the committed query root"
            )
        if not 0 < self.num_nodes <= MAX_PROPOSAL_NODES:
            raise ValueError(
                f"proposal trees require 1..{MAX_PROPOSAL_NODES} nodes; "
                f"got {self.num_nodes}"
            )
        if not self.request_id:
            raise ValueError("proposal-tree request_id must not be empty")
        devices = {
            self.token_ids.device,
            self.parent_indices.device,
            self.depths.device,
            self.query_ancestor_masks.device,
        }
        if len(devices) != 1:
            raise ValueError("proposal-tree tensors must share one device")
        return self


def select_greedy_tree_path_reference(
    token_ids: list[int],
    parent_indices: list[int],
    target_next_token_ids: list[int],
) -> tuple[list[int], list[int]]:
    """CPU oracle returning emitted tokens and accepted packed node indices."""

    num_nodes = len(token_ids)
    if len(parent_indices) != num_nodes:
        raise ValueError("token and parent counts differ")
    if len(target_next_token_ids) != num_nodes + 1:
        raise ValueError("target rows must contain the root plus every proposal node")

    emitted: list[int] = []
    selected: list[int] = []
    parent_idx = -1
    while True:
        target_token_id = target_next_token_ids[parent_idx + 1]
        emitted.append(target_token_id)
        matching_child = next(
            (
                node_idx
                for node_idx, (token_id, node_parent) in enumerate(
                    zip(token_ids, parent_indices, strict=True)
                )
                if node_parent == parent_idx and token_id == target_token_id
            ),
            None,
        )
        if matching_child is None:
            return emitted, selected
        selected.append(matching_child)
        parent_idx = matching_child


@triton.jit
def _select_greedy_tree_path_kernel(
    proposal_token_ids_ptr,
    parent_indices_ptr,
    target_next_token_ids_ptr,
    output_token_ids_ptr,
    selected_node_indices_ptr,
    num_selected_ptr,
    num_nodes,
    BLOCK_N: tl.constexpr,
    MAX_TREE_ROWS: tl.constexpr,
):
    node_offsets = tl.arange(0, BLOCK_N)
    valid_nodes = node_offsets < num_nodes
    proposal_token_ids = tl.load(
        proposal_token_ids_ptr + node_offsets, mask=valid_nodes, other=-1
    )
    parent_indices = tl.load(
        parent_indices_ptr + node_offsets, mask=valid_nodes, other=-2
    )

    parent_idx = -1
    active = True
    num_selected = 0
    for step in tl.static_range(0, MAX_TREE_ROWS):
        target_token_id = tl.load(target_next_token_ids_ptr + parent_idx + 1)
        matches = (
            valid_nodes
            & (parent_indices == parent_idx)
            & (proposal_token_ids == target_token_id)
        )
        matching_child = tl.min(tl.where(matches, node_offsets, MAX_TREE_ROWS))
        has_match = matching_child < num_nodes

        tl.store(output_token_ids_ptr + step, target_token_id, mask=active)
        tl.store(
            selected_node_indices_ptr + step,
            matching_child,
            mask=active & has_match & (step < MAX_TREE_ROWS - 1),
        )
        num_selected += tl.where(active & has_match, 1, 0)
        parent_idx = tl.where(active & has_match, matching_child, parent_idx)
        active = active & has_match

    tl.store(num_selected_ptr, num_selected)


def select_greedy_tree_path(
    proposal_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_next_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one greedy branch and its target bonus token entirely on device.

    Returns ``(output_token_ids, selected_node_indices, num_selected)``.  The
    first tensor has shape ``[1, N + 1]`` and is padded with ``-1`` after the
    bonus token.  The selected-node tensor has shape ``[N]`` and is likewise
    padded with ``-1``. ``num_selected`` is a one-element int32 device tensor.
    """

    if proposal_token_ids.ndim == 2:
        if proposal_token_ids.shape[0] != 1:
            raise ValueError("greedy tree selection currently supports batch size one")
        proposal_token_ids = proposal_token_ids[0]
    if proposal_token_ids.ndim != 1 or parent_indices.ndim != 1:
        raise ValueError("proposal tokens and parents must be one-dimensional")
    num_nodes = proposal_token_ids.numel()
    if not 0 < num_nodes <= MAX_PROPOSAL_NODES:
        raise ValueError(
            f"greedy tree selection requires 1..{MAX_PROPOSAL_NODES} nodes"
        )
    if parent_indices.numel() != num_nodes:
        raise ValueError("proposal token and parent counts differ")
    if target_next_token_ids.shape != (num_nodes + 1,):
        raise ValueError(f"target_next_token_ids must have shape ({num_nodes + 1},)")
    if not proposal_token_ids.is_cuda:
        raise ValueError("greedy tree selection requires CUDA tensors")
    if not (
        parent_indices.device
        == proposal_token_ids.device
        == target_next_token_ids.device
    ):
        raise ValueError("greedy tree selection tensors must share one device")

    output_token_ids = torch.full(
        (1, num_nodes + 1),
        -1,
        dtype=target_next_token_ids.dtype,
        device=target_next_token_ids.device,
    )
    selected_node_indices = torch.full(
        (num_nodes,),
        -1,
        dtype=torch.int32,
        device=target_next_token_ids.device,
    )
    num_selected = torch.empty(
        (1,), dtype=torch.int32, device=target_next_token_ids.device
    )
    _select_greedy_tree_path_kernel[(1,)](
        proposal_token_ids,
        parent_indices,
        target_next_token_ids,
        output_token_ids,
        selected_node_indices,
        num_selected,
        num_nodes,
        BLOCK_N=32,
        MAX_TREE_ROWS=MAX_TREE_NODES,
    )
    return output_token_ids, selected_node_indices, num_selected


@triton.jit
def _compact_tree_rows_kernel(
    rows_ptr,
    selected_node_indices_ptr,
    num_selected_ptr,
    row_width,
    BLOCK_E: tl.constexpr,
    MAX_N: tl.constexpr,
):
    element_offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    valid_elements = element_offsets < row_width
    num_selected = tl.load(num_selected_ptr)

    # Forward order is overlap-safe for topologically packed nodes: destination
    # row step+1 never lies after its selected source row.  Keeping every
    # element's copies in one program also prevents cross-pair read/write races.
    for step in tl.static_range(0, MAX_N):
        active = step < num_selected
        selected_node = tl.load(selected_node_indices_ptr + step, mask=active, other=0)
        source_row = selected_node + 1
        destination_row = step + 1
        values = tl.load(
            rows_ptr + source_row * row_width + element_offsets,
            mask=active & valid_elements,
        )
        tl.store(
            rows_ptr + destination_row * row_width + element_offsets,
            values,
            mask=active & valid_elements,
        )


def compact_tree_rows(
    rows: torch.Tensor,
    selected_node_indices: torch.Tensor,
    num_selected: torch.Tensor,
) -> None:
    """Move selected proposal rows behind root row zero, in place."""

    if not rows.is_cuda or not rows.is_contiguous():
        raise ValueError("tree-row compaction requires a contiguous CUDA tensor")
    if rows.ndim < 1:
        raise ValueError("tree-row compaction requires at least one dimension")
    if selected_node_indices.ndim != 1 or num_selected.shape != (1,):
        raise ValueError("invalid tree selection tensor shapes")
    row_width = rows[0].numel() if rows.ndim > 1 else 1
    _compact_tree_rows_kernel[(cdiv(row_width, 256),)](
        rows,
        selected_node_indices,
        num_selected,
        row_width,
        BLOCK_E=256,
        MAX_N=MAX_PROPOSAL_NODES,
    )


@triton.jit
def _compact_kv_storage_kernel(
    storage_ptr,
    layer_offsets_ptr,
    block_strides_ptr,
    token_strides_ptr,
    element_strides_ptr,
    block_sizes_ptr,
    widths_ptr,
    query_slot_mapping_ptr,
    selected_node_indices_ptr,
    num_selected_ptr,
    BLOCK_E: tl.constexpr,
    MAX_N: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    element_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    layer_offset = tl.load(layer_offsets_ptr + layer_idx)
    block_stride = tl.load(block_strides_ptr + layer_idx)
    token_stride = tl.load(token_strides_ptr + layer_idx)
    element_stride = tl.load(element_strides_ptr + layer_idx)
    block_size = tl.load(block_sizes_ptr + layer_idx)
    width = tl.load(widths_ptr + layer_idx)
    valid_elements = element_offsets < width
    num_selected = tl.load(num_selected_ptr)

    for step in tl.static_range(0, MAX_N):
        active = step < num_selected
        selected_node = tl.load(selected_node_indices_ptr + step, mask=active, other=0)
        source_slot = tl.load(
            query_slot_mapping_ptr + selected_node + 1, mask=active, other=0
        )
        destination_slot = tl.load(
            query_slot_mapping_ptr + step + 1, mask=active, other=0
        )
        source_offset = (
            layer_offset
            + (source_slot // block_size) * block_stride
            + (source_slot % block_size) * token_stride
            + element_offsets * element_stride
        )
        destination_offset = (
            layer_offset
            + (destination_slot // block_size) * block_stride
            + (destination_slot % block_size) * token_stride
            + element_offsets * element_stride
        )
        values = tl.load(storage_ptr + source_offset, mask=active & valid_elements)
        tl.store(
            storage_ptr + destination_offset,
            values,
            mask=active & valid_elements,
        )


@dataclass(frozen=True)
class _KVStorageGroup:
    storage: torch.Tensor
    layer_offsets: torch.Tensor
    block_strides: torch.Tensor
    token_strides: torch.Tensor
    element_strides: torch.Tensor
    block_sizes: torch.Tensor
    widths: torch.Tensor
    max_width: int


class KVCacheSlotCompactor:
    """Fused selected-path compaction for one or more MLA cache storages."""

    def __init__(self, kv_caches: list[torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("KV compactor requires at least one cache tensor")

        unique_caches: dict[tuple[object, ...], torch.Tensor] = {}
        for cache in kv_caches:
            if not cache.is_cuda or cache.ndim != 3:
                raise ValueError(
                    "proposal-tree KV compaction currently requires rank-3 CUDA "
                    "MLA caches"
                )
            storage = cache.untyped_storage()
            identity = (
                storage.data_ptr(),
                cache.dtype,
                cache.storage_offset(),
                tuple(cache.shape),
                tuple(cache.stride()),
            )
            unique_caches.setdefault(identity, cache)

        grouped: dict[tuple[int, torch.dtype], list[torch.Tensor]] = {}
        for cache in unique_caches.values():
            key = (cache.untyped_storage().data_ptr(), cache.dtype)
            grouped.setdefault(key, []).append(cache)

        groups: list[_KVStorageGroup] = []
        for caches in grouped.values():
            representative = caches[0]
            storage = representative.untyped_storage()
            num_storage_elements = storage.nbytes() // representative.element_size()
            storage_tensor = torch.empty(
                0, dtype=representative.dtype, device=representative.device
            ).set_(storage, 0, (num_storage_elements,), (1,))

            groups.append(
                _KVStorageGroup(
                    storage=storage_tensor,
                    layer_offsets=torch.tensor(
                        [c.storage_offset() for c in caches],
                        dtype=torch.int64,
                        device=representative.device,
                    ),
                    block_strides=torch.tensor(
                        [c.stride(0) for c in caches],
                        dtype=torch.int64,
                        device=representative.device,
                    ),
                    token_strides=torch.tensor(
                        [c.stride(1) for c in caches],
                        dtype=torch.int64,
                        device=representative.device,
                    ),
                    element_strides=torch.tensor(
                        [c.stride(2) for c in caches],
                        dtype=torch.int64,
                        device=representative.device,
                    ),
                    block_sizes=torch.tensor(
                        [c.shape[1] for c in caches],
                        dtype=torch.int64,
                        device=representative.device,
                    ),
                    widths=torch.tensor(
                        [c.shape[2] for c in caches],
                        dtype=torch.int64,
                        device=representative.device,
                    ),
                    max_width=max(c.shape[2] for c in caches),
                )
            )
        self._groups = tuple(groups)

    @property
    def num_storage_groups(self) -> int:
        return len(self._groups)

    def compact(
        self,
        query_slot_mapping: torch.Tensor,
        selected_node_indices: torch.Tensor,
        num_selected: torch.Tensor,
    ) -> None:
        if query_slot_mapping.ndim != 1:
            raise ValueError("query_slot_mapping must be one-dimensional")
        for group in self._groups:
            num_layers = group.layer_offsets.numel()
            _compact_kv_storage_kernel[(num_layers, cdiv(group.max_width, 256))](
                group.storage,
                group.layer_offsets,
                group.block_strides,
                group.token_strides,
                group.element_strides,
                group.block_sizes,
                group.widths,
                query_slot_mapping,
                selected_node_indices,
                num_selected,
                BLOCK_E=256,
                MAX_N=MAX_PROPOSAL_NODES,
            )
