# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.spec_decode.tree_ops import (
    KVCacheSlotCompactor,
    compact_tree_rows,
    select_greedy_tree_path,
    select_greedy_tree_path_reference,
)


def test_select_greedy_tree_path_reference_follows_conditioned_branch() -> None:
    # Virtual root -> {10, 20}; 10 -> 11; 20 -> 21.
    emitted, selected = select_greedy_tree_path_reference(
        [10, 20, 11, 21],
        [-1, -1, 0, 1],
        [20, 91, 21, 92, 99],
    )

    assert emitted == [20, 21, 99]
    assert selected == [1, 3]


def test_select_greedy_tree_path_reference_emits_full_path_bonus() -> None:
    emitted, selected = select_greedy_tree_path_reference(
        list(range(31)),
        list(range(-1, 30)),
        list(range(32)),
    )

    assert emitted == list(range(32))
    assert selected == list(range(31))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_select_and_compact_tree_rows_cuda() -> None:
    device = torch.device("cuda")
    proposal_tokens = torch.tensor([[10, 20, 11, 21]], device=device)
    parents = torch.tensor([-1, -1, 0, 1], dtype=torch.int32, device=device)
    target_tokens = torch.tensor([20, 91, 21, 92, 99], device=device)

    output, selected, num_selected = select_greedy_tree_path(
        proposal_tokens, parents, target_tokens
    )
    rows = torch.arange(15, device=device).view(5, 3)
    expected_root = rows[0].clone()
    expected_first = rows[2].clone()
    expected_second = rows[4].clone()
    compact_tree_rows(rows, selected, num_selected)
    torch.accelerator.synchronize()

    assert output.cpu().tolist() == [[20, 21, 99, -1, -1]]
    assert selected.cpu().tolist() == [1, 3, -1, -1]
    assert num_selected.item() == 2
    torch.testing.assert_close(rows[0], expected_root)
    torch.testing.assert_close(rows[1], expected_first)
    torch.testing.assert_close(rows[2], expected_second)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_select_greedy_tree_path_cuda_emits_full_path_bonus() -> None:
    device = torch.device("cuda")
    output, selected, num_selected = select_greedy_tree_path(
        torch.arange(31, dtype=torch.int64, device=device).view(1, 31),
        torch.arange(-1, 30, dtype=torch.int32, device=device),
        torch.arange(32, dtype=torch.int64, device=device),
    )

    assert output.cpu().tolist() == [list(range(32))]
    assert selected.cpu().tolist() == list(range(31))
    assert num_selected.item() == 31


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_kv_compactor_handles_shared_noncontiguous_layer_storage() -> None:
    device = torch.device("cuda")
    # Physical layout [block, layer, token, element] gives each logical layer
    # a non-contiguous [block, token, element] view, matching cross-layer MLA.
    backing = torch.empty((3, 2, 8, 4), dtype=torch.float32, device=device)
    caches = [backing[:, layer_idx] for layer_idx in range(2)]
    for layer_idx, cache in enumerate(caches):
        for slot in range(24):
            cache[slot // 8, slot % 8].fill_(layer_idx * 1000 + slot)

    query_slots = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int64, device=device)
    selected = torch.tensor([1, 3, -1, -1], dtype=torch.int32, device=device)
    num_selected = torch.tensor([2], dtype=torch.int32, device=device)
    compactor = KVCacheSlotCompactor(caches)

    assert compactor.num_storage_groups == 1
    compactor.compact(query_slots, selected, num_selected)
    torch.accelerator.synchronize()

    for layer_idx, cache in enumerate(caches):
        expected_base = layer_idx * 1000
        torch.testing.assert_close(
            cache[0, 3],
            torch.full((4,), expected_base + 4, device=device),
        )
        torch.testing.assert_close(
            cache[0, 4],
            torch.full((4,), expected_base + 6, device=device),
        )
