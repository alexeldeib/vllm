# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import random
from collections import Counter

import pytest

from vllm.v1.spec_decode.proposal_forest import (
    FakeProposalService,
    LinearProposal,
    ProposalForest,
    ProposalForestBuffer,
    ProposalForestCoordinator,
    ProposalForestValidationError,
    collapse_duplicated_greedy_rows,
    collapse_duplicated_probability_rows,
    context_digest,
    duplicate_leaf_paths,
    enumerate_target_path_distribution,
    merge_linear_proposals,
    select_greedy_path,
    select_stochastic_path,
)


def make_forest(**overrides) -> ProposalForest:
    fields = {
        "token_ids": (10, 20, 11, 21),
        "parent_indices": (-1, -1, 0, 1),
        "depths": (0, 0, 1, 1),
        "source_ids": (0, 1, 0, 1),
        "proposal_probs": (0.55, 0.35, 0.7, 0.8),
        "request_epoch": 4,
        "context_hash": context_digest((3, 5, 8)),
    }
    fields.update(overrides)
    return ProposalForest(**fields)


def test_validate_builds_positions_paths_and_ancestor_mask() -> None:
    forest = make_forest().validate(max_nodes=8, vocab_size=32)

    assert forest.paths() == ((0, 2), (1, 3))
    assert forest.packed_positions(100) == (100, 100, 101, 101)
    assert forest.ancestor_mask() == (
        (True, False, False, False),
        (False, True, False, False),
        (True, False, True, False),
        (False, True, False, True),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parent_indices": (-1,)}, "parent_indices has length"),
        ({"token_ids": (10, -1, 11, 21)}, "outside the vocabulary"),
        ({"parent_indices": (-1, 3, 0, 1)}, "must be -1 or precede"),
        ({"depths": (0, 0, 2, 1)}, "expected 1"),
        ({"source_ids": (0, -1, 0, 1)}, "must be non-negative"),
        ({"proposal_probs": (0.55, float("nan"), 0.7, 0.8)}, "must be finite"),
        (
            {
                "token_ids": (10, 10, 11, 21),
                "proposal_probs": (0.5, 0.4, 0.7, 0.8),
            },
            "duplicate token",
        ),
        ({"proposal_probs": (0.75, 0.35, 0.7, 0.8)}, "probability mass"),
    ],
)
def test_validate_rejects_malformed_forests(overrides, message: str) -> None:
    with pytest.raises(ProposalForestValidationError, match=message):
        make_forest(**overrides).validate(max_nodes=8, vocab_size=32)


def test_validate_rejects_budget_epoch_and_context_mismatches() -> None:
    forest = make_forest()
    with pytest.raises(ProposalForestValidationError, match="budget is 3"):
        forest.validate(max_nodes=3)
    with pytest.raises(ProposalForestValidationError, match="stale proposal epoch"):
        forest.validate(max_nodes=8, expected_epoch=5)
    with pytest.raises(ProposalForestValidationError, match="context hash is stale"):
        forest.validate(max_nodes=8, expected_context_hash="different")


def test_merge_linear_sources_deduplicates_prefixes_and_preserves_mixture() -> None:
    forest = merge_linear_proposals(
        (
            LinearProposal(
                token_ids=(10, 11, 12),
                conditional_probs=(0.8, 0.5, 0.5),
                source_id=0,
                mixture_weight=0.6,
            ),
            LinearProposal(
                token_ids=(10, 13),
                conditional_probs=(0.5, 0.8),
                source_id=1,
                mixture_weight=0.4,
            ),
        ),
        max_nodes=8,
        request_epoch=7,
        context_hash=context_digest((3, 5, 8)),
    )

    assert forest.token_ids == (10, 11, 13, 12)
    assert forest.parent_indices == (-1, 0, 0, 1)
    assert forest.depths == (0, 1, 1, 2)
    assert forest.source_ids == (0, 0, 1, 0)
    assert forest.proposal_probs == pytest.approx((0.68, 0.24 / 0.68, 0.16 / 0.68, 0.5))
    assert forest.paths() == ((0, 2), (0, 1, 3))


def test_merge_linear_sources_spends_budget_on_highest_path_mass() -> None:
    forest = merge_linear_proposals(
        (
            LinearProposal(
                token_ids=(1, 2, 4),
                conditional_probs=(1.0, 1.0, 1.0),
                source_id=0,
                mixture_weight=0.45,
            ),
            LinearProposal(
                token_ids=(3,),
                conditional_probs=(1.0,),
                source_id=1,
                mixture_weight=0.44,
            ),
        ),
        max_nodes=2,
        request_epoch=0,
        context_hash=context_digest((9,)),
    )

    # The second node on source 0 carries 0.45 mixture path mass, slightly more
    # than source 1's alternate root.
    assert forest.token_ids == (1, 2)
    assert forest.parent_indices == (-1, 0)
    assert forest.proposal_probs == pytest.approx((0.45, 1.0))


def test_merge_linear_sources_aggregates_duplicate_components() -> None:
    forest = merge_linear_proposals(
        (
            LinearProposal((7, 8), (0.5, 0.5), 2, 0.4),
            LinearProposal((7, 8), (1.0, 0.25), 1, 0.3),
        ),
        max_nodes=4,
        request_epoch=0,
        context_hash=context_digest((9,)),
    )

    assert forest.token_ids == (7, 8)
    assert forest.parent_indices == (-1, 0)
    assert forest.source_ids == (1, 2)
    assert forest.proposal_probs == pytest.approx((0.5, 0.175 / 0.5))


def test_merge_linear_sources_accepts_empty_or_zero_budget() -> None:
    empty = merge_linear_proposals(
        (),
        max_nodes=8,
        request_epoch=0,
        context_hash=context_digest((9,)),
    )
    zero_budget = merge_linear_proposals(
        (LinearProposal((7,), (1.0,), 0, 1.0),),
        max_nodes=0,
        request_epoch=0,
        context_hash=context_digest((9,)),
    )

    assert empty.num_nodes == 0
    assert zero_budget.num_nodes == 0


@pytest.mark.parametrize(
    ("proposals", "message"),
    [
        ((LinearProposal((1,), (), 0, 1.0),), "lengths differ"),
        ((LinearProposal((1,), (1.1,), 0, 1.0),), "probability 0"),
        ((LinearProposal((1,), (1.0,), -1, 1.0),), "source_id"),
        ((LinearProposal((1,), (1.0,), 0, float("nan")),), "mixture_weight"),
        (
            (
                LinearProposal((1,), (1.0,), 0, 0.6),
                LinearProposal((2,), (1.0,), 1, 0.5),
            ),
            "weights sum",
        ),
    ],
)
def test_merge_linear_sources_rejects_invalid_components(
    proposals: tuple[LinearProposal, ...], message: str
) -> None:
    with pytest.raises(ProposalForestValidationError, match=message):
        merge_linear_proposals(
            proposals,
            max_nodes=8,
            request_epoch=0,
            context_hash=context_digest((9,)),
        )


def test_greedy_selection_follows_conditioned_branch_and_appends_bonus() -> None:
    forest = make_forest().validate(max_nodes=8)
    # virtual root -> token 20/node 1 -> token 21/node 3 -> bonus token 7
    target_next_token_ids = (20, 99, 21, 98, 7)

    selected = select_greedy_path(forest, target_next_token_ids)

    assert selected.node_indices == (1, 3)
    assert selected.token_ids == (20, 21, 7)
    assert selected.num_accepted_draft_tokens == 2
    assert selected.bonus_token_id == 7


def test_duplicated_causal_paths_match_packed_greedy_selection() -> None:
    forest = make_forest().validate(max_nodes=8)
    batch = duplicate_leaf_paths(forest)

    assert batch.node_paths == ((0, 2), (1, 3))
    assert batch.token_paths == ((10, 11), (20, 21))

    # Each path has one virtual-root row plus one row per draft token.  The
    # duplicated virtual-root row agrees, while later rows are path-local.
    path_next_token_ids = (
        (20, 99, 7),
        (20, 21, 98),
    )
    packed_next_token_ids = collapse_duplicated_greedy_rows(batch, path_next_token_ids)

    assert packed_next_token_ids == (20, 99, 21, 7, 98)
    selected = select_greedy_path(forest, packed_next_token_ids)
    assert selected.node_indices == (1, 3)
    assert selected.token_ids == (20, 21, 98)


def test_duplicated_paths_reject_shared_prefix_divergence() -> None:
    forest = ProposalForest(
        token_ids=(10, 11, 12),
        parent_indices=(-1, 0, 0),
        depths=(0, 1, 1),
        source_ids=(0, 0, 1),
        proposal_probs=(0.6, 0.3, 0.2),
        request_epoch=0,
        context_hash=context_digest((9,)),
    ).validate(max_nodes=8)
    batch = duplicate_leaf_paths(forest)

    assert batch.node_paths == ((0, 1), (0, 2))
    with pytest.raises(ValueError, match="disagree at packed row 1"):
        collapse_duplicated_greedy_rows(
            batch,
            (
                (10, 21, 31),
                (10, 22, 32),
            ),
        )


def test_duplicated_probability_rows_match_packed_stochastic_oracle() -> None:
    forest = ProposalForest(
        token_ids=(0, 1, 0),
        parent_indices=(-1, -1, 0),
        depths=(0, 0, 1),
        source_ids=(0, 0, 0),
        proposal_probs=(0.6, 0.3, 0.2),
        request_epoch=0,
        context_hash=context_digest((9,)),
    ).validate(max_nodes=8, vocab_size=3)
    packed_probabilities = (
        (0.6, 0.3, 0.1),
        (0.2, 0.5, 0.3),
        (0.4, 0.4, 0.2),
        (0.1, 0.1, 0.8),
    )
    batch = duplicate_leaf_paths(forest)
    path_probabilities = tuple(
        (packed_probabilities[0],)
        + tuple(packed_probabilities[node_idx + 1] for node_idx in node_path)
        for node_path in batch.node_paths
    )

    collapsed = collapse_duplicated_probability_rows(batch, path_probabilities)

    assert collapsed == packed_probabilities
    assert enumerate_target_path_distribution(
        forest, collapsed
    ) == enumerate_target_path_distribution(forest, packed_probabilities)


def test_duplicated_empty_forest_keeps_virtual_root_row() -> None:
    forest = make_forest(
        token_ids=(),
        parent_indices=(),
        depths=(),
        source_ids=(),
        proposal_probs=(),
    ).validate(max_nodes=8)
    batch = duplicate_leaf_paths(forest)

    assert batch.node_paths == ((),)
    assert batch.token_paths == ((),)
    assert collapse_duplicated_greedy_rows(batch, ((6,),)) == (6,)
    assert collapse_duplicated_probability_rows(batch, (((0.25, 0.75),),)) == (
        (0.25, 0.75),
    )


@pytest.mark.parametrize(
    ("path_rows", "message"),
    [
        (((20, 99, 7),), "1 path results, expected 2"),
        (((20, 99), (20, 21, 98)), "path 0 has 2 target rows, expected 3"),
    ],
)
def test_duplicated_paths_reject_malformed_target_shapes(
    path_rows: tuple[tuple[int, ...], ...], message: str
) -> None:
    batch = duplicate_leaf_paths(make_forest().validate(max_nodes=8))

    with pytest.raises(ValueError, match=message):
        collapse_duplicated_greedy_rows(batch, path_rows)


def test_empty_forest_falls_back_to_one_target_token() -> None:
    forest = make_forest(
        token_ids=(),
        parent_indices=(),
        depths=(),
        source_ids=(),
        proposal_probs=(),
    ).validate(max_nodes=8)

    assert select_greedy_path(forest, (6,)).token_ids == (6,)
    assert select_stochastic_path(forest, ((0.25, 0.75),), (0.9,)).token_ids == (1,)


def test_exact_stochastic_distribution_matches_manual_enumeration() -> None:
    forest = ProposalForest(
        token_ids=(0, 1, 0),
        parent_indices=(-1, -1, 0),
        depths=(0, 0, 1),
        source_ids=(0, 0, 0),
        proposal_probs=(0.6, 0.3, 0.2),
        request_epoch=0,
        context_hash=context_digest((9,)),
    ).validate(max_nodes=8, vocab_size=3)
    target_probabilities = (
        (0.6, 0.3, 0.1),  # after committed context
        (0.2, 0.5, 0.3),  # after root token 0
        (0.4, 0.4, 0.2),  # after root token 1
        (0.1, 0.1, 0.8),  # after path 0 -> 0
    )
    expected = {
        (2,): 0.1,
        (0, 1): 0.3,
        (0, 2): 0.18,
        (0, 0, 0): 0.012,
        (0, 0, 1): 0.012,
        (0, 0, 2): 0.096,
        (1, 0): 0.12,
        (1, 1): 0.12,
        (1, 2): 0.06,
    }

    enumerated = enumerate_target_path_distribution(forest, target_probabilities)

    assert enumerated == pytest.approx(expected)
    assert sum(enumerated.values()) == pytest.approx(1.0)

    rng = random.Random(42)
    samples = Counter(
        select_stochastic_path(
            forest,
            target_probabilities,
            (rng.random(), rng.random(), rng.random(), rng.random()),
        ).token_ids
        for _ in range(50_000)
    )
    for sequence, probability in expected.items():
        assert samples[sequence] / 50_000 == pytest.approx(probability, abs=0.006)


def test_buffer_drops_stale_epochs_and_keeps_future_lookahead() -> None:
    buffer = ProposalForestBuffer(max_nodes=8, max_pending_per_request=3)
    epoch4 = make_forest(request_epoch=4)
    epoch5 = make_forest(request_epoch=5)
    epoch6 = make_forest(request_epoch=6)
    buffer.publish("req", epoch4)
    buffer.publish("req", epoch5)
    buffer.publish("req", epoch6)

    selected = buffer.take(
        "req",
        expected_epoch=5,
        expected_context_hash=epoch5.context_hash,
    )

    assert selected is epoch5
    selected_lookahead = buffer.take(
        "req",
        expected_epoch=6,
        expected_context_hash=epoch6.context_hash,
    )
    assert selected_lookahead is epoch6


def test_buffer_rejects_wrong_context_and_cancellation_falls_back() -> None:
    buffer = ProposalForestBuffer(max_nodes=8)
    buffer.publish("req", make_forest())
    with pytest.raises(ProposalForestValidationError, match="context hash is stale"):
        buffer.take("req", expected_epoch=4, expected_context_hash="wrong")

    buffer.publish("req", make_forest(request_epoch=5))
    buffer.cancel("req")
    assert (
        buffer.take(
            "req",
            expected_epoch=5,
            expected_context_hash=make_forest().context_hash,
        )
        is None
    )


def test_coordinator_double_buffers_current_and_lookahead_epochs() -> None:
    service = FakeProposalService()
    coordinator = ProposalForestCoordinator(service, max_nodes=8)
    epoch4 = make_forest(request_epoch=4)
    epoch5 = make_forest(request_epoch=5)
    service.plan("req", epoch4)
    service.plan("req", epoch5, delay=1.0)

    coordinator.submit(
        "req",
        request_epoch=4,
        context_hash=epoch4.context_hash,
        now=0.0,
        timeout=2.0,
    )
    coordinator.submit(
        "req",
        request_epoch=5,
        context_hash=epoch5.context_hash,
        now=0.0,
        timeout=2.0,
    )

    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=4,
            expected_context_hash=epoch4.context_hash,
            now=0.0,
        )
        is epoch4
    )
    assert coordinator.num_inflight == 1
    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=5,
            expected_context_hash=epoch5.context_hash,
            now=1.0,
        )
        is epoch5
    )
    assert coordinator.num_inflight == 0


def test_slow_current_epoch_times_out_without_discarding_ready_lookahead() -> None:
    service = FakeProposalService(honor_cancellation=False)
    coordinator = ProposalForestCoordinator(service, max_nodes=8)
    current = make_forest(request_epoch=4)
    lookahead = make_forest(request_epoch=5)
    service.plan("req", current, delay=3.0)
    service.plan("req", lookahead, delay=0.5)
    coordinator.submit(
        "req",
        request_epoch=4,
        context_hash=current.context_hash,
        now=0.0,
        timeout=1.0,
    )
    coordinator.submit(
        "req",
        request_epoch=5,
        context_hash=lookahead.context_hash,
        now=0.0,
        timeout=2.0,
    )

    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=4,
            expected_context_hash=current.context_hash,
            now=1.0,
        )
        is None
    )
    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=5,
            expected_context_hash=lookahead.context_hash,
            now=1.0,
        )
        is lookahead
    )
    assert coordinator.timed_out_requests == 1

    coordinator.poll(3.0)
    assert coordinator.stale_completions == 1


def test_target_rollback_rejects_late_generation_and_accepts_replacement() -> None:
    service = FakeProposalService(honor_cancellation=False)
    coordinator = ProposalForestCoordinator(service, max_nodes=8)
    old = make_forest(request_epoch=4)
    replacement = make_forest(
        token_ids=(12, 20, 13, 21),
        request_epoch=4,
    )
    service.plan("req", old, delay=2.0)
    first_request = coordinator.submit(
        "req",
        request_epoch=4,
        context_hash=old.context_hash,
        now=0.0,
        timeout=3.0,
    )

    coordinator.cancel_request("req")
    service.plan("req", replacement)
    replacement_request = coordinator.submit(
        "req",
        request_epoch=4,
        context_hash=replacement.context_hash,
        now=0.0,
        timeout=3.0,
    )

    assert replacement_request.generation == first_request.generation + 1
    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=4,
            expected_context_hash=replacement.context_hash,
            now=0.0,
        )
        is replacement
    )
    coordinator.poll(2.0)
    assert coordinator.stale_completions == 1


def test_partial_and_empty_completions_are_valid_scheduler_inputs() -> None:
    service = FakeProposalService()
    coordinator = ProposalForestCoordinator(service, max_nodes=8)
    partial = make_forest(
        token_ids=(10,),
        parent_indices=(-1,),
        depths=(0,),
        source_ids=(0,),
        proposal_probs=(0.55,),
        request_epoch=4,
    )
    empty = make_forest(
        token_ids=(),
        parent_indices=(),
        depths=(),
        source_ids=(),
        proposal_probs=(),
        request_epoch=5,
    )
    service.plan("req", partial)
    service.plan("req", empty)
    coordinator.submit(
        "req",
        request_epoch=4,
        context_hash=partial.context_hash,
        now=0.0,
        timeout=1.0,
    )
    coordinator.submit(
        "req",
        request_epoch=5,
        context_hash=empty.context_hash,
        now=0.0,
        timeout=1.0,
    )

    selected_partial = coordinator.take_or_fallback(
        "req",
        expected_epoch=4,
        expected_context_hash=partial.context_hash,
        now=0.0,
    )
    selected_empty = coordinator.take_or_fallback(
        "req",
        expected_epoch=5,
        expected_context_hash=empty.context_hash,
        now=0.0,
    )

    assert selected_partial is partial
    assert selected_empty is empty
    assert select_greedy_path(selected_empty, (6,)).token_ids == (6,)


def test_malformed_or_wrong_context_completion_falls_back() -> None:
    service = FakeProposalService()
    coordinator = ProposalForestCoordinator(service, max_nodes=8)
    malformed = make_forest(
        token_ids=(10, 10, 11, 21),
        proposal_probs=(0.5, 0.4, 0.7, 0.8),
    )
    service.plan("req", malformed)
    coordinator.submit(
        "req",
        request_epoch=4,
        context_hash=malformed.context_hash,
        now=0.0,
        timeout=1.0,
    )

    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=4,
            expected_context_hash=malformed.context_hash,
            now=0.0,
        )
        is None
    )
    assert coordinator.rejected_completions == 1

    valid = make_forest(request_epoch=5)
    service.plan("req", valid)
    coordinator.submit(
        "req",
        request_epoch=5,
        context_hash=valid.context_hash,
        now=1.0,
        timeout=1.0,
    )
    assert (
        coordinator.take_or_fallback(
            "req",
            expected_epoch=5,
            expected_context_hash="rolled-back-context",
            now=1.0,
        )
        is None
    )
    assert coordinator.rejected_completions == 2


def test_coordinator_bounds_current_and_lookahead_window() -> None:
    service = FakeProposalService()
    coordinator = ProposalForestCoordinator(
        service,
        max_nodes=8,
        max_inflight_per_request=2,
    )
    for epoch in (4, 5):
        coordinator.submit(
            "req",
            request_epoch=epoch,
            context_hash=make_forest().context_hash,
            now=0.0,
            timeout=10.0,
        )

    with pytest.raises(ValueError, match="proposal window is full"):
        coordinator.submit(
            "req",
            request_epoch=6,
            context_hash=make_forest().context_hash,
            now=0.0,
            timeout=10.0,
        )


def test_ready_forests_still_count_against_double_buffer_capacity() -> None:
    service = FakeProposalService()
    coordinator = ProposalForestCoordinator(service, max_nodes=8)
    for epoch in (4, 5):
        forest = make_forest(request_epoch=epoch)
        service.plan("req", forest)
        coordinator.submit(
            "req",
            request_epoch=epoch,
            context_hash=forest.context_hash,
            now=0.0,
            timeout=10.0,
        )
    coordinator.poll(0.0)
    assert coordinator.num_inflight == 0

    with pytest.raises(ValueError, match="proposal window is full"):
        coordinator.submit(
            "req",
            request_epoch=6,
            context_hash=make_forest().context_hash,
            now=0.0,
            timeout=10.0,
        )

    coordinator.take_or_fallback(
        "req",
        expected_epoch=4,
        expected_context_hash=make_forest().context_hash,
        now=0.0,
    )
    coordinator.submit(
        "req",
        request_epoch=6,
        context_hash=make_forest().context_hash,
        now=0.0,
        timeout=10.0,
    )
