# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU reference types for conditioned speculative proposal forests.

The serving hot path will eventually use device tensors, but its wire and
scheduler contract needs a small, deterministic implementation that can reject
stale or malformed work before any target-model kernel is launched.  This
module deliberately has no torch dependency so that tree construction,
selection, and rollback semantics can be tested without GPUs.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast


class ProposalForestValidationError(ValueError):
    """Raised when proposal metadata is unsafe to consume."""


def context_digest(token_ids: Sequence[int]) -> str:
    """Return a stable digest for the committed token context.

    Length-prefixing each integer avoids ambiguity and supports vocabularies
    larger than 16 bits without depending on platform byte order.
    """

    digest = hashlib.blake2b(digest_size=16)
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError("context token IDs must be non-negative")
        encoded = str(token_id).encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProposalForest:
    """Topologically packed proposal nodes for one request epoch.

    ``parent_indices`` uses ``-1`` for children of the virtual committed root.
    Every other parent must precede its child.  A node's token is the proposed
    next token after its parent; ``proposal_probs`` is its proposal model's
    conditional probability at that parent.  Multiple roots therefore encode
    alternate first tokens, while deeper siblings encode conditioned branches.
    """

    token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]
    depths: tuple[int, ...]
    source_ids: tuple[int, ...]
    proposal_probs: tuple[float, ...]
    request_epoch: int
    context_hash: str

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)

    def validate(
        self,
        *,
        max_nodes: int,
        expected_epoch: int | None = None,
        expected_context_hash: str | None = None,
        vocab_size: int | None = None,
    ) -> ProposalForest:
        """Validate the complete boundary contract and return ``self``."""

        if max_nodes < 0:
            raise ProposalForestValidationError("max_nodes must be non-negative")
        if self.num_nodes > max_nodes:
            raise ProposalForestValidationError(
                f"proposal has {self.num_nodes} nodes, budget is {max_nodes}"
            )
        if self.request_epoch < 0:
            raise ProposalForestValidationError("request_epoch must be non-negative")
        if not self.context_hash:
            raise ProposalForestValidationError("context_hash must not be empty")
        if expected_epoch is not None and self.request_epoch != expected_epoch:
            raise ProposalForestValidationError(
                f"stale proposal epoch {self.request_epoch}; expected {expected_epoch}"
            )
        if (
            expected_context_hash is not None
            and self.context_hash != expected_context_hash
        ):
            raise ProposalForestValidationError("proposal context hash is stale")
        if vocab_size is not None and vocab_size <= 0:
            raise ProposalForestValidationError("vocab_size must be positive")

        fields = {
            "parent_indices": self.parent_indices,
            "depths": self.depths,
            "source_ids": self.source_ids,
            "proposal_probs": self.proposal_probs,
        }
        for name, values in fields.items():
            if len(values) != self.num_nodes:
                raise ProposalForestValidationError(
                    f"{name} has length {len(values)}, expected {self.num_nodes}"
                )

        sibling_tokens: dict[int, set[int]] = defaultdict(set)
        sibling_probability_mass: dict[int, float] = defaultdict(float)
        for node_idx, (
            token_id,
            parent_idx,
            depth,
            source_id,
            proposal_prob,
        ) in enumerate(
            zip(
                self.token_ids,
                self.parent_indices,
                self.depths,
                self.source_ids,
                self.proposal_probs,
                strict=True,
            )
        ):
            if token_id < 0 or (vocab_size is not None and token_id >= vocab_size):
                raise ProposalForestValidationError(
                    f"token_ids[{node_idx}]={token_id} is outside the vocabulary"
                )
            if parent_idx < -1 or parent_idx >= node_idx:
                raise ProposalForestValidationError(
                    f"parent_indices[{node_idx}]={parent_idx} must be -1 or precede "
                    "its child"
                )
            expected_depth = 0 if parent_idx == -1 else self.depths[parent_idx] + 1
            if depth != expected_depth:
                raise ProposalForestValidationError(
                    f"depths[{node_idx}]={depth}, expected {expected_depth}"
                )
            if source_id < 0:
                raise ProposalForestValidationError(
                    f"source_ids[{node_idx}] must be non-negative"
                )
            if not math.isfinite(proposal_prob) or not 0.0 <= proposal_prob <= 1.0:
                raise ProposalForestValidationError(
                    f"proposal_probs[{node_idx}] must be finite and in [0, 1]"
                )
            if token_id in sibling_tokens[parent_idx]:
                raise ProposalForestValidationError(
                    f"parent {parent_idx} has duplicate token {token_id}"
                )
            sibling_tokens[parent_idx].add(token_id)
            sibling_probability_mass[parent_idx] += proposal_prob

        for parent_idx, probability_mass in sibling_probability_mass.items():
            if probability_mass > 1.0 + 1e-6:
                raise ProposalForestValidationError(
                    f"children of parent {parent_idx} have probability mass "
                    f"{probability_mass:.8f} > 1"
                )
        return self

    def children(self, parent_idx: int) -> tuple[int, ...]:
        if parent_idx < -1 or parent_idx >= self.num_nodes:
            raise IndexError(f"invalid parent index {parent_idx}")
        return tuple(
            node_idx
            for node_idx, node_parent in enumerate(self.parent_indices)
            if node_parent == parent_idx
        )

    def paths(self) -> tuple[tuple[int, ...], ...]:
        """Return every root-to-leaf node-index path in packed order."""

        if self.num_nodes == 0:
            return ()
        child_counts = [0] * self.num_nodes
        for parent_idx in self.parent_indices:
            if parent_idx >= 0:
                child_counts[parent_idx] += 1
        paths: list[tuple[int, ...]] = []
        for leaf_idx, child_count in enumerate(child_counts):
            if child_count:
                continue
            path: list[int] = []
            node_idx = leaf_idx
            while node_idx >= 0:
                path.append(node_idx)
                node_idx = self.parent_indices[node_idx]
            paths.append(tuple(reversed(path)))
        return tuple(paths)

    def ancestor_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Return the inclusive ancestor mask consumed by tree attention."""

        rows: list[tuple[bool, ...]] = []
        for node_idx, parent_idx in enumerate(self.parent_indices):
            row = [False] * self.num_nodes
            row[node_idx] = True
            if parent_idx >= 0:
                parent_row = rows[parent_idx]
                for ancestor_idx, is_ancestor in enumerate(parent_row):
                    if is_ancestor:
                        row[ancestor_idx] = True
            rows.append(tuple(row))
        return tuple(rows)

    def packed_positions(self, context_length: int) -> tuple[int, ...]:
        """Map proposal depth to the target's absolute token positions."""

        if context_length < 0:
            raise ValueError("context_length must be non-negative")
        return tuple(context_length + depth for depth in self.depths)


@dataclass(frozen=True, slots=True)
class PathSelection:
    """A verified path followed by exactly one target bonus token."""

    node_indices: tuple[int, ...]
    token_ids: tuple[int, ...]

    @property
    def num_accepted_draft_tokens(self) -> int:
        return len(self.node_indices)

    @property
    def bonus_token_id(self) -> int:
        return self.token_ids[-1]


@dataclass(frozen=True, slots=True)
class DuplicatedPathBatch:
    """Root-to-leaf causal paths used as a tree-verifier correctness oracle.

    An ordinary causal target can score each path independently.  Shared tree
    nodes are intentionally duplicated, and their target rows must agree when
    collapsed back to the packed forest layout.  A future MLA tree kernel can
    be gated against this representation before duplicated prefixes are
    removed from the hot path.
    """

    node_paths: tuple[tuple[int, ...], ...]
    token_paths: tuple[tuple[int, ...], ...]
    num_nodes: int


def duplicate_leaf_paths(forest: ProposalForest) -> DuplicatedPathBatch:
    """Expand a packed forest into independent root-to-leaf causal paths."""

    node_paths = forest.paths()
    if not node_paths:
        node_paths = ((),)
    token_paths = tuple(
        tuple(forest.token_ids[node_idx] for node_idx in path) for path in node_paths
    )
    return DuplicatedPathBatch(
        node_paths=node_paths,
        token_paths=token_paths,
        num_nodes=forest.num_nodes,
    )


_T = TypeVar("_T")


def _collapse_duplicated_rows(
    batch: DuplicatedPathBatch,
    path_rows: Sequence[Sequence[_T]],
    *,
    rows_equal: Callable[[_T, _T], bool],
) -> tuple[_T, ...]:
    if len(path_rows) != len(batch.node_paths):
        raise ValueError(
            f"received {len(path_rows)} path results, expected {len(batch.node_paths)}"
        )
    missing = object()
    packed_rows: list[object] = [missing] * (batch.num_nodes + 1)
    for path_idx, (node_path, rows) in enumerate(
        zip(batch.node_paths, path_rows, strict=True)
    ):
        if len(rows) != len(node_path) + 1:
            raise ValueError(
                f"path {path_idx} has {len(rows)} target rows, expected "
                f"{len(node_path) + 1}"
            )
        assignments = ((0, rows[0]),) + tuple(
            (node_idx + 1, rows[path_position + 1])
            for path_position, node_idx in enumerate(node_path)
        )
        for packed_idx, row in assignments:
            existing = packed_rows[packed_idx]
            if existing is missing:
                packed_rows[packed_idx] = row
            elif not rows_equal(cast(_T, existing), row):
                raise ValueError(
                    f"duplicated target rows disagree at packed row {packed_idx}"
                )
    missing_rows = [
        row_idx for row_idx, row in enumerate(packed_rows) if row is missing
    ]
    if missing_rows:
        raise ValueError(f"duplicated paths did not cover packed rows {missing_rows}")
    return tuple(cast(_T, row) for row in packed_rows)


def collapse_duplicated_greedy_rows(
    batch: DuplicatedPathBatch,
    path_next_token_ids: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Collapse path-local target argmax rows to virtual-root-plus-node order."""

    return _collapse_duplicated_rows(
        batch,
        path_next_token_ids,
        rows_equal=lambda lhs, rhs: lhs == rhs,
    )


def collapse_duplicated_probability_rows(
    batch: DuplicatedPathBatch,
    path_probabilities: Sequence[Sequence[Sequence[float]]],
    *,
    abs_tol: float = 1e-7,
) -> tuple[tuple[float, ...], ...]:
    """Collapse path-local target distributions and check shared-prefix parity."""

    if not math.isfinite(abs_tol) or abs_tol < 0:
        raise ValueError("abs_tol must be finite and non-negative")
    normalized_rows = tuple(
        tuple(tuple(probabilities) for probabilities in path_rows)
        for path_rows in path_probabilities
    )

    def rows_equal(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
        return len(lhs) == len(rhs) and all(
            math.isclose(
                lhs_probability,
                rhs_probability,
                rel_tol=0.0,
                abs_tol=abs_tol,
            )
            for lhs_probability, rhs_probability in zip(lhs, rhs, strict=True)
        )

    return _collapse_duplicated_rows(
        batch,
        normalized_rows,
        rows_equal=rows_equal,
    )


def _target_row_index(parent_idx: int) -> int:
    """Map virtual root ``-1`` and node parents to target row indices."""

    return parent_idx + 1


def select_greedy_path(
    forest: ProposalForest,
    target_next_token_ids: Sequence[int],
) -> PathSelection:
    """Select the longest target-greedy path and append its bonus token.

    ``target_next_token_ids[0]`` is the target token after the committed
    context.  Row ``node_idx + 1`` is the target token after that proposal
    node.  The latter is valid only when the node's full ancestor path matches.
    """

    if len(target_next_token_ids) != forest.num_nodes + 1:
        raise ValueError(
            "target_next_token_ids must contain the virtual-root row plus one "
            "row per proposal node"
        )
    accepted_nodes: list[int] = []
    accepted_tokens: list[int] = []
    parent_idx = -1
    while True:
        target_token_id = target_next_token_ids[_target_row_index(parent_idx)]
        matching_child = next(
            (
                child_idx
                for child_idx in forest.children(parent_idx)
                if forest.token_ids[child_idx] == target_token_id
            ),
            None,
        )
        if matching_child is None:
            accepted_tokens.append(target_token_id)
            return PathSelection(tuple(accepted_nodes), tuple(accepted_tokens))
        accepted_nodes.append(matching_child)
        accepted_tokens.append(target_token_id)
        parent_idx = matching_child


def _validated_target_probabilities(
    forest: ProposalForest,
    target_probabilities: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if len(target_probabilities) != forest.num_nodes + 1:
        raise ValueError(
            "target_probabilities must contain the virtual-root row plus one "
            "row per proposal node"
        )
    vocab_size: int | None = None
    rows: list[tuple[float, ...]] = []
    for row_idx, probabilities in enumerate(target_probabilities):
        row = tuple(probabilities)
        if vocab_size is None:
            vocab_size = len(row)
            if vocab_size == 0:
                raise ValueError("target probability rows must not be empty")
        elif len(row) != vocab_size:
            raise ValueError("target probability rows must share one vocabulary")
        if any(
            not math.isfinite(probability) or probability < 0 for probability in row
        ):
            raise ValueError(f"target probability row {row_idx} is invalid")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"target probability row {row_idx} must sum to one")
        rows.append(row)
    return tuple(rows)


def _sample_categorical(probabilities: Sequence[float], uniform: float) -> int:
    if not 0.0 <= uniform < 1.0:
        raise ValueError("uniform samples must be in [0, 1)")
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if uniform < cumulative:
            return token_id
    return len(probabilities) - 1


def select_stochastic_path(
    forest: ProposalForest,
    target_probabilities: Sequence[Sequence[float]],
    uniforms: Sequence[float],
) -> PathSelection:
    """Sample an exact target-distribution path through a verified forest.

    At each selected parent, sample directly from the target distribution.  A
    matching proposed child lets verification continue; otherwise the sampled
    token is the bonus token and decoding falls back to the normal target
    step.  Because only distributions conditioned on the selected ancestor
    path are consumed, this preserves the target's autoregressive distribution
    without relying on approximate draft acceptance.
    """

    rows = _validated_target_probabilities(forest, target_probabilities)
    accepted_nodes: list[int] = []
    accepted_tokens: list[int] = []
    parent_idx = -1
    for uniform in uniforms:
        target_token_id = _sample_categorical(
            rows[_target_row_index(parent_idx)], uniform
        )
        matching_child = next(
            (
                child_idx
                for child_idx in forest.children(parent_idx)
                if forest.token_ids[child_idx] == target_token_id
            ),
            None,
        )
        if matching_child is None:
            accepted_tokens.append(target_token_id)
            return PathSelection(tuple(accepted_nodes), tuple(accepted_tokens))
        accepted_nodes.append(matching_child)
        accepted_tokens.append(target_token_id)
        parent_idx = matching_child
    raise ValueError("uniforms ended before a target bonus token was sampled")


def enumerate_target_path_distribution(
    forest: ProposalForest,
    target_probabilities: Sequence[Sequence[float]],
) -> dict[tuple[int, ...], float]:
    """Enumerate the exact emitted-sequence distribution for a tiny forest."""

    rows = _validated_target_probabilities(forest, target_probabilities)
    result: dict[tuple[int, ...], float] = defaultdict(float)

    def visit(parent_idx: int, prefix: tuple[int, ...], prefix_prob: float) -> None:
        children_by_token = {
            forest.token_ids[child_idx]: child_idx
            for child_idx in forest.children(parent_idx)
        }
        for token_id, token_prob in enumerate(rows[_target_row_index(parent_idx)]):
            if token_prob == 0.0:
                continue
            emitted = (*prefix, token_id)
            child_idx = children_by_token.get(token_id)
            if child_idx is None:
                result[emitted] += prefix_prob * token_prob
            else:
                visit(child_idx, emitted, prefix_prob * token_prob)

    visit(-1, (), 1.0)
    return dict(result)


class ProposalForestBuffer:
    """Bounded per-request buffer for current and look-ahead proposal epochs."""

    def __init__(self, *, max_nodes: int, max_pending_per_request: int = 2) -> None:
        if max_nodes < 0:
            raise ValueError("max_nodes must be non-negative")
        if max_pending_per_request <= 0:
            raise ValueError("max_pending_per_request must be positive")
        self.max_nodes = max_nodes
        self.max_pending_per_request = max_pending_per_request
        self._pending: dict[str, dict[int, ProposalForest]] = {}

    def publish(self, request_id: str, forest: ProposalForest) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        forest.validate(max_nodes=self.max_nodes)
        request_pending = self._pending.setdefault(request_id, {})
        if forest.request_epoch in request_pending:
            raise ProposalForestValidationError(
                f"request {request_id} already has epoch {forest.request_epoch}"
            )
        if len(request_pending) >= self.max_pending_per_request:
            raise ProposalForestValidationError(
                f"request {request_id} proposal buffer is full"
            )
        request_pending[forest.request_epoch] = forest

    def take(
        self,
        request_id: str,
        *,
        expected_epoch: int,
        expected_context_hash: str,
    ) -> ProposalForest | None:
        """Consume an exact epoch or return ``None`` for target-only fallback."""

        request_pending = self._pending.get(request_id)
        if not request_pending:
            return None
        for stale_epoch in tuple(
            epoch for epoch in request_pending if epoch < expected_epoch
        ):
            del request_pending[stale_epoch]
        forest = request_pending.pop(expected_epoch, None)
        if not request_pending:
            self._pending.pop(request_id, None)
        if forest is None:
            return None
        return forest.validate(
            max_nodes=self.max_nodes,
            expected_epoch=expected_epoch,
            expected_context_hash=expected_context_hash,
        )

    def cancel(self, request_id: str) -> None:
        self._pending.pop(request_id, None)

    def pending_epochs(self, request_id: str) -> tuple[int, ...]:
        """Return buffered epochs for capacity checks and diagnostics."""

        return tuple(sorted(self._pending.get(request_id, ())))


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """One asynchronous current-epoch or look-ahead proposal request."""

    request_id: str
    request_epoch: int
    context_hash: str
    max_nodes: int
    generation: int
    submitted_at: float
    deadline: float


@dataclass(frozen=True, slots=True)
class ProposalCompletion:
    """A proposal service response tied to the submitted request generation."""

    request: ProposalRequest
    forest: ProposalForest


class ProposalService(Protocol):
    """Minimal transport-independent interface used by the CPU coordinator."""

    def submit(self, request: ProposalRequest) -> None: ...

    def poll(self, now: float) -> tuple[ProposalCompletion, ...]: ...

    def cancel(
        self,
        request_id: str,
        *,
        request_epoch: int | None = None,
        generation: int | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _PlannedCompletion:
    delay: float
    forest: ProposalForest


@dataclass(frozen=True, slots=True)
class _PendingCompletion:
    ready_at: float
    completion: ProposalCompletion


class FakeProposalService:
    """Deterministic logical-clock service for scheduler correctness tests.

    Plans are keyed by request ID, epoch, and context hash.  A request with no
    plan intentionally remains silent, which models a stalled remote drafter.
    Setting ``honor_cancellation=False`` models a transport where a cancelled
    response can still arrive and must be rejected by generation metadata.
    """

    def __init__(self, *, honor_cancellation: bool = True) -> None:
        self.honor_cancellation = honor_cancellation
        self._plans: dict[tuple[str, int, str], list[_PlannedCompletion]] = defaultdict(
            list
        )
        self._pending: list[_PendingCompletion] = []

    def plan(
        self,
        request_id: str,
        forest: ProposalForest,
        *,
        delay: float = 0.0,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("delay must be finite and non-negative")
        key = (request_id, forest.request_epoch, forest.context_hash)
        self._plans[key].append(_PlannedCompletion(delay=delay, forest=forest))

    def submit(self, request: ProposalRequest) -> None:
        key = (request.request_id, request.request_epoch, request.context_hash)
        plans = self._plans.get(key)
        if not plans:
            return
        plan = plans.pop(0)
        if not plans:
            self._plans.pop(key, None)
        self._pending.append(
            _PendingCompletion(
                ready_at=request.submitted_at + plan.delay,
                completion=ProposalCompletion(request=request, forest=plan.forest),
            )
        )

    def poll(self, now: float) -> tuple[ProposalCompletion, ...]:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        ready = sorted(
            (pending for pending in self._pending if pending.ready_at <= now),
            key=lambda pending: (
                pending.ready_at,
                pending.completion.request.request_epoch,
            ),
        )
        if not ready:
            return ()
        ready_ids = {id(pending) for pending in ready}
        self._pending = [
            pending for pending in self._pending if id(pending) not in ready_ids
        ]
        return tuple(pending.completion for pending in ready)

    def cancel(
        self,
        request_id: str,
        *,
        request_epoch: int | None = None,
        generation: int | None = None,
    ) -> None:
        if not self.honor_cancellation:
            return

        def keep(pending: _PendingCompletion) -> bool:
            request = pending.completion.request
            if request.request_id != request_id:
                return True
            if request_epoch is not None and request.request_epoch != request_epoch:
                return True
            return generation is not None and request.generation != generation

        self._pending = [pending for pending in self._pending if keep(pending)]


class ProposalForestCoordinator:
    """Fail-closed double buffer for current and look-ahead proposal epochs.

    The coordinator never waits for proposal work.  At the target scheduling
    boundary, ``take_or_fallback`` either returns a fully validated forest for
    the exact request epoch and context, or ``None`` so normal one-token target
    decoding can proceed.  Rollback increments a per-request generation so a
    late response remains stale even when the transport cannot cancel it.
    """

    def __init__(
        self,
        service: ProposalService,
        *,
        max_nodes: int,
        max_inflight_per_request: int = 2,
    ) -> None:
        if max_nodes < 0:
            raise ValueError("max_nodes must be non-negative")
        if max_inflight_per_request <= 0:
            raise ValueError("max_inflight_per_request must be positive")
        self.service = service
        self.max_nodes = max_nodes
        self.max_inflight_per_request = max_inflight_per_request
        self.buffer = ProposalForestBuffer(
            max_nodes=max_nodes,
            max_pending_per_request=max_inflight_per_request,
        )
        self._generations: dict[str, int] = defaultdict(int)
        self._inflight: dict[tuple[str, int], ProposalRequest] = {}
        self.rejected_completions = 0
        self.stale_completions = 0
        self.timed_out_requests = 0

    @property
    def num_inflight(self) -> int:
        return len(self._inflight)

    def submit(
        self,
        request_id: str,
        *,
        request_epoch: int,
        context_hash: str,
        now: float,
        timeout: float,
    ) -> ProposalRequest:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_epoch < 0:
            raise ValueError("request_epoch must be non-negative")
        if not context_hash:
            raise ValueError("context_hash must not be empty")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        key = (request_id, request_epoch)
        if key in self._inflight:
            raise ValueError(
                f"request {request_id} epoch {request_epoch} is already in flight"
            )
        pending_epochs = self.buffer.pending_epochs(request_id)
        if request_epoch in pending_epochs:
            raise ValueError(
                f"request {request_id} epoch {request_epoch} is already buffered"
            )
        inflight_count = sum(
            request.request_id == request_id for request in self._inflight.values()
        )
        if inflight_count + len(pending_epochs) >= self.max_inflight_per_request:
            raise ValueError(f"request {request_id} proposal window is full")
        request = ProposalRequest(
            request_id=request_id,
            request_epoch=request_epoch,
            context_hash=context_hash,
            max_nodes=self.max_nodes,
            generation=self._generations[request_id],
            submitted_at=now,
            deadline=now + timeout,
        )
        self._inflight[key] = request
        try:
            self.service.submit(request)
        except Exception:
            del self._inflight[key]
            raise
        return request

    def poll(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        for completion in self.service.poll(now):
            completed_request = completion.request
            key = (completed_request.request_id, completed_request.request_epoch)
            expected_request = self._inflight.get(key)
            if expected_request != completed_request:
                self.stale_completions += 1
                continue
            del self._inflight[key]
            if now > expected_request.deadline:
                self.timed_out_requests += 1
                continue
            try:
                completion.forest.validate(
                    max_nodes=expected_request.max_nodes,
                    expected_epoch=expected_request.request_epoch,
                    expected_context_hash=expected_request.context_hash,
                )
                self.buffer.publish(
                    expected_request.request_id,
                    completion.forest,
                )
            except ProposalForestValidationError:
                self.rejected_completions += 1

        for key, request in tuple(self._inflight.items()):
            if now < request.deadline:
                continue
            del self._inflight[key]
            self.timed_out_requests += 1
            self.service.cancel(
                request.request_id,
                request_epoch=request.request_epoch,
                generation=request.generation,
            )

    def take_or_fallback(
        self,
        request_id: str,
        *,
        expected_epoch: int,
        expected_context_hash: str,
        now: float,
    ) -> ProposalForest | None:
        """Return ready work or cancel this epoch and let the target proceed."""

        self.poll(now)
        try:
            forest = self.buffer.take(
                request_id,
                expected_epoch=expected_epoch,
                expected_context_hash=expected_context_hash,
            )
        except ProposalForestValidationError:
            self.rejected_completions += 1
            forest = None
        if forest is not None:
            return forest

        key = (request_id, expected_epoch)
        request = self._inflight.pop(key, None)
        if request is not None:
            self.service.cancel(
                request_id,
                request_epoch=expected_epoch,
                generation=request.generation,
            )
        return None

    def cancel_request(self, request_id: str) -> None:
        """Cancel all work and invalidate late results for target rollback."""

        if not request_id:
            raise ValueError("request_id must not be empty")
        generation = self._generations[request_id]
        self._generations[request_id] = generation + 1
        self.buffer.cancel(request_id)
        for key, request in tuple(self._inflight.items()):
            if request.request_id == request_id:
                del self._inflight[key]
        self.service.cancel(request_id, generation=generation)
