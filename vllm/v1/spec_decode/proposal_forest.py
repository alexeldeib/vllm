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
from collections.abc import Sequence
from dataclasses import dataclass


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
