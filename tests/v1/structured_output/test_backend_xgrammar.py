# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``XgrammarGrammar`` terminated-state safety under spec decode.

Speculative decoding builds a per-position grammar bitmask by repeatedly
advancing the matcher through scheduled draft tokens and then rolling back
(``StructuredOutputManager.grammar_bitmask``), and it filters draft tokens via
``validate_tokens``. xgrammar does not reliably clear the terminated flag on
``rollback`` (vllm-project/vllm#27210), so ``matcher.is_terminated()`` cannot be
trusted after a rollback. If the grammar gets wedged "terminated" every later
``fill_bitmask`` emits an all-allowed mask (silent unconstrained output) and the
next real ``accept_tokens`` returns False ("grammar rejected tokens ...
Terminating request"). These tests cover the position-tracked termination logic
that keeps the flag correct across accept-then-rollback sequences.
"""

from unittest.mock import MagicMock

import pytest

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

pytestmark = pytest.mark.cpu_test


def _grammar(matcher: MagicMock, terminated: bool = False) -> XgrammarGrammar:
    grammar = object.__new__(XgrammarGrammar)
    grammar.matcher = matcher
    grammar.num_processed_tokens = 0
    grammar._is_terminated = terminated
    grammar._terminated_at = None
    return grammar


# ---------------------------------------------------------------------------
# validate_tokens (draft-token filtering)
# ---------------------------------------------------------------------------
def test_validate_tokens_skips_already_terminated_matcher():
    matcher = MagicMock()
    grammar = _grammar(matcher, terminated=True)

    assert grammar.validate_tokens([1, 2, 3]) == []
    matcher.accept_token.assert_not_called()
    matcher.rollback.assert_not_called()


def test_validate_tokens_stops_once_matcher_terminates_mid_loop():
    matcher = MagicMock()
    # Not terminated before token 1; terminated when checked before token 2.
    matcher.is_terminated.side_effect = [False, True]
    matcher.accept_token.return_value = True
    grammar = _grammar(matcher, terminated=False)

    assert grammar.validate_tokens([1, 2, 3]) == [1]
    assert matcher.accept_token.call_count == 1
    matcher.rollback.assert_called_once_with(1)


def test_validate_tokens_accepts_valid_prefix_when_not_terminated():
    matcher = MagicMock()
    matcher.is_terminated.return_value = False
    matcher.accept_token.side_effect = [True, True, False]
    grammar = _grammar(matcher, terminated=False)

    assert grammar.validate_tokens([1, 2, 3]) == [1, 2]
    matcher.rollback.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# accept_tokens / rollback termination tracking (vllm-project/vllm#27210)
# ---------------------------------------------------------------------------
def test_accept_tokens_records_first_terminating_position():
    matcher = MagicMock()
    matcher.accept_token.return_value = True
    # Matcher terminates while accepting the 3rd token.
    matcher.is_terminated.side_effect = [False, False, True]
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("req", [1, 2, 3]) is True
    assert grammar._terminated_at == 3
    assert grammar.num_processed_tokens == 3
    assert grammar.is_terminated() is True


def test_rollback_past_termination_clears_stuck_flag():
    # Reproduces #27210: the matcher keeps reporting terminated after rollback.
    matcher = MagicMock()
    matcher.accept_token.return_value = True
    matcher.is_terminated.return_value = True  # buggy: never clears
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("req", [1, 2, 3]) is True
    assert grammar.is_terminated() is True
    assert grammar._terminated_at == 1  # terminated on the first token

    grammar.rollback(3)

    # Despite matcher.is_terminated() still returning True, the grammar must
    # report not-terminated after rolling back past the terminating position.
    assert grammar._terminated_at is None
    assert grammar.is_terminated() is False
    assert grammar.num_processed_tokens == 0


def test_rollback_within_terminated_region_keeps_terminated():
    # Trailing-accepting grammar: termination first occurs at token 2 but
    # tokens 3 and 4 are still accepted (matcher stays terminated).
    matcher = MagicMock()
    matcher.accept_token.return_value = True
    matcher.is_terminated.side_effect = [False, True, True, True]
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("req", [1, 2, 3, 4]) is True
    assert grammar._terminated_at == 2
    assert grammar.num_processed_tokens == 4

    # Roll back into the still-terminated region (4 -> 3, which is >= 2).
    grammar.rollback(1)
    assert grammar.is_terminated() is True
    assert grammar._terminated_at == 2

    # Roll back past the terminating position (3 -> 1, which is < 2).
    grammar.rollback(2)
    assert grammar.is_terminated() is False
    assert grammar._terminated_at is None


def test_accept_tokens_short_circuits_when_terminated():
    matcher = MagicMock()
    grammar = _grammar(matcher, terminated=True)

    assert grammar.accept_tokens("req", [1, 2]) is False
    matcher.accept_token.assert_not_called()
