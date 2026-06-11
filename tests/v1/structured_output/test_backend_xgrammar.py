# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``XgrammarGrammar`` terminated-state safety under spec decode.

Speculative decoding builds a per-position grammar bitmask by advancing the
matcher through the scheduled draft tokens, and it filters draft tokens via
``validate_tokens``. Doing this on the *live* matcher and then rolling back
corrupts its parse state: xgrammar's ``rollback`` does not reliably restore the
full state (a broader form of vllm-project/vllm#27210). The corruption shows up
under spec decode at small batch sizes as an all-masked bitmask, which forces an
invalid token and terminates the request ("grammar rejected tokens ...").

The fix builds these speculative bitmasks (and validates draft tokens) on an
independent ``fork()`` of the matcher, never mutating the live matcher. These
tests cover fork-based validation, the fork() helper, and the accept/rollback
terminated-state tracking that remains for the non-speculative paths.
"""

from unittest.mock import MagicMock

import pytest

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

pytestmark = pytest.mark.cpu_test


def _grammar(matcher: MagicMock, terminated: bool = False) -> XgrammarGrammar:
    grammar = object.__new__(XgrammarGrammar)
    grammar.matcher = matcher
    grammar.vocab_size = 32
    grammar.ctx = MagicMock()
    grammar.num_processed_tokens = 0
    grammar._is_terminated = terminated
    grammar._terminated_at = None
    return grammar


# ---------------------------------------------------------------------------
# validate_tokens (draft-token filtering) — must NOT mutate the live matcher
# ---------------------------------------------------------------------------
def test_validate_tokens_skips_already_terminated_matcher():
    matcher = MagicMock()
    grammar = _grammar(matcher, terminated=True)

    assert grammar.validate_tokens([1, 2, 3]) == []
    matcher.fork.assert_not_called()
    matcher.accept_token.assert_not_called()


def test_validate_tokens_runs_on_fork_leaving_live_matcher_untouched():
    matcher = MagicMock()
    fork = matcher.fork.return_value
    fork.is_terminated.return_value = False
    fork.accept_token.side_effect = [True, True, False]
    grammar = _grammar(matcher, terminated=False)

    assert grammar.validate_tokens([1, 2, 3]) == [1, 2]
    matcher.fork.assert_called_once()
    # The live matcher must never be advanced or rolled back.
    matcher.accept_token.assert_not_called()
    matcher.rollback.assert_not_called()
    assert fork.accept_token.call_count == 3


def test_validate_tokens_stops_once_fork_terminates_mid_loop():
    matcher = MagicMock()
    fork = matcher.fork.return_value
    fork.is_terminated.side_effect = [False, True]
    fork.accept_token.return_value = True
    grammar = _grammar(matcher, terminated=False)

    assert grammar.validate_tokens([1, 2, 3]) == [1]
    assert fork.accept_token.call_count == 1
    matcher.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# fork()
# ---------------------------------------------------------------------------
def test_fork_returns_independent_grammar_at_same_state():
    matcher = MagicMock()
    forked_matcher = matcher.fork.return_value
    grammar = _grammar(matcher, terminated=False)
    grammar.num_processed_tokens = 4
    grammar._terminated_at = None

    forked = grammar.fork()

    assert isinstance(forked, XgrammarGrammar)
    assert forked is not grammar
    assert forked.matcher is forked_matcher
    assert forked.matcher is not grammar.matcher
    assert forked.num_processed_tokens == 4
    assert forked.vocab_size == grammar.vocab_size


# ---------------------------------------------------------------------------
# accept_tokens / rollback termination tracking (vllm-project/vllm#27210)
# ---------------------------------------------------------------------------
def test_accept_tokens_records_first_terminating_position():
    matcher = MagicMock()
    matcher.accept_token.return_value = True
    matcher.is_terminated.side_effect = [False, False, True]
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("req", [1, 2, 3]) is True
    assert grammar._terminated_at == 3
    assert grammar.is_terminated() is True


def test_rollback_past_termination_clears_stuck_flag():
    # Reproduces #27210: the matcher keeps reporting terminated after rollback.
    matcher = MagicMock()
    matcher.accept_token.return_value = True
    matcher.is_terminated.return_value = True  # buggy: never clears
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("req", [1, 2, 3]) is True
    assert grammar._terminated_at == 1

    grammar.rollback(3)

    assert grammar._terminated_at is None
    assert grammar.is_terminated() is False
    assert grammar.num_processed_tokens == 0


def test_rollback_within_terminated_region_keeps_terminated():
    matcher = MagicMock()
    matcher.accept_token.return_value = True
    matcher.is_terminated.side_effect = [False, True, True, True]
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("req", [1, 2, 3, 4]) is True
    assert grammar._terminated_at == 2

    grammar.rollback(1)  # 4 -> 3, still >= 2
    assert grammar.is_terminated() is True

    grammar.rollback(2)  # 3 -> 1, < 2
    assert grammar.is_terminated() is False
    assert grammar._terminated_at is None


def test_accept_tokens_short_circuits_when_terminated():
    matcher = MagicMock()
    grammar = _grammar(matcher, terminated=True)

    assert grammar.accept_tokens("req", [1, 2]) is False
    matcher.accept_token.assert_not_called()
