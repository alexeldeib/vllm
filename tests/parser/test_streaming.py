# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.parser.abstract_parser import (
    Parser,
    _WrappedParser,
    parse_delta_with_optional_finished,
)
from vllm.parser.request_utils import extract_reasoning_with_machine_output_contract
from vllm.reasoning import ReasoningParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class ThinkReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"


class NewlineTrimmingReasoningParser(ThinkReasoningParser):
    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        if previous_text.endswith(self.end_token) and delta_text.startswith("\n"):
            content = delta_text.removeprefix("\n")
            return DeltaMessage(content=content or None)

        result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if result is None:
            return None

        reasoning = result.reasoning
        content = result.content
        if reasoning is not None and self.end_token in delta_text:
            reasoning = reasoning.removesuffix("\n")
        if content is not None and self.end_token in delta_text:
            content = content.removeprefix("\n")
        return DeltaMessage(reasoning=reasoning or None, content=content or None)


class BaseStreamingEndReasoningParser(ReasoningParser):
    """Parser that inherits the base is_reasoning_end_streaming implementation."""

    @property
    def reasoning_end_str(self) -> str:
        return "</hidden>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.end_token_id = max(self.vocab.values()) + 1

    def is_reasoning_end(self, input_ids) -> bool:
        return self.end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id not in input_ids:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        return DeltaMessage(reasoning=delta_text or None)

    def extract_reasoning(self, model_output, request):
        return model_output, None


class SectionStartEndReasoningParser(BaseStreamingEndReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.section_start_token_id = tokenizer.encode(
            " section", add_special_tokens=False
        )[0]
        self.section_start_text = tokenizer.decode([self.section_start_token_id])

    @property
    def reasoning_end_strs(self) -> tuple[str, ...]:
        return (self.reasoning_end_str, self.section_start_text)

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return super().is_reasoning_end_streaming(
            input_ids, delta_ids
        ) or self.section_start_token_id in set(delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.section_start_token_id in input_ids:
            return input_ids[input_ids.index(self.section_start_token_id) :]
        return super().extract_content_ids(input_ids)


class SplittingSectionStartEndReasoningParser(SectionStartEndReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        self.start_token = tokenizer.decode([self.start_token_id])

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        if self.start_token_id in delta_token_ids:
            start_index = delta_text.find(self.start_token)
            if start_index >= 0:
                delta_text = delta_text[start_index + len(self.start_token) :]

        if self.section_start_token_id in delta_token_ids:
            section_index = delta_text.find(self.section_start_text)
            if section_index >= 0:
                return DeltaMessage(
                    reasoning=delta_text[:section_index] or None,
                    content=delta_text[section_index:] or None,
                )

        return DeltaMessage(reasoning=delta_text or None)


class HiddenSectionStartEndReasoningParser(SectionStartEndReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.section_start_token_id = max(self.vocab.values()) + 2
        self.section_start_text = "<|section|>"


class DelayedStreamingEndReasoningParser(BaseThinkingReasoningParser):
    """Parser that confirms a hidden end token after seeing the next token."""

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._end_token_pending = False

    def is_reasoning_end(self, input_ids) -> bool:
        return self._is_reasoning_end_from_ids(input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self._is_reasoning_end_from_ids(tuple(delta_ids))

    def _is_reasoning_end_from_ids(self, input_ids) -> bool:
        for idx in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[idx]
            if token_id == self.start_token_id:
                if self._end_token_pending:
                    return False
                self._end_token_pending = False
                return False
            if token_id == self.end_token_id:
                if idx < len(input_ids) - 1:
                    self._end_token_pending = False
                    return True
                self._end_token_pending = True
                return False

        if self._end_token_pending and input_ids:
            self._end_token_pending = False
            return True
        return False


MODEL_OUTPUT = (
    "<think>let me think about this</think>"
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
        },
    }


def _responses_tool():
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object"},
    }


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )


def make_parser(tokenizer, reasoning=False, tool=False):
    class WrappedParser(_WrappedParser):
        reasoning_parser_cls = ThinkReasoningParser if reasoning else None
        tool_parser_cls = Hermes2ProToolParser if tool else None

    return WrappedParser(tokenizer)


def make_wrapped_parser(tokenizer, reasoning_parser_cls, tool_parser_cls=None):
    reasoning_cls = reasoning_parser_cls
    tool_cls = tool_parser_cls

    class WrappedParser(_WrappedParser):
        reasoning_parser_cls = reasoning_cls
        tool_parser_cls = tool_cls

    return WrappedParser(tokenizer)


def patch_hidden_section_decode(parser, monkeypatch):
    reasoning_parser = parser._reasoning_parser
    hidden_token_id = reasoning_parser.section_start_token_id
    hidden_text = reasoning_parser.section_start_text
    decode = parser.model_tokenizer.decode

    def decode_with_hidden_token(token_ids, *args, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        parts: list[str] = []
        visible_ids: list[int] = []

        def flush_visible_ids():
            if visible_ids:
                parts.append(decode(visible_ids, *args, **kwargs))
                visible_ids.clear()

        for token_id in token_ids:
            if token_id == hidden_token_id:
                flush_visible_ids()
                if not skip_special_tokens:
                    parts.append(hidden_text)
            else:
                visible_ids.append(token_id)
        flush_visible_ids()
        return "".join(parts)

    monkeypatch.setattr(parser.model_tokenizer, "decode", decode_with_hidden_token)


def stream_text(parser, tokenizer, text, request, prompt_token_ids=None):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    for tid in token_ids:
        delta_text = tokenizer.decode([tid])
        result = parser.parse_delta(
            delta_text, [tid], request, prompt_token_ids=prompt_token_ids
        )
        prompt_token_ids = None
        results.append(result)
    return results


def collect_fields(results):
    all_reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    all_content = "".join(r.content for r in results if r and r.content)
    all_tool_calls = [tc for r in results if r and r.tool_calls for tc in r.tool_calls]
    return all_reasoning, all_content, all_tool_calls


def test_parse_delta_neither_parser(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=False, tool=False)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == ""
    assert len(tool_calls) == 0
    assert "<think>" in content
    assert "let me think about this" in content
    assert "<tool_call>" in content
    assert "get_weather" in content


def test_parse_delta_tool_parser_only(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == ""
    assert "<think>" in content
    assert "let me think about this" in content
    assert "</think>" in content

    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}


def test_parse_delta_reasoning_parser_only(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert len(tool_calls) == 0
    assert "<tool_call>" in content
    assert "get_weather" in content
    assert "</tool_call>" in content


def test_parse_delta_both_parsers(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert content == ""

    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}


def test_parse_delta_reasoning_only_thinking_disabled(tokenizer, request_obj):
    """Regression test for vllm-project/vllm#40466.

    When enable_thinking=False, the chat template places <think>\\n\\n</think>
    in the prompt. The model then generates pure content (no think tokens).
    All streaming output must go to delta.content, not delta.reasoning.
    """
    parser = make_parser(tokenizer, reasoning=True, tool=False)

    end_token_id = parser._reasoning_parser.end_token_id
    prompt_token_ids = [1, 2, end_token_id, 3]

    content_text = "Hello! How can I assist you today?"
    results = stream_text(
        parser,
        tokenizer,
        content_text,
        request_obj,
        prompt_token_ids=prompt_token_ids,
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == "", f"Expected no reasoning, got: {reasoning!r}"
    assert "Hello" in content
    assert "assist" in content
    assert len(tool_calls) == 0


def test_parse_delta_waits_for_prompt_token_ids_before_routing(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=True, tool=False)

    first = parser.parse_delta("", [], request_obj, prompt_token_ids=None)
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )
    second = parser.parse_delta(
        "answer",
        [token_id],
        request_obj,
        prompt_token_ids=[parser._reasoning_parser.end_token_id],
    )

    assert first is not None
    assert first.content == ""
    assert second is not None
    assert second.content == "answer"
    assert second.reasoning is None


def test_parse_delta_buffered_reasoning_end_splits_later_visible_boundary(
    tokenizer,
    request_obj,
):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    reasoning_parser = parser._reasoning_parser
    token_id = max(reasoning_parser.start_token_id, reasoning_parser.end_token_id) + 1

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )
    reasoning_delta = parser.parse_delta("thought", [token_id], request_obj)
    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == "thought"

    assert (
        parser.parse_delta(
            "",
            [reasoning_parser.end_token_id],
            request_obj,
        )
        is None
    )
    content_delta = parser.parse_delta("</think>answer", [token_id + 1], request_obj)

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == "answer"


def test_parse_delta_buffered_reasoning_end_handles_stripped_boundary(
    tokenizer,
    request_obj,
):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    reasoning_parser = parser._reasoning_parser
    token_id = max(reasoning_parser.start_token_id, reasoning_parser.end_token_id) + 1

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )
    assert parser.parse_delta("thought", [token_id], request_obj) is not None
    assert (
        parser.parse_delta(
            "",
            [reasoning_parser.end_token_id],
            request_obj,
        )
        is None
    )
    content_delta = parser.parse_delta("answer", [token_id + 1], request_obj)

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == "answer"


def test_parse_delta_buffered_reasoning_end_keeps_visible_prefix_as_reasoning(
    tokenizer,
    request_obj,
):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    reasoning_parser = parser._reasoning_parser
    token_id = max(reasoning_parser.start_token_id, reasoning_parser.end_token_id) + 1

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )

    reasoning_delta = parser.parse_delta(
        "last",
        [token_id, reasoning_parser.end_token_id],
        request_obj,
    )
    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == "last"
    assert reasoning_delta.content is None

    content_delta = parser.parse_delta("answer", [token_id + 1], request_obj)

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == "answer"


def test_parse_delta_hidden_reasoning_end_does_not_retrigger_base_streaming(
    tokenizer,
    request_obj,
):
    class WrappedParser(_WrappedParser):
        reasoning_parser_cls = BaseStreamingEndReasoningParser
        tool_parser_cls = None

    parser = WrappedParser(tokenizer)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    content_token_id = tokenizer.encode(" answer", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    content_text = tokenizer.decode([content_token_id])

    reasoning_delta = parser.parse_delta(
        reasoning_text,
        [reasoning_token_id, reasoning_parser.end_token_id],
        request_obj,
        prompt_token_ids=[],
    )
    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == reasoning_text
    assert reasoning_delta.content is None

    content_delta = parser.parse_delta(
        content_text,
        [content_token_id],
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == content_text


def test_parse_delta_hidden_reasoning_end_flushes_divergent_boundary_prefix(
    tokenizer,
    request_obj,
):
    class WrappedParser(_WrappedParser):
        reasoning_parser_cls = BaseStreamingEndReasoningParser
        tool_parser_cls = None

    parser = WrappedParser(tokenizer)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])

    reasoning_delta = parser.parse_delta(
        reasoning_text,
        [reasoning_token_id, reasoning_parser.end_token_id],
        request_obj,
        prompt_token_ids=[],
    )
    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == reasoning_text
    assert reasoning_delta.content is None

    prefix_text = "</hid"
    assert (
        parser.parse_delta(
            prefix_text,
            tokenizer.encode(prefix_text, add_special_tokens=False),
            request_obj,
        )
        is None
    )

    content_delta = parser.parse_delta(
        "x",
        tokenizer.encode("x", add_special_tokens=False),
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == prefix_text + "x"


@pytest.mark.parametrize(
    ("tail_text", "expected_content"),
    [
        ("den>answer", "answer"),
        ("x", "</hidx"),
    ],
)
def test_parse_delta_same_delta_hidden_reasoning_end_buffers_boundary_prefix(
    tokenizer,
    request_obj,
    tail_text,
    expected_content,
):
    parser = make_wrapped_parser(tokenizer, BaseStreamingEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    prefix_text = "</hid"
    prefix_token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    reasoning_delta = parser.parse_delta(
        reasoning_text + prefix_text,
        [reasoning_token_id, reasoning_parser.end_token_id, *prefix_token_ids],
        request_obj,
        prompt_token_ids=[],
    )

    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == reasoning_text
    assert reasoning_delta.content is None

    content_delta = parser.parse_delta(
        tail_text,
        tokenizer.encode(tail_text, add_special_tokens=False),
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == expected_content


def test_parse_delta_same_delta_hidden_reasoning_end_flushes_final_prefix(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, BaseStreamingEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    prefix_text = "</hid"
    prefix_token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    result = parser.parse_delta(
        reasoning_text + prefix_text,
        [reasoning_token_id, reasoning_parser.end_token_id, *prefix_token_ids],
        request_obj,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning == reasoning_text
    assert result.content == prefix_text


@pytest.mark.parametrize("same_delta", [True, False])
def test_parse_delta_hidden_reasoning_end_preserves_later_boundary_substring(
    tokenizer,
    request_obj,
    same_delta,
):
    parser = make_wrapped_parser(tokenizer, BaseStreamingEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    content_text = "foo</hidden>bar"
    content_token_ids = tokenizer.encode(content_text, add_special_tokens=False)

    if same_delta:
        content_delta = parser.parse_delta(
            reasoning_text + content_text,
            [reasoning_token_id, reasoning_parser.end_token_id, *content_token_ids],
            request_obj,
            prompt_token_ids=[],
        )
        assert content_delta is not None
        assert content_delta.reasoning == reasoning_text
    else:
        assert (
            parser.parse_delta(
                reasoning_text,
                [reasoning_token_id, reasoning_parser.end_token_id],
                request_obj,
                prompt_token_ids=[],
            )
            is not None
        )
        content_delta = parser.parse_delta(
            content_text,
            content_token_ids,
            request_obj,
        )
        assert content_delta is not None
        assert content_delta.reasoning is None

    assert content_delta.content == content_text


def test_parse_delta_hidden_reasoning_end_preserves_content_section_marker(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, SectionStartEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    content_text = reasoning_parser.section_start_text + "payload"
    payload_token_ids = tokenizer.encode("payload", add_special_tokens=False)

    result = parser.parse_delta(
        reasoning_text + content_text,
        [
            reasoning_token_id,
            reasoning_parser.section_start_token_id,
            *payload_token_ids,
        ],
        request_obj,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.reasoning == reasoning_text
    assert result.content == content_text


def test_parse_delta_alternate_reasoning_end_uses_parser_cleanup(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, SplittingSectionStartEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_text = " final"
    payload_text = "payload"
    content_text = reasoning_parser.section_start_text + payload_text
    reasoning_token_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    payload_token_ids = tokenizer.encode(payload_text, add_special_tokens=False)

    result = parser.parse_delta(
        reasoning_parser.start_token + reasoning_text + content_text,
        [
            reasoning_parser.start_token_id,
            *reasoning_token_ids,
            reasoning_parser.section_start_token_id,
            *payload_token_ids,
        ],
        request_obj,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.reasoning == reasoning_text
    assert result.content == content_text


def test_parse_delta_hidden_section_marker_replays_when_content_repeats_marker(
    tokenizer,
    request_obj,
    monkeypatch,
):
    parser = make_wrapped_parser(tokenizer, HiddenSectionStartEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    hidden_token_id = reasoning_parser.section_start_token_id
    hidden_text = reasoning_parser.section_start_text
    patch_hidden_section_decode(parser, monkeypatch)

    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    content_text = "payload" + hidden_text + "tail"
    content_token_ids = tokenizer.encode(content_text, add_special_tokens=False)

    result = parser.parse_delta(
        reasoning_text + content_text,
        [reasoning_token_id, hidden_token_id, *content_token_ids],
        request_obj,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning == reasoning_text
    assert result.content == hidden_text + content_text


def test_parse_delta_pending_hidden_section_marker_preserves_later_literal_marker(
    tokenizer,
    request_obj,
    monkeypatch,
):
    parser = make_wrapped_parser(tokenizer, HiddenSectionStartEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    hidden_token_id = reasoning_parser.section_start_token_id
    hidden_text = reasoning_parser.section_start_text
    patch_hidden_section_decode(parser, monkeypatch)

    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    payload_text = "payload"
    payload_token_ids = tokenizer.encode(payload_text, add_special_tokens=False)

    first = parser.parse_delta(
        reasoning_text + payload_text,
        [reasoning_token_id, hidden_token_id, *payload_token_ids],
        request_obj,
        prompt_token_ids=[],
    )

    assert first is not None
    assert first.reasoning == reasoning_text
    assert first.content is None

    content_text = "prefix" + hidden_text + "tail"
    result = parser.parse_delta(
        content_text,
        tokenizer.encode(content_text, add_special_tokens=False),
        request_obj,
        finished=True,
    )

    assert result is not None
    assert result.reasoning is None
    assert result.content == hidden_text + payload_text + content_text


def test_parse_delta_hidden_reasoning_end_buffers_hidden_section_marker(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, SectionStartEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    payload_token_ids = tokenizer.encode("payload", add_special_tokens=False)

    result = parser.parse_delta(
        "payload",
        [reasoning_parser.section_start_token_id, *payload_token_ids],
        request_obj,
        prompt_token_ids=[],
    )

    assert result is None
    assert not parser._stream_state.reasoning_ended
    assert not parser._stream_state.tool_call_text_started

    replay = parser.parse_delta(
        "more" + reasoning_parser.section_start_text + "tail",
        tokenizer.encode("moretail", add_special_tokens=False),
        request_obj,
    )

    assert replay is not None
    assert replay.reasoning is None
    assert replay.content == reasoning_parser.section_start_text + "payloadmoretail"


def test_parse_delta_hidden_reasoning_end_flushes_final_hidden_section_marker(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, SectionStartEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    payload_token_ids = tokenizer.encode("payload", add_special_tokens=False)

    result = parser.parse_delta(
        "payload",
        [reasoning_parser.section_start_token_id, *payload_token_ids],
        request_obj,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning is None
    assert result.content == reasoning_parser.section_start_text + "payload"


@pytest.mark.parametrize("same_delta", [True, False])
def test_parse_delta_delayed_hidden_reasoning_end_splits_content(
    tokenizer, request_obj, same_delta
):
    parser = make_wrapped_parser(tokenizer, DelayedStreamingEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    content_token_id = tokenizer.encode(" answer", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    content_text = tokenizer.decode([content_token_id])

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )
    assert (
        parser.parse_delta(reasoning_text, [reasoning_token_id], request_obj)
        is not None
    )
    if same_delta:
        content_delta = parser.parse_delta(
            content_text,
            [reasoning_parser.end_token_id, content_token_id],
            request_obj,
        )
    else:
        assert (
            parser.parse_delta("", [reasoning_parser.end_token_id], request_obj) is None
        )
        content_delta = parser.parse_delta(
            content_text,
            [content_token_id],
            request_obj,
        )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == content_text


def test_parse_delta_delayed_hidden_reasoning_end_splits_after_reasoning_delta(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, DelayedStreamingEndReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_token_id = tokenizer.encode(" last", add_special_tokens=False)[0]
    content_token_id = tokenizer.encode(" answer", add_special_tokens=False)[0]
    reasoning_text = tokenizer.decode([reasoning_token_id])
    content_text = tokenizer.decode([content_token_id])

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )

    reasoning_delta = parser.parse_delta(
        reasoning_text,
        [reasoning_token_id, reasoning_parser.end_token_id],
        request_obj,
    )

    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == reasoning_text
    assert reasoning_delta.content is None

    content_delta = parser.parse_delta(
        content_text,
        [content_token_id],
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == content_text


def test_parse_delta_same_delta_hidden_reasoning_end_preserves_parser_cleanup(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, NewlineTrimmingReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_text = " final\n"
    content_text = "\nanswer"
    reasoning_token_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    content_token_ids = tokenizer.encode(content_text, add_special_tokens=False)

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )

    result = parser.parse_delta(
        reasoning_text + content_text,
        [
            *reasoning_token_ids,
            reasoning_parser.end_token_id,
            *content_token_ids,
        ],
        request_obj,
    )

    assert result is not None
    assert result.reasoning == reasoning_text.removesuffix("\n")
    assert result.content == content_text.removeprefix("\n")


def test_parse_delta_same_delta_hidden_end_visible_boundary_uses_cleanup(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, NewlineTrimmingReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_text = " final\n"
    content_text = "</think>\nanswer"
    reasoning_token_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    content_token_ids = tokenizer.encode(content_text, add_special_tokens=False)

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )

    result = parser.parse_delta(
        reasoning_text + content_text,
        [
            *reasoning_token_ids,
            reasoning_parser.end_token_id,
            *content_token_ids,
        ],
        request_obj,
    )

    assert result is not None
    assert result.reasoning == reasoning_text.removesuffix("\n")
    assert result.content == "\nanswer".removeprefix("\n")


def test_parse_delta_pending_hidden_reasoning_end_preserves_parser_cleanup(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, NewlineTrimmingReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_text = " final"
    content_text = "\nanswer"
    reasoning_token_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    content_token_ids = tokenizer.encode(content_text, add_special_tokens=False)

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )
    reasoning_delta = parser.parse_delta(
        reasoning_text,
        [*reasoning_token_ids, reasoning_parser.end_token_id],
        request_obj,
    )
    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == reasoning_text
    assert reasoning_delta.content is None

    content_delta = parser.parse_delta(
        content_text,
        content_token_ids,
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == content_text.removeprefix("\n")


def test_parse_delta_pending_hidden_end_visible_boundary_uses_cleanup(
    tokenizer,
    request_obj,
):
    parser = make_wrapped_parser(tokenizer, NewlineTrimmingReasoningParser)
    reasoning_parser = parser._reasoning_parser
    reasoning_text = " final"
    content_text = "</think>\nanswer"
    reasoning_token_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    content_token_ids = tokenizer.encode(content_text, add_special_tokens=False)

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )
    reasoning_delta = parser.parse_delta(
        reasoning_text,
        [*reasoning_token_ids, reasoning_parser.end_token_id],
        request_obj,
    )
    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == reasoning_text
    assert reasoning_delta.content is None

    content_delta = parser.parse_delta(
        content_text,
        content_token_ids,
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == "\nanswer".removeprefix("\n")


def test_parse_delta_buffered_reasoning_end_keeps_visible_suffix_as_content(
    tokenizer,
    request_obj,
):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    reasoning_parser = parser._reasoning_parser
    token_id = max(reasoning_parser.start_token_id, reasoning_parser.end_token_id) + 1

    assert (
        parser.parse_delta(
            "<think>",
            [reasoning_parser.start_token_id],
            request_obj,
            prompt_token_ids=[],
        )
        is None
    )

    content_delta = parser.parse_delta(
        "answer",
        [reasoning_parser.end_token_id, token_id],
        request_obj,
    )

    assert content_delta is not None
    assert content_delta.reasoning is None
    assert content_delta.content == "answer"


def test_parse_delta_machine_output_buffers_ambiguous_thinking_prefix(
    tokenizer,
):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["<ok"]),
    )
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )

    assert parser.parse_delta("<", [token_id], request, prompt_token_ids=[]) is None
    result = parser.parse_delta("ok", [token_id + 1], request)

    assert result is not None
    assert result.content == "<ok"
    assert result.reasoning is None


def test_parse_delta_machine_output_flushes_finished_prefix(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["<"]),
    )
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )

    result = parser.parse_delta(
        "<",
        [token_id],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "<"
    assert result.reasoning is None


def test_parse_delta_machine_output_streams_incremental_content(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )

    deltas = [
        parser.parse_delta("{", [token_id], request, prompt_token_ids=[]),
        parser.parse_delta('"answer"', [token_id + 1], request),
        parser.parse_delta(":42}", [token_id + 2], request, finished=True),
    ]

    assert [delta.content for delta in deltas if delta] == [
        "{",
        '"answer"',
        ":42}",
    ]
    assert all(delta.reasoning is None for delta in deltas if delta)


def test_parse_delta_machine_output_flushes_finished_whitespace(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=[" "]),
    )
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )

    result = parser.parse_delta(
        " ",
        [token_id],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == " "
    assert result.reasoning is None


def test_parse_delta_machine_output_flushes_finished_token_buffer(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )
    text = '{"answer":42}'
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    for idx, token_id in enumerate(token_ids[:-1]):
        result = parser.parse_delta(
            "",
            [token_id],
            request,
            prompt_token_ids=[] if idx == 0 else None,
        )
        assert result is None

    result = parser.parse_delta("", [token_ids[-1]], request, finished=True)

    assert result is not None
    assert result.content == tokenizer.decode(token_ids, skip_special_tokens=False)
    assert result.reasoning is None


def test_parse_delta_machine_output_content_contract_bypasses_tool_parser(
    tokenizer,
):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["<"]),
    )
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )

    result = parser.parse_delta(
        "<",
        [token_id],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "<"
    assert result.reasoning is None
    assert result.tool_calls == []


def test_parse_delta_auto_tool_structured_outputs_uses_tool_parser(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice="auto",
        structured_outputs=StructuredOutputsParams(structural_tag='{"tags": []}'),
    )

    results = stream_text(parser, tokenizer, MODEL_OUTPUT, request, prompt_token_ids=[])
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert "<tool_call>" not in content
    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}


def test_parse_delta_auto_tool_content_structured_outputs_bypasses_tool_parser(
    tokenizer,
):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice="auto",
        structured_outputs=StructuredOutputsParams(choice=["<"]),
    )
    token_id = (
        max(
            parser._reasoning_parser.start_token_id,
            parser._reasoning_parser.end_token_id,
        )
        + 1
    )

    result = parser.parse_delta(
        "<",
        [token_id],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "<"
    assert result.reasoning is None
    assert result.tool_calls == []


def test_non_streaming_chat_auto_tool_content_contract_bypasses_tool_parser(
    tokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )

    tool_calls, content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=tokenizer,
        content=MODEL_OUTPUT,
        enable_auto_tools=True,
        tool_parser_cls=Hermes2ProToolParser,
    )

    assert not tool_calls
    assert content == MODEL_OUTPUT


def test_non_streaming_responses_auto_tool_content_contract_bypasses_tool_parser(
    tokenizer,
):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = ResponsesRequest(
        model="test-model",
        input="Return JSON",
        tools=[_responses_tool()],
        tool_choice="auto",
        text=ResponseTextConfig(
            format=ResponseFormatTextJSONSchemaConfig(
                type="json_schema",
                name="answer",
                schema={"type": "object"},
                strict=True,
            )
        ),
    )

    outputs = parser.extract_response_outputs(
        model_output=MODEL_OUTPUT,
        model_output_token_ids=[],
        request=request,
        enable_auto_tools=True,
    )

    assert [output.type for output in outputs] == ["message"]
    assert outputs[0].content[0].text == MODEL_OUTPUT


def test_extract_reasoning_machine_output_full_prefix_is_content(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["<"]),
    )

    reasoning, content = parser.extract_reasoning("<", request)

    assert reasoning is None
    assert content == "<"


@pytest.mark.parametrize("text", ["<think>", "</think>"])
def test_extract_reasoning_machine_output_exact_boundary_is_content(tokenizer, text):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=[text]),
    )

    reasoning, content = parser.extract_reasoning(text, request)

    assert reasoning is None
    assert content == text


def test_extract_reasoning_helper_preserves_untagged_structured_output(
    tokenizer,
):
    reasoning_parser = ThinkReasoningParser(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["ok"]),
    )

    reasoning, content = extract_reasoning_with_machine_output_contract(
        model_output='{"answer": 42}',
        request=request,
        reasoning_parser=reasoning_parser,
    )

    assert reasoning is None
    assert content == '{"answer": 42}'


def test_parse_delta_with_optional_finished_accepts_old_parser_signature(
    tokenizer,
    request_obj,
):
    class OldSignatureParser(Parser):
        def extract_reasoning(self, model_output, request):
            return None, model_output

        def extract_tool_calls(self, model_output, request):
            return None

        def extract_tool_calls_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        ):
            return None

        def parse_delta(
            self,
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=None,
        ):
            return DeltaMessage(content=delta_text)

    parser = OldSignatureParser(tokenizer)
    result = parse_delta_with_optional_finished(
        parser,
        delta_text="done",
        delta_token_ids=[1],
        request=request_obj,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "done"


def test_parse_delta_with_optional_finished_accepts_instance_parser_method(
    request_obj,
):
    class InstanceParser:
        pass

    parser = InstanceParser()

    def parse_delta(
        delta_text,
        delta_token_ids,  # noqa: ARG001
        request,  # noqa: ARG001
        prompt_token_ids=None,  # noqa: ARG001
        finished=False,
    ):
        return DeltaMessage(content=f"{delta_text}:{finished}")

    parser.parse_delta = parse_delta

    result = parse_delta_with_optional_finished(
        parser,
        delta_text="done",
        delta_token_ids=[1],
        request=request_obj,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "done:True"
