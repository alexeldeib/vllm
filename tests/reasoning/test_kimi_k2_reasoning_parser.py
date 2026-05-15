# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }


def _responses_json_schema_request():
    return ResponsesRequest(
        model="test-model",
        input="Return JSON",
        text=ResponseTextConfig(
            format=ResponseFormatTextJSONSchemaConfig(
                type="json_schema",
                name="answer",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                },
                strict=True,
            )
        ),
    )


@pytest.fixture
def mock_kimi_k2_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 100,
        "</think>": 101,
        "<|tool_calls_section_begin|>": 200,
        "<|tool_calls_section_end|>": 201,
        "<|tool_call_begin|>": 202,
        "<|tool_call_end|>": 203,
    }
    return tokenizer


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=REASONING_MODEL_NAME, trust_remote_code=True)


class _KimiReasoningParser(_WrappedParser):
    reasoning_parser_cls = KimiK2ReasoningParser
    tool_parser_cls = None


class _KimiReasoningAndToolParser(_WrappedParser):
    reasoning_parser_cls = KimiK2ReasoningParser
    tool_parser_cls = KimiK2ToolParser


def test_parser_selection_thinking_enabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": True}
    )
    assert parser._identity_parser is None


def test_parser_selection_thinking_disabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": False}
    )
    assert isinstance(parser._identity_parser, IdentityReasoningParser)


def test_extract_reasoning_with_think_tags(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>step by step reasoning</think>final answer", request
    )
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_extract_reasoning_empty_thinking(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think></think>final answer", request
    )
    assert reasoning == ""
    assert content == "final answer"


def test_extract_reasoning_implicit_start(kimi_k2_tokenizer):
    """When there's no <think> tag, everything is treated as reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "implicit reasoning with no tags", request
    )
    assert reasoning == "implicit reasoning with no tags"
    assert content is None


def test_extract_reasoning_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> implicitly ends reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    text = "some reasoning<|tool_calls_section_begin|>tool call data"
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "some reasoning"
    assert content == "<|tool_calls_section_begin|>tool call data"


def test_extract_reasoning_json_response_format_bypasses_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    reasoning, content = parser.extract_reasoning('{"answer": 42}', request)

    assert reasoning is None
    assert content == '{"answer": 42}'


def test_extract_reasoning_structured_output_bypasses_implicit_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["ok"]),
    )

    reasoning, content = parser.extract_reasoning("ok", request)

    assert reasoning is None
    assert content == "ok"


@pytest.mark.parametrize("model_output", ["<", "<thi", "</th"])
def test_extract_reasoning_structured_output_prefix_is_content(
    mock_kimi_k2_tokenizer,
    model_output,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=[model_output]),
    )

    reasoning, content = parser.extract_reasoning(model_output, request)

    assert reasoning is None
    assert content == model_output


@pytest.mark.parametrize("model_output", ["<think>", "</think>"])
def test_extract_reasoning_structured_output_exact_boundary_is_content(
    mock_kimi_k2_tokenizer,
    model_output,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=[model_output]),
    )

    reasoning, content = parser.extract_reasoning(model_output, request)

    assert reasoning is None
    assert content == model_output


def test_extract_reasoning_machine_output_boundary_substring_is_content(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["hello </think> world"]),
    )

    reasoning, content = parser.extract_reasoning("hello </think> world", request)

    assert reasoning is None
    assert content == "hello </think> world"


def test_extract_reasoning_machine_output_strips_explicit_think(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    reasoning, content = parser.extract_reasoning(
        '<think>hidden</think>{"answer": 42}', request
    )

    assert reasoning == "hidden"
    assert content == '{"answer": 42}'


def test_extract_reasoning_text_response_format_keeps_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "text"},
    )

    reasoning, content = parser.extract_reasoning('{"answer": 42}', request)

    assert reasoning == '{"answer": 42}'
    assert content is None


def test_extract_reasoning_auto_tool_embedded_json_stays_content(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    text = (
        "<think>done</think>Here is JSON, not a tool call: "
        '[{"name": "get_weather", "parameters": {"city": "Paris"}}] '
        "and this sentence is still assistant content."
    )
    reasoning, content = parser.extract_reasoning(text, request)

    assert reasoning == "done"
    assert content == (
        "Here is JSON, not a tool call: "
        '[{"name": "get_weather", "parameters": {"city": "Paris"}}] '
        "and this sentence is still assistant content."
    )


def test_parse_delta_response_format_json_bypasses_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    result = parser.parse_delta(
        '{"answer": 42}',
        [999],
        request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.content == '{"answer": 42}'
    assert result.reasoning is None
    assert result.tool_calls == []


def test_parse_delta_response_format_json_with_tool_parser_streams_content(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningAndToolParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    result = parser.parse_delta(
        '{"answer": 42}',
        [999],
        request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.content == '{"answer": 42}'
    assert result.reasoning is None
    assert result.tool_calls == []


def test_parse_delta_machine_output_explicit_think_is_stripped(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    assert parser.parse_delta("  ", [997], request, prompt_token_ids=[]) is None
    assert (
        parser.parse_delta(
            "<think>",
            [parser._reasoning_parser._start_token_id],
            request,
            prompt_token_ids=[],
        )
        is None
    )
    reasoning_delta = parser.parse_delta("hidden", [998], request)
    end_delta = parser.parse_delta(
        "</think>",
        [parser._reasoning_parser._end_token_id],
        request,
    )
    content_delta = parser.parse_delta('{"answer": 42}', [999], request)

    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == "hidden"
    assert end_delta is None
    assert content_delta is not None
    assert content_delta.content == '{"answer": 42}'
    assert content_delta.reasoning is None


def test_parse_delta_machine_output_combined_think_delta_is_stripped(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    result = parser.parse_delta(
        '  <think>hidden</think>{"answer": 42}',
        [997, parser._reasoning_parser._start_token_id, 998, 999],
        request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.reasoning == "hidden"
    assert result.content == '{"answer": 42}'


def test_parse_delta_machine_output_split_think_prefix_is_buffered(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    assert parser.parse_delta("<", [997], request, prompt_token_ids=[]) is None
    result = parser.parse_delta(
        'think>hidden</think>{"answer": 42}',
        [998, parser._reasoning_parser._end_token_id, 999],
        request,
    )

    assert result is not None
    assert result.reasoning == "hidden"
    assert result.content == '{"answer": 42}'


def test_parse_delta_machine_output_split_non_think_prefix_is_content(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["<ok"]),
    )

    assert parser.parse_delta("<", [997], request, prompt_token_ids=[]) is None
    result = parser.parse_delta("ok", [998], request)

    assert result is not None
    assert result.content == "<ok"
    assert result.reasoning is None


def test_parse_delta_machine_output_finished_prefix_is_content(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=["<"]),
    )

    result = parser.parse_delta(
        "<",
        [997],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "<"
    assert result.reasoning is None


@pytest.mark.parametrize(
    ("text", "token_id"),
    [
        ("<think>", 100),
        ("</think>", 101),
    ],
)
def test_parse_delta_machine_output_finished_exact_boundary_is_content(
    mock_kimi_k2_tokenizer,
    text,
    token_id,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=[text]),
    )

    result = parser.parse_delta(
        text,
        [token_id],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == text
    assert result.reasoning is None


@pytest.mark.parametrize(
    ("first", "second", "token_ids", "expected"),
    [
        ("<", "think>", [997, 998], "<think>"),
        ("</", "think>", [999, 1000], "</think>"),
    ],
)
def test_parse_delta_machine_output_buffered_exact_boundary_is_content(
    mock_kimi_k2_tokenizer,
    first,
    second,
    token_ids,
    expected,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        structured_outputs=StructuredOutputsParams(choice=[expected]),
    )

    assert (
        parser.parse_delta(first, [token_ids[0]], request, prompt_token_ids=[]) is None
    )
    result = parser.parse_delta(second, [token_ids[1]], request, finished=True)

    assert result is not None
    assert result.content == expected
    assert result.reasoning is None


@pytest.mark.parametrize("prefix", ["<", "<thi"])
def test_parse_delta_finished_reasoning_prefix_is_reasoning(
    mock_kimi_k2_tokenizer,
    prefix,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[])

    result = parser.parse_delta(
        prefix,
        [997],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning == prefix
    assert result.content is None


def test_parse_delta_structural_tag_response_format_keeps_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "structural_tag", "format": {}},
    )

    result = parser.parse_delta(
        "implicit reasoning",
        [997],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning == "implicit reasoning"
    assert result.content is None


def test_parse_delta_text_response_format_keeps_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "text"},
    )

    result = parser.parse_delta(
        '{"answer": 42}',
        [999],
        request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.reasoning == '{"answer": 42}'
    assert result.content is None
    assert result.tool_calls == []


def test_parse_delta_required_tool_choice_streams_tool_call(
    mock_kimi_k2_tokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice="required",
    )
    parser = _KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)

    result = parser.parse_delta(
        '[{"name": "get_weather", "parameters": {"city": "Paris"}}]',
        [999],
        request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.reasoning is None
    assert result.content is None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "get_weather"


def test_parse_delta_named_tool_choice_streams_tool_call(
    mock_kimi_k2_tokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    parser = _KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)

    result = parser.parse_delta(
        '{"city": "Paris"}',
        [999],
        request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.reasoning is None
    assert result.content is None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city": "Paris"}'


def test_parse_delta_buffered_reasoning_end_starts_tool_phase_from_text(
    mock_kimi_k2_tokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice="auto",
    )
    parser = _KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)

    first = parser.parse_delta(
        "buffered",
        [parser._reasoning_parser._tool_section_start_token_id, 998],
        request,
        prompt_token_ids=[parser._reasoning_parser._start_token_id],
    )
    assert first is None
    assert not parser._stream_state.reasoning_ended
    assert not parser._stream_state.tool_call_text_started

    result = parser.parse_delta(
        (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>get_weather:0"
            '<|tool_call_argument_begin|>{"city": "Paris"}'
            "<|tool_call_end|>"
        ),
        [999],
        request,
    )

    assert result is not None
    assert parser._stream_state.reasoning_ended
    assert result.reasoning is None
    assert result.content is None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "get_weather"


def test_responses_extract_outputs_text_format_bypasses_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    outputs = parser.extract_response_outputs(
        model_output='{"answer": 42}',
        model_output_token_ids=[999],
        request=_responses_json_schema_request(),
    )

    assert len(outputs) == 1
    assert outputs[0].type == "message"
    assert outputs[0].content[0].text == '{"answer": 42}'


def test_responses_parse_delta_text_format_bypasses_reasoning(
    mock_kimi_k2_tokenizer,
):
    parser = _KimiReasoningParser(mock_kimi_k2_tokenizer)
    result = parser.parse_delta(
        '{"answer": 42}',
        [999],
        _responses_json_schema_request(),
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.content == '{"answer": 42}'
    assert result.reasoning is None


def test_parse_delta_auto_tool_embedded_json_stays_content(
    mock_kimi_k2_tokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[_weather_tool()],
        tool_choice="auto",
    )
    parser = _KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)
    content = (
        "Here is JSON, not a tool call: "
        '[{"name": "get_weather", "parameters": {"city": "Paris"}}] '
        "and this sentence is still assistant content."
    )

    result = parser.parse_delta(
        content,
        [999],
        request,
        prompt_token_ids=[parser._reasoning_parser._end_token_id],
    )

    assert result is not None
    assert result.content == content
    assert result.reasoning is None
    assert result.tool_calls == []


def test_streaming_reasoning_then_content(kimi_k2_tokenizer):
    """Token-by-token streaming: reasoning tokens then content after </think>."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    end_think_id = parser._end_token_id
    # Use a real token ID from the tokenizer for regular content
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # First token: <think> — single special token should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>",
        delta_text="<think>",
        previous_token_ids=[],
        current_token_ids=[think_id],
        delta_token_ids=[think_id],
    )
    assert result is None

    # Reasoning token
    result = parser.extract_reasoning_streaming(
        previous_text="<think>",
        current_text="<think>step one",
        delta_text="step one",
        previous_token_ids=[think_id],
        current_token_ids=[think_id, regular_id],
        delta_token_ids=[regular_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.reasoning == "step one"
    assert result.content is None

    # End token </think> as single token — should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one",
        current_text="<think>step one</think>",
        delta_text="</think>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, end_think_id],
        delta_token_ids=[end_think_id],
    )
    assert result is None

    # Content after </think>
    content_id = kimi_k2_tokenizer.encode("world", add_special_tokens=False)[0]
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one</think>",
        current_text="<think>step one</think>answer",
        delta_text="answer",
        previous_token_ids=[think_id, regular_id, end_think_id],
        current_token_ids=[think_id, regular_id, end_think_id, content_id],
        delta_token_ids=[content_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "answer"


def test_streaming_split_non_think_prefix_preserves_prefix(mock_kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)

    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<",
        delta_text="<",
        previous_token_ids=[],
        current_token_ids=[997],
        delta_token_ids=[997],
    )
    assert result is None

    result = parser.extract_reasoning_streaming(
        previous_text="<",
        current_text="<ok",
        delta_text="ok",
        previous_token_ids=[997],
        current_token_ids=[997, 998],
        delta_token_ids=[998],
    )

    assert isinstance(result, DeltaMessage)
    assert result.reasoning == "<ok"


def test_streaming_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> in delta ends reasoning during streaming."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # Tool section token arrives — should transition from reasoning to content
    result = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, tool_begin_id],
        delta_token_ids=[tool_begin_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "<|tool_calls_section_begin|>"


def test_streaming_end_token_id_buffered(mock_kimi_k2_tokenizer):
    """When stop sequences buffer text, </think> ID arrives before its text.

    The token ID is present in delta_token_ids but the actual string is not
    yet in delta_text (still buffered). The parser must return None to wait
    for the next delta, instead of calling find() which returns -1 and
    silently corrupting the text split.
    """
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    end_think_id = parser._end_token_id

    # Simulate: </think> ID arrived but text not yet flushed.
    # Two token IDs in delta to bypass the single-special-token guard.
    result = parser.extract_reasoning_streaming(
        previous_text="some reasoning",
        current_text="some reasoning extra",
        delta_text="extra",  # </think> text not yet flushed
        previous_token_ids=[think_id],
        current_token_ids=[think_id, end_think_id, 999],
        delta_token_ids=[end_think_id, 999],
    )
    assert result is None


def test_streaming_tool_section_id_buffered(mock_kimi_k2_tokenizer):
    """When stop sequences buffer text, tool section start ID arrives before its text.

    Same buffering scenario as above but for <|tool_calls_section_begin|>.
    Without the guard, find() returns -1 and delta_text[:tool_index] silently
    drops the last character of reasoning.
    """
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id

    result = parser.extract_reasoning_streaming(
        previous_text="some reasoning",
        current_text="some reasoning extra",
        delta_text="extra",  # tool section text not yet flushed
        previous_token_ids=[think_id],
        current_token_ids=[think_id, tool_begin_id, 999],
        delta_token_ids=[tool_begin_id, 999],
    )
    assert result is None
