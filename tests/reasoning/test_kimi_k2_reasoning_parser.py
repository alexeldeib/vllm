# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from openai.types.responses import ResponseOutputMessage

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.parser.responses_parser import ResponsesParser
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.outputs import CompletionOutput
from vllm.parser.abstract_parser import _WrappedParser
from vllm.parser.request_utils import preserve_request_machine_output_contract
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"
JSON_ANSWER = '{"answer": 42}'
EMBEDDED_JSON_CONTENT = (
    "Here is JSON, not a tool call: "
    '[{"name": "get_weather", "parameters": {"city": "Paris"}}] '
    "and this sentence is still assistant content."
)


def _responses_json_schema_request(
    *,
    input_text: str = "Return the answer as JSON.",
    schema: dict | None = None,
    **kwargs,
):
    return ResponsesRequest(
        model="test-model",
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": schema
                or {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        },
        **kwargs,
    )


def _get_weather_tool():
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


def _get_weather_responses_tool():
    function = dict(_get_weather_tool()["function"])
    function["type"] = "function"
    return function


class KimiReasoningParser(_WrappedParser):
    reasoning_parser_cls = KimiK2ReasoningParser
    tool_parser_cls = None


class KimiReasoningAndToolParser(_WrappedParser):
    reasoning_parser_cls = KimiK2ReasoningParser
    tool_parser_cls = KimiK2ToolParser


def _chat_request(**kwargs):
    return ChatCompletionRequest(model="test-model", messages=[], **kwargs)


def _rewritten_structured_choice_request(choice: str):
    request = _chat_request(
        structured_outputs=StructuredOutputsParams(choice=[choice]),
    )
    preserve_request_machine_output_contract(request)
    request.structured_outputs = StructuredOutputsParams(structural_tag="{}")
    return request


def _assert_single_response_message_text(output_items, expected_text: str) -> None:
    assert len(output_items) == 1
    message = output_items[0]
    assert isinstance(message, ResponseOutputMessage)
    assert message.content[0].text == expected_text


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
    request = _chat_request(temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>step by step reasoning</think>final answer", request
    )
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_extract_reasoning_empty_thinking(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = _chat_request(temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think></think>final answer", request
    )
    assert reasoning == ""
    assert content == "final answer"


def test_extract_reasoning_implicit_start(kimi_k2_tokenizer):
    """When there's no <think> tag, everything is treated as reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = _chat_request(temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "implicit reasoning with no tags", request
    )
    assert reasoning == "implicit reasoning with no tags"
    assert content is None


def test_extract_reasoning_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> implicitly ends reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = _chat_request(temperature=1.0)

    text = "some reasoning<|tool_calls_section_begin|>tool call data"
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "some reasoning"
    assert content == "<|tool_calls_section_begin|>tool call data"


@pytest.mark.parametrize(
    ("completion_request", "text", "expected_reasoning", "expected_content"),
    [
        (
            _chat_request(response_format={"type": "json_object"}),
            JSON_ANSWER,
            None,
            JSON_ANSWER,
        ),
        (
            _chat_request(response_format={"type": "text"}),
            JSON_ANSWER,
            JSON_ANSWER,
            None,
        ),
        (
            _responses_json_schema_request(),
            JSON_ANSWER,
            None,
            JSON_ANSWER,
        ),
        (_chat_request(structured_outputs={"choice": ["42"]}), "42", None, "42"),
        (
            _chat_request(
                structured_outputs=StructuredOutputsParams(choice=["<think>literal"]),
            ),
            "<think>literal",
            None,
            "<think>literal",
        ),
        (
            _rewritten_structured_choice_request("<think>literal"),
            " <think>literal",
            None,
            " <think>literal",
        ),
        (
            _chat_request(response_format={"type": "json_object"}),
            "<think>literal",
            "literal",
            None,
        ),
    ],
)
def test_extract_reasoning_routes_machine_readable_output(
    mock_kimi_k2_tokenizer,
    completion_request,
    text,
    expected_reasoning,
    expected_content,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)

    reasoning, content = parser.extract_reasoning(text, completion_request)

    assert reasoning == expected_reasoning
    assert content == expected_content


def test_extract_reasoning_auto_tool_embedded_json_stays_content(
    mock_kimi_k2_tokenizer,
):
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    request = _chat_request(tools=[_get_weather_tool()], tool_choice="auto")

    text = f"<think>done</think>{EMBEDDED_JSON_CONTENT}"
    reasoning, content = parser.extract_reasoning(text, request)

    assert reasoning == "done"
    assert content == EMBEDDED_JSON_CONTENT


@pytest.mark.parametrize(
    ("completion_request", "expected_content", "expected_reasoning"),
    [
        (
            _chat_request(response_format={"type": "json_object"}),
            JSON_ANSWER,
            None,
        ),
        (_chat_request(response_format={"type": "text"}), None, JSON_ANSWER),
        (_responses_json_schema_request(), JSON_ANSWER, None),
    ],
)
def test_parse_delta_routes_one_shot_output(
    mock_kimi_k2_tokenizer,
    completion_request,
    expected_content,
    expected_reasoning,
):
    parser = KimiReasoningParser(mock_kimi_k2_tokenizer)

    result = parser.parse_delta(
        JSON_ANSWER,
        [999],
        completion_request,
        prompt_token_ids=[],
    )

    assert result is not None
    assert result.content == expected_content
    assert result.reasoning == expected_reasoning
    assert result.tool_calls == []


@pytest.mark.parametrize("with_tool_parser", [False, True])
def test_parse_delta_explicit_think_then_machine_readable_output(
    mock_kimi_k2_tokenizer,
    with_tool_parser,
):
    request_kwargs = {"response_format": {"type": "json_object"}}
    if with_tool_parser:
        request_kwargs.update(tools=[_get_weather_tool()], tool_choice="auto")
    request = _chat_request(**request_kwargs)
    parser = (
        KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)
        if with_tool_parser
        else KimiReasoningParser(mock_kimi_k2_tokenizer)
    )

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
    content_delta = parser.parse_delta(JSON_ANSWER, [999], request)

    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == "hidden"
    assert end_delta is None
    assert content_delta is not None
    assert content_delta.content == JSON_ANSWER
    assert content_delta.reasoning is None
    if with_tool_parser:
        assert content_delta.tool_calls == []


def test_parse_delta_machine_output_reasoning_end_with_content_strips_boundary(
    mock_kimi_k2_tokenizer,
):
    request = _chat_request(
        tools=[_get_weather_tool()],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    parser = KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)

    assert (
        parser.parse_delta(
            "<think>hidden",
            [parser._reasoning_parser._start_token_id, 998],
            request,
            prompt_token_ids=[],
        )
        is not None
    )
    result = parser.parse_delta(
        f"</think>{JSON_ANSWER}",
        [parser._reasoning_parser._end_token_id, 999],
        request,
    )

    assert result is not None
    assert result.content == JSON_ANSWER
    assert result.reasoning is None
    assert result.tool_calls == []


def test_parse_delta_machine_output_split_non_think_prefix_is_content(
    mock_kimi_k2_tokenizer,
):
    parser = KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = _chat_request(
        structured_outputs=StructuredOutputsParams(choice=["<ok"]),
    )

    assert parser.parse_delta("<", [997], request, prompt_token_ids=[]) is None
    result = parser.parse_delta("ok", [998], request)

    assert result is not None
    assert result.content == "<ok"
    assert result.reasoning is None


def test_parse_delta_machine_output_finished_exact_boundary_is_content(
    mock_kimi_k2_tokenizer,
):
    parser = KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = _chat_request(
        structured_outputs=StructuredOutputsParams(choice=["<think>"]),
    )

    result = parser.parse_delta(
        "<think>",
        [parser._reasoning_parser._start_token_id],
        request,
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "<think>"
    assert result.reasoning is None


def test_parse_delta_machine_output_think_choice_literal_is_content(
    mock_kimi_k2_tokenizer,
):
    parser = KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = _chat_request(
        structured_outputs=StructuredOutputsParams(choice=["<think>literal"]),
    )

    assert (
        parser.parse_delta(
            "<think>",
            [parser._reasoning_parser._start_token_id],
            request,
            prompt_token_ids=[],
        )
        is None
    )
    result = parser.parse_delta("literal", [999], request, finished=True)

    assert result is not None
    assert result.content == "<think>literal"
    assert result.reasoning is None


def test_extract_response_outputs_responses_json_schema_keeps_output_text(
    mock_kimi_k2_tokenizer,
):
    parser = KimiReasoningParser(mock_kimi_k2_tokenizer)
    request = _responses_json_schema_request()

    output_items = parser.extract_response_outputs(
        model_output=JSON_ANSWER,
        model_output_token_ids=[999],
        request=request,
    )

    _assert_single_response_message_text(output_items, JSON_ANSWER)


def test_responses_parser_json_schema_auto_tools_preserves_output_text(
    mock_kimi_k2_tokenizer,
):
    request = _responses_json_schema_request(
        input_text="Return JSON.",
        schema={"type": "object"},
        tools=[_get_weather_responses_tool()],
        tool_choice="auto",
    )
    parser = ResponsesParser(
        tokenizer=mock_kimi_k2_tokenizer,
        reasoning_parser_cls=KimiK2ReasoningParser,
        response_messages=[],
        request=request,
        tool_parser_cls=KimiK2ToolParser,
        chat_template=None,
        chat_template_content_format="auto",
    )
    output = CompletionOutput(
        index=0,
        text='{"answer":"Paris"}',
        token_ids=[999],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )

    parser.process(output)

    _assert_single_response_message_text(parser.response_messages, output.text)
    assert all(item.type != "function_call" for item in parser.response_messages)


def test_parse_delta_auto_tool_embedded_json_stays_content(
    mock_kimi_k2_tokenizer,
):
    request = _chat_request(
        tools=[_get_weather_tool()],
        tool_choice="auto",
    )
    parser = KimiReasoningAndToolParser(mock_kimi_k2_tokenizer, request.tools)

    result = parser.parse_delta(
        EMBEDDED_JSON_CONTENT,
        [999],
        request,
        prompt_token_ids=[parser._reasoning_parser._end_token_id],
    )

    assert result is not None
    assert result.content == EMBEDDED_JSON_CONTENT
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
