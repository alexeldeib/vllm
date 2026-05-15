# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.parser.request_utils import (
    output_is_exact_reasoning_boundary,
    output_starts_with_reasoning_boundary,
    preserve_request_machine_output_contract,
    request_allows_auto_tool_output,
    request_has_machine_output_contract,
)
from vllm.sampling_params import StructuredOutputsParams


class _ReasoningParser:
    reasoning_start_str = "<think>"
    reasoning_end_str = "</think>"


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


def test_request_has_machine_output_contract_chat_response_format():
    assert not request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            response_format={"type": "text"},
        )
    )
    assert request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            response_format={"type": "json_object"},
        )
    )
    assert request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                },
            },
        )
    )


def test_request_has_machine_output_contract_structured_outputs():
    assert request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            structured_outputs=StructuredOutputsParams(choice=["ok", "no"]),
        )
    )
    assert not request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            structured_outputs=StructuredOutputsParams(structural_tag='{"tags": []}'),
        )
    )
    assert not request_has_machine_output_contract(
        SimpleNamespace(
            response_format=None,
            structured_outputs={"structural_tag": '{"tags": []}'},
            text=None,
            tool_choice="auto",
        )
    )
    assert not request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            response_format={"type": "structural_tag", "format": {}},
        )
    )


def test_request_has_machine_output_contract_tool_choice():
    assert not request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="auto",
        )
    )
    assert request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="required",
        )
    )
    assert request_has_machine_output_contract(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
        )
    )


def test_request_allows_auto_tool_output():
    assert request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="auto",
        )
    )
    assert not request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="required",
        )
    )
    assert not request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tool_choice="auto",
        )
    )
    assert not request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="auto",
            response_format={"type": "json_object"},
        )
    )
    assert not request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="auto",
            structured_outputs=StructuredOutputsParams(choice=["<"]),
        )
    )
    assert request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="auto",
            structured_outputs=StructuredOutputsParams(structural_tag='{"tags": []}'),
        )
    )
    assert request_allows_auto_tool_output(
        ChatCompletionRequest(
            model="test-model",
            messages=[],
            tools=[_weather_tool()],
            tool_choice="auto",
            response_format={"type": "structural_tag", "format": {}},
        )
    )


def test_request_has_machine_output_contract_responses_text_format_and_tool_choice():
    text_format_request = ResponsesRequest(
        model="test-model",
        input="Return JSON",
        text=ResponseTextConfig(
            format=ResponseFormatTextJSONSchemaConfig(
                type="json_schema",
                name="answer",
                schema={"type": "object"},
                strict=True,
            )
        ),
    )
    assert request_has_machine_output_contract(text_format_request)

    named_tool_request = ResponsesRequest(
        model="test-model",
        input="Call the tool",
        tools=[_responses_tool()],
        tool_choice={"type": "function", "name": "get_weather"},
    )
    assert request_has_machine_output_contract(named_tool_request)


def test_request_has_machine_output_contract_accepts_dict_fields():
    text_request = SimpleNamespace(
        response_format={"type": "text"},
        structured_outputs=None,
        text={"format": {"type": "text"}},
        tool_choice="auto",
    )
    assert not request_has_machine_output_contract(text_request)

    json_request = SimpleNamespace(
        response_format={"type": "text"},
        structured_outputs=None,
        text={"format": {"type": "json_schema"}},
        tool_choice="auto",
    )
    assert request_has_machine_output_contract(json_request)

    named_tool_request = SimpleNamespace(
        response_format=None,
        structured_outputs=None,
        text=None,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert request_has_machine_output_contract(named_tool_request)


def test_preserve_request_machine_output_contract_survives_structural_tag_rewrite():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    preserve_request_machine_output_contract(request)
    request.response_format = None
    request.structured_outputs = StructuredOutputsParams(structural_tag='{"tags": []}')
    request.tools = [_weather_tool()]
    request.tool_choice = "auto"

    assert request_has_machine_output_contract(request)
    assert not request_allows_auto_tool_output(request)


def test_output_starts_with_reasoning_boundary_full_output_requires_boundary():
    parser = _ReasoningParser()

    assert output_starts_with_reasoning_boundary("<think>", parser)
    assert output_starts_with_reasoning_boundary("  <think>", parser)
    assert output_starts_with_reasoning_boundary("</think>", parser)
    assert not output_starts_with_reasoning_boundary("<", parser)
    assert not output_starts_with_reasoning_boundary("<thi", parser)
    assert not output_starts_with_reasoning_boundary("</th", parser)
    assert not output_starts_with_reasoning_boundary("  ", parser)
    assert not output_starts_with_reasoning_boundary('{"answer": 42}', parser)


@pytest.mark.parametrize("text", ["<think>", " </think> "])
def test_output_is_exact_reasoning_boundary(text):
    parser = _ReasoningParser()

    assert output_is_exact_reasoning_boundary(text, parser)


def test_output_starts_with_reasoning_boundary_streaming_accepts_prefixes():
    parser = _ReasoningParser()

    assert output_starts_with_reasoning_boundary("<", parser, allow_prefix=True)
    assert output_starts_with_reasoning_boundary("<thi", parser, allow_prefix=True)
    assert output_starts_with_reasoning_boundary("</th", parser, allow_prefix=True)
