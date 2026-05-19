# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.request_utils import (
    output_starts_with_machine_output_contract,
    output_starts_with_reasoning_boundary,
    preserve_request_machine_output_contract,
    request_has_generation_structured_text_contract,
    request_has_machine_output_contract,
    request_requires_tool_output,
)
from vllm.sampling_params import StructuredOutputsParams


def _chat_request(**kwargs):
    return ChatCompletionRequest(model="test-model", messages=[], **kwargs)


def _responses_request(format_: dict):
    return ResponsesRequest(
        model="test-model",
        input="test",
        text={"format": format_},
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


@pytest.mark.parametrize(
    ("completion_request", "expected"),
    [
        (_chat_request(response_format={"type": "text"}), False),
        (_chat_request(response_format={"type": "json_object"}), True),
        (
            _chat_request(structured_outputs=StructuredOutputsParams(choice=["42"])),
            True,
        ),
        (
            SimpleNamespace(
                response_format=None,
                structured_outputs=SimpleNamespace(structural_tag={"type": "object"}),
                text=None,
                tool_choice="auto",
            ),
            False,
        ),
        (_chat_request(tools=[_get_weather_tool()], tool_choice="required"), True),
        (
            _chat_request(
                tools=[_get_weather_tool()],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ),
            True,
        ),
        (_responses_request({"type": "text"}), False),
        (
            _responses_request(
                {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                    },
                }
            ),
            True,
        ),
    ],
)
def test_request_has_machine_output_contract(completion_request, expected):
    assert request_has_machine_output_contract(completion_request) is expected


@pytest.mark.parametrize(
    ("completion_request", "expected"),
    [
        (_chat_request(tools=[_get_weather_tool()], tool_choice="auto"), False),
        (
            _chat_request(
                tools=[_get_weather_tool()],
                tool_choice="auto",
                response_format={"type": "json_object"},
            ),
            False,
        ),
        (_chat_request(tools=[_get_weather_tool()], tool_choice="required"), True),
    ],
)
def test_request_requires_tool_output(completion_request, expected):
    assert request_requires_tool_output(completion_request) is expected


def test_preserve_request_machine_output_contract_survives_rewrite():
    request = _chat_request(response_format={"type": "json_object"})

    preserve_request_machine_output_contract(request)
    request.response_format = {"type": "structural_tag", "format": {}}
    request.structured_outputs = StructuredOutputsParams(structural_tag="{}")

    assert request_has_machine_output_contract(request)
    assert request_has_generation_structured_text_contract(request)


def test_preserve_request_machine_output_contract_keeps_choices():
    parser = SimpleNamespace(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
        reasoning_implicit_end_strs=("<|tool_calls_section_begin|>",),
    )
    request = _chat_request(
        structured_outputs=StructuredOutputsParams(choice=["<think>literal"]),
    )

    preserve_request_machine_output_contract(request)
    request.structured_outputs = StructuredOutputsParams(structural_tag="{}")

    assert request_has_machine_output_contract(request)
    assert request_has_generation_structured_text_contract(request)
    assert output_starts_with_machine_output_contract("<think>literal", request, parser)


def test_preserved_forced_tool_contract_does_not_look_like_structured_text():
    request = _chat_request(tools=[_get_weather_tool()], tool_choice="required")

    preserve_request_machine_output_contract(request)
    request.structured_outputs = StructuredOutputsParams(
        json={
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }
    )

    assert request_has_machine_output_contract(request)
    assert not request_has_generation_structured_text_contract(request)


def test_output_starts_with_reasoning_boundary():
    parser = SimpleNamespace(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
    )

    assert output_starts_with_reasoning_boundary("<think>", parser)
    assert output_starts_with_reasoning_boundary("  </think>", parser)
    assert not output_starts_with_reasoning_boundary("<thi", parser)
    assert output_starts_with_reasoning_boundary("<thi", parser, allow_prefix=True)
    assert not output_starts_with_reasoning_boundary('{"answer": 42}', parser)


def test_output_starts_with_machine_output_contract():
    parser = SimpleNamespace(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
        reasoning_implicit_end_strs=("<|tool_calls_section_begin|>",),
    )
    structured_request = _chat_request(
        structured_outputs=StructuredOutputsParams(choice=["<think>literal"]),
    )
    json_request = _chat_request(response_format={"type": "json_object"})
    forced_tool_request = _chat_request(
        tools=[_get_weather_tool()],
        tool_choice="required",
    )

    assert output_starts_with_machine_output_contract(
        "<think>literal", structured_request, parser
    )
    assert not output_starts_with_machine_output_contract(
        "<think>", structured_request, parser
    )
    assert not output_starts_with_machine_output_contract(
        "<think>hidden</think>{}", structured_request, parser
    )
    assert not output_starts_with_machine_output_contract(
        "<think>literal", json_request, parser
    )
    assert output_starts_with_machine_output_contract(
        '{"name": "get_weather"}', forced_tool_request, parser
    )
    assert not output_starts_with_machine_output_contract(
        "natural language reasoning", forced_tool_request, parser
    )
