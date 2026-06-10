# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import regex as re
from pydantic import TypeAdapter

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.streaming import extract_required_tool_call_streaming
from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
]


def _required_tool_schema_matches(sample_output) -> bool:
    tools = TypeAdapter(list[ChatCompletionToolsParam]).validate_python(WEATHER_TOOLS)
    schema = get_json_schema_from_tools(tools=tools, tool_choice="required")
    assert isinstance(schema, dict)

    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(json.dumps(schema))
    return re.compile(regex).fullmatch(json.dumps(sample_output)) is not None


def _stream_required_tool_call_output(output_json: str, delta_len: int):
    previous_text = ""
    function_name_returned = False
    messages = []
    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned = extract_required_tool_call_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            function_name_returned=function_name_returned,
            tool_call_idx=None,
            tool_call_id_type="random",
        )

        if delta_message:
            messages.append(delta_message)

        previous_text = current_text

    return messages


def _reconstruct_tool_calls(messages):
    tool_calls = []
    for message in messages:
        tool_call = message.tool_calls[0]
        if tool_call.function.name:
            tool_calls.append(
                {
                    "name": tool_call.function.name,
                    "parameters": tool_call.function.arguments,
                }
            )
        else:
            tool_calls[-1]["parameters"] += tool_call.function.arguments

    for tool_call in tool_calls:
        tool_call["parameters"] = json.loads(tool_call["parameters"])

    return tool_calls


def test_required_tool_schema_rejects_extra_top_level_fields():
    assert _required_tool_schema_matches(
        [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}]
    )
    assert not _required_tool_schema_matches(
        [
            {
                "name": "get_current_weather",
                "parameters": {"city": "Vienna"},
                "type": "function",
            }
        ]
    )


@pytest.mark.parametrize("delta_len", [1, 3, 13])
@pytest.mark.parametrize(
    "output_json, expected",
    [
        (
            json.dumps(
                [
                    {
                        "name": "get_current_weather",
                        "parameters": {"city": "Vienna"},
                    }
                ]
            ),
            [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}],
        ),
        (
            json.dumps(
                [
                    {
                        "name": "get_current_weather",
                        "parameters": {"city": "Vienna"},
                        "type": "function",
                    }
                ]
            ),
            [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}],
        ),
        (
            json.dumps(
                [
                    {
                        "name": "get_current_weather",
                        "parameters": {"city": "Vienna"},
                        "strict": True,
                        "tool": "get_current_weather",
                    }
                ]
            ),
            [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}],
        ),
        (
            json.dumps(
                [
                    {
                        "name": "get_current_weather",
                        "parameters": {
                            "city": "Vienna",
                            "parameters": {"unit": "celsius"},
                        },
                        "type": "function",
                    }
                ]
            ),
            [
                {
                    "name": "get_current_weather",
                    "parameters": {
                        "city": "Vienna",
                        "parameters": {"unit": "celsius"},
                    },
                }
            ],
        ),
        (
            json.dumps(
                [
                    {
                        "name": "get_current_weather",
                        "parameters": {"city": "Vienna"},
                    },
                    {
                        "name": "get_current_weather",
                        "parameters": {"city": "Berlin"},
                        "type": "function",
                    },
                ]
            ),
            [
                {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
                {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
            ],
        ),
    ],
)
def test_required_tool_streaming_ignores_extra_top_level_fields(
    output_json, expected, delta_len
):
    messages = _stream_required_tool_call_output(output_json, delta_len)

    assert len(messages) > 0
    assert _reconstruct_tool_calls(messages) == expected
