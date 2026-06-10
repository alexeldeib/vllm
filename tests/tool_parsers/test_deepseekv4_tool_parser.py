# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV4ToolParser."""

import json
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
            },
        ),
    ]


def make_parser(tools=None) -> DeepSeekV4ToolParser:
    return DeepSeekV4ToolParser(MOCK_TOKENIZER, tools=tools)


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    return req


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    param_strs = "".join(
        f'{PARAM_START}{k}" string="true">{v}{PARAM_END}\n' for k, v in params.items()
    )
    return f'{TC_START}\n{INV_START}{func_name}">\n{param_strs}{INV_END}\n{TC_END}'


def stream(parser: DeepSeekV4ToolParser, full_text: str, chunk_size: int = 7):
    deltas = []
    previous_text = ""
    for start in range(0, len(full_text), chunk_size):
        delta_text = full_text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=make_request(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_args(deltas, tool_index: int = 0) -> str:
    fragments = []
    for delta in deltas:
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if (
                    tool_call.index == tool_index
                    and tool_call.function
                    and tool_call.function.arguments
                ):
                    fragments.append(tool_call.function.arguments)
    return "".join(fragments)


def test_registered():
    assert ToolParserManager.get_tool_parser("deepseek_v4") is DeepSeekV4ToolParser


def test_extract_tool_calls():
    parser = make_parser()
    model_output = "Let me check. " + build_tool_call(
        "get_weather", {"location": "Beijing", "unit": "celsius"}
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.content == "Let me check. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Beijing",
        "unit": "celsius",
    }


def test_function_calls_block_is_not_accepted():
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "vllm"}).replace(
        "tool_calls", "function_calls"
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert not result.tools_called
    assert result.content == model_output


def test_streaming_extracts_complete_invokes():
    parser = make_parser()
    full_text = build_tool_call("search", {"query": "deepseek v4"})

    deltas = stream(parser, full_text, chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
        if tool_call.function.name
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "deepseek v4"}


def test_streaming_emits_incremental_argument_chunks():
    tool = ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="plan_trip",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        ),
    )
    parser = make_parser(tools=[tool])
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}plan_trip">\n'
        f'{PARAM_START}days" string="false">3{PARAM_END}\n'
        f'{PARAM_START}flexible" string="false">false{PARAM_END}\n'
        f'{PARAM_START}cities" string="false">'
        f'["Beijing","Shanghai","Tokyo","New York"]{PARAM_END}\n'
        f'{PARAM_START}notes" string="true">靠窗座位{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )

    deltas = stream(parser, full_text, chunk_size=4)
    arg_chunks = [
        tool_call.function.arguments
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert len([chunk for chunk in arg_chunks if chunk]) > 2
    assert json.loads("".join(arg_chunks)) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }


def test_get_vllm_registry_structural_tag_returns_structural_tag(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser()
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="auto",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    if sample_tools:
        tool = sample_tools[0]
        req = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=sample_tools,
        )
        req.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=tool.function.name)
        )
        tag = parser.get_structural_tag(req)
        assert isinstance(tag, StructuralTag)


def test_required_tool_choice_uses_phase_aware_tool_grammar(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser(sample_tools)
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
        chat_template_kwargs={"thinking": True, "enable_thinking": True},
    )

    out = parser.adjust_request(req)

    assert out._grammar_from_tool_parser is True
    assert out.response_format is None
    assert out.skip_special_tokens is False
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
    loaded = json.loads(out.structured_outputs.structural_tag)
    serialized = json.dumps(loaded, ensure_ascii=False)
    assert "</think>" not in serialized
    assert "<｜DSML｜tool_calls>" in serialized


def test_named_tool_choice_uses_phase_aware_tool_grammar(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser(sample_tools)
    tool = sample_tools[0]
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        chat_template_kwargs={"thinking": True, "enable_thinking": True},
    )
    req.tool_choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name=tool.function.name)
    )

    out = parser.adjust_request(req)

    assert out._grammar_from_tool_parser is True
    assert out.response_format is None
    assert out.skip_special_tokens is False
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
    loaded = json.loads(out.structured_outputs.structural_tag)
    serialized = json.dumps(loaded, ensure_ascii=False)
    assert "</think>" not in serialized
    assert tool.function.name in serialized


def test_named_tool_choice_without_thinking_omits_thinking_prefix(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser(sample_tools)
    tool = sample_tools[0]
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        chat_template_kwargs={"thinking": False, "enable_thinking": False},
    )
    req.tool_choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name=tool.function.name)
    )

    out = parser.adjust_request(req)

    assert out._grammar_from_tool_parser is True
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
    loaded = json.loads(out.structured_outputs.structural_tag)
    serialized = json.dumps(loaded, ensure_ascii=False)
    assert "</think>" not in serialized
    assert tool.function.name in serialized


def test_parser_owned_grammar_streams_tool_calls_in_thinking_phase(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {"<think>": 1, "</think>": 2}

    class Parser(DelegatingParser):
        reasoning_parser_cls = DeepSeekV3ReasoningParser
        tool_parser_cls = DeepSeekV4ToolParser

    parser = Parser(
        tokenizer,
        tools=sample_tools,
        chat_template_kwargs={"thinking": True, "enable_thinking": True},
    )
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
        chat_template_kwargs={"thinking": True, "enable_thinking": True},
    )
    request._grammar_from_tool_parser = True
    full_text = build_tool_call(
        "get_current_weather",
        {"city": "Boston", "state": "MA", "unit": "fahrenheit"},
    )

    deltas = []
    for start in range(0, len(full_text), 4):
        chunk = full_text[start : start + 4]
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[start + 100],
            request=request,
            prompt_token_ids=[],
            finished=start + 4 >= len(full_text),
        )
        if delta is not None:
            deltas.append(delta)

    assert all(delta.reasoning is None for delta in deltas)
    assert "".join(delta.content or "" for delta in deltas) == ""
    names = [
        tool_call.function.name
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.name
    ]
    assert names == ["get_current_weather"]
    assert json.loads(reconstruct_args(deltas)) == {
        "city": "Boston",
        "state": "MA",
        "unit": "fahrenheit",
    }


def test_extract_tool_calls_arguments_wrapper():
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {}

    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )

    parser = DeepSeekV4ToolParser(mock_tokenizer, tools=[tool])
    request = MagicMock()
    request.tools = [tool]

    model_output = (
        f"{TC_START}"
        f'{INV_START}get_weather">'
        f'{PARAM_START}arguments" string="false">{{"location":"Beijing"}}{PARAM_END}'
        f"{INV_END}"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"location": "Beijing"}


@pytest.mark.skip_global_cleanup
def test_composed_schema_converts_object_and_array_params():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "wait": {
                        "anyOf": [
                            {"type": "object"},
                            {"type": "null"},
                        ],
                    },
                    "patches": {
                        "allOf": [
                            {"type": "array", "items": {"type": "object"}},
                        ],
                    },
                },
            },
        },
    )
    parser = make_parser(tools=[tool])
    request = make_request(tools=[tool])
    model_output = (
        f"{TC_START}\n"
        f'{INV_START}set_timer">\n'
        f'{PARAM_START}wait" string="false">'
        f'{{"type":"for","minutes":2880}}'
        f"{PARAM_END}\n"
        f'{PARAM_START}patches" string="false">'
        f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {
        "wait": {"type": "for", "minutes": 2880},
        "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
    }
