# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import TYPE_CHECKING

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.tool_parsers.utils import partial_json_loads
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object


def _bracket_level(s: str, opening: str = "{", closing: str = "}") -> int:
    """Calculate the current level of nested brackets in a string."""
    level = 0
    for char in s:
        if char == opening:
            level += 1
        elif char == closing:
            level -= 1
    return level


def _consume_space(s: str, index: int) -> int:
    while index < len(s) and s[index].isspace():
        index += 1
    return index


def _consume_json_string(s: str, index: int) -> tuple[str, int] | None:
    if index >= len(s) or s[index] != '"':
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(s[index:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, str):
        return None
    return value, index + end


def _find_last_top_level_parameters_start(s: str) -> int | None:
    """Find the latest tool-call object's top-level parameters value."""
    array_level = 0
    object_level = 0
    parameters_start: int | None = None
    index = 0

    while index < len(s):
        char = s[index]

        if char == '"':
            parsed_string = _consume_json_string(s, index)
            if parsed_string is None:
                return parameters_start

            value, end = parsed_string
            after_key = _consume_space(s, end)
            if (
                value == "parameters"
                and array_level == 1
                and object_level == 1
                and after_key < len(s)
                and s[after_key] == ":"
            ):
                parameters_start = _consume_space(s, after_key + 1)

            index = end
            continue

        if char == "[":
            array_level += 1
        elif char == "]":
            array_level -= 1
        elif char == "{":
            object_level += 1
        elif char == "}":
            object_level -= 1

        index += 1

    return parameters_start


def _trim_to_json_value(s: str) -> str:
    """Return the complete leading JSON value, or the input if incomplete."""
    lstripped = s.lstrip()
    if not lstripped:
        return s

    leading_ws = len(s) - len(lstripped)
    try:
        _, end = json.JSONDecoder().raw_decode(lstripped)
    except json.JSONDecodeError:
        return s

    return s[: leading_ws + end]


def _parameters_text(s: str) -> str | None:
    parameters_start = _find_last_top_level_parameters_start(s)
    if parameters_start is None:
        return None
    return _trim_to_json_value(s[parameters_start:])


def _parameters_delta(previous_text: str, current_text: str) -> str | None:
    current = _parameters_text(current_text)
    if current is None:
        return None

    previous = _parameters_text(previous_text) or ""
    if current.startswith(previous):
        return current[len(previous) :]

    return current


def filter_delta_text(
    delta_text: str,
    previous_text: str,
) -> tuple[str, bool]:
    """Trim trailing tool-list delimiters from required-tool streaming text."""
    bracket_level = _bracket_level(previous_text)
    updated_delta = ""
    passed_zero = False
    for char in delta_text:
        if char == "{":
            bracket_level += 1
            passed_zero = bracket_level == 0
        elif char == "}":
            bracket_level -= 1
            passed_zero = bracket_level == 0

        if bracket_level != 0:
            updated_delta += char
        else:
            if char == ",":
                break
    return updated_delta, passed_zero


def extract_named_tool_call_streaming(
    *,
    delta_text: str,
    function_name: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
    tokenizer: "TokenizerLike",
    tool_call_array_index: int = 0,
) -> tuple[DeltaMessage | None, bool]:
    """Build a streaming tool-call delta for forced named tool choice."""
    if function_name_returned:
        delta_tool_call = DeltaToolCall(
            function=DeltaFunctionCall(arguments=delta_text),
            index=tool_call_array_index,
        )
    else:
        if is_mistral_tokenizer(tokenizer):
            tool_call_id = MistralToolCall.generate_random_id()
        else:
            tool_call_id = make_tool_call_id(
                id_type=tool_call_id_type,
                func_name=function_name,
                idx=tool_call_idx,
            )
        delta_tool_call = DeltaToolCall(
            id=tool_call_id,
            type="function",
            function=DeltaFunctionCall(
                name=function_name,
                arguments=delta_text,
            ),
            index=tool_call_array_index,
        )
        function_name_returned = True
    return (
        DeltaMessage(tool_calls=[delta_tool_call]),
        function_name_returned,
    )


def extract_required_tool_call_streaming(
    *,
    previous_text: str,
    current_text: str | None,
    delta_text: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, bool]:
    if current_text is None or current_text == "":
        # if the current text is empty, we cannot parse it
        return None, function_name_returned
    try:
        flags = Allow.ALL
        obj, _ = partial_json_loads(current_text, flags)
    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
    ):
        obj = None

    # check if the current text is a valid array
    # containing a partial tool calling object
    # if not repeat
    if obj is None or not isinstance(obj, list) or not len(obj) > 0:
        function_name_returned = False
        delta_message = None
    else:
        _, finishes_previous_tool = filter_delta_text(delta_text, previous_text)
        # take the last tool call from the generated list
        current_tool_call = obj[-1]

        # once parameters have been generated the name is complete as well
        if not finishes_previous_tool and (
            "name" not in current_tool_call or "parameters" not in current_tool_call
        ):
            function_name_returned = False
            delta_message = None
        else:
            if not function_name_returned:
                # get partly generated arguments from the latest tool call
                arguments = _parameters_delta(
                    previous_text=previous_text,
                    current_text=current_text,
                )
                if arguments is None:
                    arguments = ""
                arguments, _ = filter_delta_text(arguments, previous_text)

                # if this iteration finishes a previous tool call but a
                # new incomplete tool is already generated, take the
                # previous from the list
                if finishes_previous_tool and "parameters" not in current_tool_call:
                    current_tool_call = obj[-2]

                function_name_returned = True
                tool_call_id = make_tool_call_id(
                    id_type=tool_call_id_type,
                    func_name=current_tool_call["name"],
                    idx=tool_call_idx,
                )
                delta_message = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            id=tool_call_id,
                            function=DeltaFunctionCall(
                                name=current_tool_call["name"], arguments=arguments
                            ),
                            index=len(obj) - 1,
                            type="function",
                        )
                    ]
                )

            else:
                delta_text = _parameters_delta(previous_text, current_text) or ""
                delta_text, _ = filter_delta_text(delta_text, previous_text)

                if delta_text != "":
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                    name=None,
                                    arguments=delta_text,
                                ),
                                index=len(obj) - 1,
                            )
                        ]
                    )
                else:
                    delta_message = None

    return delta_message, function_name_returned
