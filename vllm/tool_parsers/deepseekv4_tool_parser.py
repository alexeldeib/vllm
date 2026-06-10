# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    DeepSeek V4 DSML tool parser.

    V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
    ``<｜DSML｜tool_calls>`` instead of ``<｜DSML｜function_calls>``.
    """

    # DeepSeek-V4 emits native DSML tool-call blocks, not the generic JSON
    # array vLLM's standard required/named tool_choice path expects.
    supports_required_and_named = False

    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        structure_tag = None
        chat_request = None
        if (
            isinstance(request, ChatCompletionRequest)
            and request.tools
            and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
            )
        ):
            chat_request = request
            structure_tag = self.get_structural_tag(chat_request)

        if structure_tag is None:
            request = super().adjust_request(request)
        else:
            structural_tag = json.dumps(structure_tag.model_dump())
            assert chat_request is not None
            if chat_request.structured_outputs is not None:
                # Rebuild so mutually exclusive constraints are dropped while
                # the whitespace knobs survive.
                chat_request.structured_outputs = StructuredOutputsParams(
                    structural_tag=structural_tag,
                    disable_any_whitespace=(
                        chat_request.structured_outputs.disable_any_whitespace
                    ),
                    disable_additional_properties=(
                        chat_request.structured_outputs.disable_additional_properties
                    ),
                    whitespace_pattern=(
                        chat_request.structured_outputs.whitespace_pattern
                    ),
                )
            else:
                chat_request.structured_outputs = StructuredOutputsParams(
                    structural_tag=structural_tag
                )
            chat_request.response_format = None
            chat_request._grammar_from_tool_parser = True
            request = chat_request

        if request.tools and request.tool_choice != "none":
            # Ensure DSML tool-call markers are decoded as literal text so the
            # text-based parser can recover them.
            request.skip_special_tokens = False
        return request

    def get_structural_tag(self, request: ChatCompletionRequest):
        chat_template_kwargs = request.chat_template_kwargs or {}
        thinking = bool(
            chat_template_kwargs.get("thinking")
            or chat_template_kwargs.get("enable_thinking")
        )
        return get_model_structural_tag(
            model="deepseek_v4",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=thinking,
        )
