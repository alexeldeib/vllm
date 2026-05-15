# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import functools
import json
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from inspect import Parameter, signature

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolChoiceFunction,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from pydantic import TypeAdapter, ValidationError

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.request_utils import (
    extract_reasoning_with_machine_output_contract,
    output_is_exact_reasoning_boundary,
    output_starts_with_reasoning_boundary,
    request_allows_auto_tool_output,
    request_has_machine_output_contract,
    request_requires_tool_output,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.streaming import (
    extract_named_tool_call_streaming,
    extract_required_tool_call_streaming,
)
from vllm.tool_parsers.utils import Tool
from vllm.utils import random_uuid

logger = init_logger(__name__)


@functools.cache
def _parse_delta_cls_accepts_finished(parser_cls: type["Parser"]) -> bool | None:
    try:
        parse_delta = parser_cls.parse_delta
    except AttributeError:
        return None
    return _parse_delta_signature_accepts_finished(parse_delta)


def _parse_delta_signature_accepts_finished(parse_delta: object) -> bool:
    try:
        parameters = signature(parse_delta).parameters
    except (TypeError, ValueError):
        return True
    return "finished" in parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def parse_delta_with_optional_finished(
    parser: "Parser",
    *,
    delta_text: str,
    delta_token_ids: list[int],
    request: ChatCompletionRequest | ResponsesRequest,
    prompt_token_ids: list[int] | None = None,
    finished: bool = False,
) -> DeltaMessage | None:
    """Call Parser.parse_delta without breaking old custom parser signatures."""
    accepts_finished = _parse_delta_cls_accepts_finished(type(parser))
    if accepts_finished is None:
        accepts_finished = _parse_delta_signature_accepts_finished(parser.parse_delta)
    if accepts_finished:
        return parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=delta_token_ids,
            request=request,
            prompt_token_ids=prompt_token_ids,
            finished=finished,
        )
    return parser.parse_delta(
        delta_text=delta_text,
        delta_token_ids=delta_token_ids,
        request=request,
        prompt_token_ids=prompt_token_ids,
    )


@dataclass
class StreamState:
    """Mutable state for ``Parser.parse_delta()``. One per stream."""

    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    prompt_reasoning_checked: bool = False
    previous_text: str = ""
    previous_token_ids: list[int] = field(default_factory=list)
    history_tool_call_cnt: int = 0
    tool_call_id_type: str = "random"
    pending_reasoning_end: bool = False
    pending_reasoning_end_content_start: int | None = None
    pending_content_start_boundary: str | None = None
    pending_content_start_buffer: str = ""
    pending_content_start_token_ids: list[int] = field(default_factory=list)
    # only used for "required" and "named tool" choices,
    # tracks whether function name has been fully returned in the stream yet
    function_name_returned: bool = False


class Parser:
    """
    Abstract Parser class that unifies ReasoningParser and ToolParser into
    a single interface for parsing model output.

    This class provides a unified way to handle both reasoning extraction
    (e.g., chain-of-thought content in <think> tags) and tool call extraction
    (e.g., function calls in XML/JSON format) from model outputs.

    Subclasses can either:
    1. Override the abstract methods directly for custom parsing logic
    2. Set `reasoning_parser` and `tool_parser` properties to delegate to
       existing parser implementations

    Class Attributes:
        reasoning_parser_cls: The ReasoningParser class to use (for compatibility
            with code that needs the class, not instance).
        tool_parser_cls: The ToolParser class to use (for compatibility with
            code that needs the class, not instance).
    """

    # Class-level parser classes for compatibility with existing patterns
    # Subclasses should override these if they use specific parser classes
    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        """
        Initialize the Parser.

        Args:
            tokenizer: The tokenizer used by the model. This is required for
                token-based parsing operations.
        """
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None
        self._stream_state = StreamState()

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping from tokens to IDs."""
        return self.model_tokenizer.get_vocab()

    @property
    def reasoning_parser(self) -> ReasoningParser | None:
        """The underlying reasoning parser, if any."""
        return self._reasoning_parser

    @reasoning_parser.setter
    def reasoning_parser(self, parser: ReasoningParser | None) -> None:
        self._reasoning_parser = parser

    @property
    def tool_parser(self) -> ToolParser | None:
        """The underlying tool parser, if any."""
        return self._tool_parser

    @tool_parser.setter
    def tool_parser(self, parser: ToolParser | None) -> None:
        self._tool_parser = parser

    # ========== Reasoning Parser Methods ==========

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Used by structured engines like `xgrammar` to check if the
        reasoning content ends in the model output.

        Args:
            input_ids: The token IDs of the model output.

        Returns:
            True if the reasoning content ends in the input_ids.
        """

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """
        Check if the reasoning content ends during a decode step.

        Args:
            input_ids: The entire model output token IDs.
            delta_ids: The last few computed tokens at the current decode step.

        Returns:
            True if the reasoning content ends in the delta_ids.
        """
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token IDs from the input_ids.

        This extracts the non-reasoning content (e.g., everything after
        the </think> tag).

        Args:
            input_ids: The token IDs of the model output.

        Returns:
            The extracted content token IDs.
        """

    @abstractmethod
    def extract_response_outputs(
        self,
        *,
        model_output: str,
        model_output_token_ids: Sequence[int],
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        """
        Extract reasoning, content, and tool calls from a complete
        model-generated string and return as ResponseOutputItem objects.

        Used for non-streaming responses where we have the entire model
        response available before sending to the client.

        Args:
            model_output: The complete model-generated string.
            model_output_token_ids: The token IDs of the model output.
            request: The request object used to generate the output.
            enable_auto_tools: Whether to enable automatic tool call parsing.
            tool_call_id_type: Type of tool call ID generation ("random", etc).
            logprobs: Pre-computed logprobs for the output text, if any.

        Returns:
            A list of ResponseOutputItem objects.
        """

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model
        response available before sending to the client.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.

        Returns:
            A tuple of (reasoning, response_content).
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta message.

        Args:
            previous_text: Text from all previous tokens.
            current_text: Text including the current delta.
            delta_text: The new text in this delta.
            previous_token_ids: Token IDs from previous generation.
            current_token_ids: All token IDs including current.
            delta_token_ids: The new token IDs in this delta.

        Returns:
            A DeltaMessage with reasoning and/or content fields, or None.
        """

    # ========== Tool Parser Methods ==========

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """
        Adjust the request parameters for tool calling.

        Can be overridden by subclasses to modify request parameters
        (e.g., setting structured output schemas for tool calling).

        Args:
            request: The original request.

        Returns:
            The adjusted request.
        """
        return request

    @abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model-generated string.

        Used for non-streaming responses.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.

        Returns:
            ExtractedToolCallInformation containing the tool calls.
        """

    @abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from a streaming delta message.

        Args:
            previous_text: Text from all previous tokens.
            current_text: Text including the current delta.
            delta_text: The new text in this delta.
            previous_token_ids: Token IDs from previous generation.
            current_token_ids: All token IDs including current.
            delta_token_ids: The new token IDs in this delta.
            request: The request object.

        Returns:
            A DeltaMessage with tool_calls field, or None.
        """

    @abstractmethod
    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        finished: bool = False,
    ) -> DeltaMessage | None:
        """Parse a single streaming delta, orchestrating reasoning then
        tool call extraction via internal stream state.
        """


class DelegatingParser(Parser):
    """
    A Parser implementation that delegates to separate ReasoningParser and
    ToolParser instances.

    This is the recommended base class for creating model-specific parsers
    that combine existing reasoning and tool parser implementations.
    Subclasses should set `self._reasoning_parser` and `self._tool_parser`
    in their `__init__` method.

    If either parser is None, the corresponding methods will return default
    values (no reasoning extraction, no tool calls).
    """

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._reasoning_parser is None:
            return None, model_output
        return extract_reasoning_with_machine_output_contract(
            model_output=model_output,
            request=request,
            reasoning_parser=self._reasoning_parser,
        )

    def extract_response_outputs(
        self,
        *,
        model_output: str,
        model_output_token_ids: Sequence[int],
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        # First extract reasoning
        reasoning, content = self.extract_reasoning(model_output, request)

        # Then parse tool calls from the content
        tool_calls, content = self._parse_tool_calls(
            request=request,
            content=content,
            enable_auto_tools=enable_auto_tools,
        )

        # Build output items
        outputs: list[ResponseOutputItem] = []

        # Add reasoning item if present
        if reasoning:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning, type="reasoning_text")
                ],
                status=None,  # NOTE: Only the last output item has status.
            )
            outputs.append(reasoning_item)

        # Add message item if there's content
        if content:
            res_text_part = ResponseOutputText(
                text=content,
                annotations=[],
                type="output_text",
                logprobs=logprobs,
            )
            message_item = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[res_text_part],
                role="assistant",
                status="completed",
                type="message",
            )
            outputs.append(message_item)

        if tool_calls:
            # We use a simple counter for history_tool_call_count because
            # we don't track the history of tool calls in the Responses API yet.
            # This means that the tool call index will start from 0 for each
            # request.
            for history_tool_call_cnt, tool_call in enumerate(tool_calls):
                tool_call_item = ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()}",
                    call_id=tool_call.id
                    if tool_call.id
                    else make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=tool_call.name,
                        idx=history_tool_call_cnt,
                    ),
                    type="function_call",
                    status="completed",
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                outputs.append(tool_call_item)

        return outputs

    def _get_function_name(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> str:
        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            return request.tool_choice.name
        if request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            return request.tool_choice.function.name
        raise ValueError("Invalid tool_choice for function name extraction.")

    def _parse_tool_calls(
        self,
        request: ResponsesRequest,
        content: str | None,
        enable_auto_tools: bool,
    ) -> tuple[list[FunctionCall], str | None]:
        """
        TODO(qandrew): merge _parse_tool_calls_from_content
        for ChatCompletions into this function
        Parse tool calls from content based on request tool_choice settings.

        Returns:
            A tuple of (function_calls, remaining_content) if tool calls
            were parsed
        """
        function_calls: list[FunctionCall] = []

        if request.tool_choice and isinstance(
            request.tool_choice,
            (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam),
        ):
            # Forced Function Call
            if content is None:
                return [], None
            function_calls.append(
                FunctionCall(name=self._get_function_name(request), arguments=content)
            )
            return function_calls, None  # Clear content since tool is called.

        if request.tool_choice == "required":
            # Required tool calls - parse JSON
            tool_calls = []
            with contextlib.suppress(ValidationError):
                content = content or ""
                tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(
                    content
                )
            for tool_call in tool_calls:
                function_calls.append(
                    FunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                    )
                )
            return function_calls, None  # Clear content since tool is called.

        if (
            self._tool_parser is not None
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
            and request_allows_auto_tool_output(request)
        ):
            # Automatic Tool Call Parsing
            tool_call_info = self._tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                function_calls.extend(
                    FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                remaining_content = tool_call_info.content
                if remaining_content and remaining_content.strip() == "":
                    remaining_content = None
                return function_calls, remaining_content

        # No tool calls
        return [], content

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if self._reasoning_parser is not None:
            request = self._reasoning_parser.adjust_request(request)
        if self._tool_parser is not None:
            request = self._tool_parser.adjust_request(request)
        return request

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._reasoning_parser is None:
            return DeltaMessage(content=delta_text)
        return self._reasoning_parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self._tool_parser is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return self._tool_parser.extract_tool_calls(model_output, request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._tool_parser is None:
            return None
        return self._tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    def _extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
        # The following parameters are used for "required" tool choice parsing and are
        # tracked in StreamState for streaming parsing.
        tool_call_idx: int | None = None,
        tool_call_id_type: str = "random",
        function_name_returned: bool = False,
    ) -> tuple[DeltaMessage | None, bool]:
        assert self._tool_parser is not None
        supports_required_and_named = self._tool_parser.supports_required_and_named
        if (
            supports_required_and_named
            and request.tool_choice
            and isinstance(
                request.tool_choice,
                (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam),
            )
        ):
            delta_message, function_name_returned = extract_named_tool_call_streaming(
                delta_text=delta_text,
                function_name=self._get_function_name(request),
                function_name_returned=function_name_returned,
                tool_call_idx=tool_call_idx,
                tool_call_id_type=tool_call_id_type,
                tokenizer=self.model_tokenizer,
            )
            return delta_message, function_name_returned

        if supports_required_and_named and request.tool_choice == "required":
            delta_message, function_name_returned = (
                extract_required_tool_call_streaming(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    function_name_returned=function_name_returned,
                    tool_call_idx=tool_call_idx,
                    tool_call_id_type=tool_call_id_type,
                )
            )
            return delta_message, function_name_returned
        return self.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,  # type: ignore[arg-type]
        ), False

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self._reasoning_parser is None:
            return False
        return self._reasoning_parser.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._reasoning_parser is None:
            return input_ids
        return self._reasoning_parser.extract_content_ids(input_ids)

    def _in_reasoning_phase(self, state: StreamState) -> bool:
        if self._reasoning_parser is None:
            return False
        return not state.reasoning_ended

    def _in_tool_call_phase(self, state: StreamState) -> bool:
        if self._tool_parser is None:
            return False
        return state.reasoning_ended

    def _machine_output_starts_in_content_phase(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        current_text: str,
        *,
        allow_prefix: bool,
    ) -> bool:
        if self._reasoning_parser is None:
            return False
        if not request_has_machine_output_contract(request):
            return False
        if not current_text.strip():
            return False
        return not output_starts_with_reasoning_boundary(
            current_text, self._reasoning_parser, allow_prefix=allow_prefix
        )

    def _reasoning_end_text_visible(self, text: str) -> bool:
        if self._reasoning_parser is None:
            return False
        for boundary in self._reasoning_parser.reasoning_end_strs:
            if boundary and boundary in text:
                return True
        return False

    def _split_primary_reasoning_end_text(self, text: str) -> tuple[str, str] | None:
        if self._reasoning_parser is None:
            return None
        boundary = self._reasoning_parser.reasoning_end_str
        if not boundary:
            return None
        idx = text.find(boundary)
        if idx == -1:
            return None
        return text[:idx], text[idx + len(boundary) :]

    def _is_reasoning_end_prefix(self, text: str) -> bool:
        if self._reasoning_parser is None:
            return False
        boundary = self._reasoning_parser.reasoning_end_str
        return bool(
            boundary and text and boundary.startswith(text) and text != boundary
        )

    def _reasoning_end_token_delta_index(
        self, delta_token_ids: list[int]
    ) -> int | None:
        if self._reasoning_parser is None:
            return None
        for attr_name in ("end_token_id", "think_end_token_id"):
            token_id = getattr(self._reasoning_parser, attr_name, None)
            if isinstance(token_id, int) and token_id in delta_token_ids:
                return delta_token_ids.index(token_id) + 1
        return None

    def _reasoning_end_delta_index(
        self,
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> int | None:
        if self._reasoning_parser is None:
            return None
        content_ids = self.extract_content_ids(current_token_ids)
        if (
            content_ids
            and delta_token_ids
            and current_token_ids[-len(content_ids) :] == content_ids
        ):
            content_delta_len = min(len(content_ids), len(delta_token_ids))
            if content_ids[-content_delta_len:] == delta_token_ids[-content_delta_len:]:
                return len(delta_token_ids) - content_delta_len
        token_delta_index = self._reasoning_end_token_delta_index(delta_token_ids)
        if token_delta_index is not None:
            return token_delta_index
        previous_len = len(current_token_ids) - len(delta_token_ids)
        for idx in range(1, len(delta_token_ids) + 1):
            if self._reasoning_parser.is_reasoning_end_streaming(
                current_token_ids[: previous_len + idx],
                delta_token_ids[:idx],
            ):
                return idx
        return None

    def _decode_visible_delta(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.model_tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return self.model_tokenizer.decode(token_ids)

    def _decode_raw_delta(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.model_tokenizer.decode(token_ids, skip_special_tokens=False)
        except TypeError:
            return self.model_tokenizer.decode(token_ids)

    def _delta_token_before_content_is_reasoning_end(
        self,
        delta_token_ids: list[int],
        content_start_index: int,
    ) -> bool:
        if self._reasoning_parser is None or content_start_index <= 0:
            return False
        boundary = self._reasoning_parser.reasoning_end_str
        if not boundary:
            return False
        token_text = self._decode_raw_delta([delta_token_ids[content_start_index - 1]])
        return token_text == boundary

    def _pending_content_start_boundary(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        content_start_index: int,
    ) -> str | None:
        if (
            self._reasoning_parser is None
            or content_start_index < 0
            or content_start_index >= len(delta_token_ids)
            or self._delta_token_before_content_is_reasoning_end(
                delta_token_ids, content_start_index
            )
        ):
            return None
        raw_content = self._decode_raw_delta(delta_token_ids[content_start_index:])
        visible_content = self._decode_visible_delta(
            delta_token_ids[content_start_index:]
        )
        primary_boundary = self._reasoning_parser.reasoning_end_str
        for boundary in self._reasoning_parser.reasoning_end_strs:
            if not boundary or boundary == primary_boundary:
                continue
            if (
                raw_content.startswith(boundary)
                and visible_content == raw_content[len(boundary) :]
            ):
                return boundary
        return None

    def _delayed_content_start_boundary_index(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        boundary: str,
    ) -> int | None:
        visible_delta = self._decode_visible_delta(delta_token_ids)
        boundary_index = delta_text.find(boundary)
        while boundary_index >= 0:
            without_boundary = (
                delta_text[:boundary_index]
                + delta_text[boundary_index + len(boundary) :]
            )
            if without_boundary == visible_delta:
                return boundary_index
            boundary_index = delta_text.find(boundary, boundary_index + 1)
        return None

    def _extract_reasoning_delta_with_reasoning_parser(
        self,
        *,
        state: StreamState,
        delta_text: str,
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        if self._reasoning_parser is None or not delta_text:
            return None
        return self.extract_reasoning_streaming(
            previous_text=state.previous_text,
            current_text=state.previous_text + delta_text,
            delta_text=delta_text,
            previous_token_ids=state.previous_token_ids,
            current_token_ids=state.previous_token_ids + delta_token_ids,
            delta_token_ids=delta_token_ids,
        )

    def _split_hidden_reasoning_end_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        reasoning_end_delta_index: int,
    ) -> tuple[str, str]:
        content_start_index = reasoning_end_delta_index
        has_end_token_before_content = (
            self._delta_token_before_content_is_reasoning_end(
                delta_token_ids, content_start_index
            )
        )
        before_end_ids = (
            delta_token_ids[: max(content_start_index - 1, 0)]
            if has_end_token_before_content
            else delta_token_ids[:content_start_index]
        )
        after_end_ids = delta_token_ids[content_start_index:]
        before_text = self._decode_visible_delta(before_end_ids)
        after_text = self._decode_visible_delta(after_end_ids)

        if before_text or after_text:
            combined = before_text + after_text
            if combined == delta_text:
                return before_text, after_text
            if before_text and delta_text.startswith(before_text):
                return before_text, delta_text[len(before_text) :]
            if after_text and delta_text.endswith(after_text):
                return delta_text[: -len(after_text)], after_text

        if after_end_ids and not before_end_ids:
            return "", delta_text
        return delta_text, ""

    def _visible_end_token_boundary(self) -> str | None:
        if self._reasoning_parser is None:
            return None
        boundary = self._reasoning_parser.reasoning_end_str
        if not boundary:
            return None
        try:
            end_token = self._reasoning_parser.end_token
        except (AttributeError, NotImplementedError):
            return None
        if end_token != boundary:
            return None
        return boundary

    def _extract_hidden_end_with_reasoning_parser(
        self,
        *,
        state: StreamState,
        reasoning_delta: str | None,
        content_delta: str | None,
        delta_text: str,
        delta_token_ids: list[int],
        reasoning_end_delta_index: int,
    ) -> DeltaMessage | None:
        primary_boundary_before_content = (
            self._delta_token_before_content_is_reasoning_end(
                delta_token_ids, reasoning_end_delta_index
            )
        )
        if primary_boundary_before_content:
            boundary = self._visible_end_token_boundary()
            if boundary is None:
                return None
            parser_delta_text = (
                (reasoning_delta or "") + boundary + (content_delta or "")
            )
        else:
            parser_delta_text = delta_text
        if not parser_delta_text:
            return None
        delta_message = self.extract_reasoning_streaming(
            previous_text=state.previous_text,
            current_text=state.previous_text + parser_delta_text,
            delta_text=parser_delta_text,
            previous_token_ids=state.previous_token_ids,
            current_token_ids=state.previous_token_ids + delta_token_ids,
            delta_token_ids=delta_token_ids,
        )
        if primary_boundary_before_content:
            return delta_message
        if delta_message is not None and delta_message.content is not None:
            return delta_message
        return None

    def _extract_after_hidden_end_with_reasoning_parser(
        self,
        *,
        state: StreamState,
        content_delta: str,
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        boundary = self._visible_end_token_boundary()
        if boundary is None:
            return None
        previous_text = state.previous_text + boundary
        previous_token_ids = list(state.previous_token_ids)
        end_token_id = getattr(self._reasoning_parser, "end_token_id", None)
        if isinstance(end_token_id, int):
            previous_token_ids.append(end_token_id)
        return self.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=previous_text + content_delta,
            delta_text=content_delta,
            previous_token_ids=previous_token_ids,
            current_token_ids=previous_token_ids + delta_token_ids,
            delta_token_ids=delta_token_ids,
        )

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        finished: bool = False,
    ) -> DeltaMessage | None:
        state = self._stream_state
        machine_output_contract = request_has_machine_output_contract(request)
        tool_output_contract = request_requires_tool_output(request)
        tool_parser_output_contract = (
            tool_output_contract or request_allows_auto_tool_output(request)
        )

        if not state.prompt_reasoning_checked:
            if self._reasoning_parser is None:
                state.prompt_reasoning_checked = True
                state.reasoning_ended = True
            elif machine_output_contract:
                state.prompt_reasoning_checked = True
            elif prompt_token_ids is not None:
                state.prompt_reasoning_checked = True
                if self.is_reasoning_end(prompt_token_ids):
                    state.reasoning_ended = True

        current_text = state.previous_text + delta_text
        current_token_ids = state.previous_token_ids + delta_token_ids
        delta_message: DeltaMessage | None = None
        machine_output_started_content_phase = False

        if (
            self._reasoning_parser is not None
            and machine_output_contract
            and not state.reasoning_ended
            and not current_text.strip()
        ):
            if finished:
                if current_token_ids:
                    decoded_current_text = self.model_tokenizer.decode(
                        current_token_ids, skip_special_tokens=False
                    )
                    if decoded_current_text and (
                        not current_text
                        or (
                            not current_text.strip()
                            and decoded_current_text.startswith(current_text)
                            and decoded_current_text.strip()
                        )
                    ):
                        delta_text = (
                            decoded_current_text[len(state.previous_text) :]
                            if (
                                state.previous_text
                                and decoded_current_text.startswith(state.previous_text)
                            )
                            else decoded_current_text
                        )
                        current_text = decoded_current_text
                if not current_text.strip():
                    state.reasoning_ended = True
                    machine_output_started_content_phase = True
            else:
                state.previous_text = current_text
                state.previous_token_ids = current_token_ids
                return None

        if (
            self._reasoning_parser is not None
            and machine_output_contract
            and not state.reasoning_ended
            and finished
            and output_is_exact_reasoning_boundary(current_text, self._reasoning_parser)
        ):
            state.reasoning_ended = True
            machine_output_started_content_phase = True

        if (
            self._reasoning_parser is not None
            and machine_output_contract
            and not state.reasoning_ended
            and not finished
            and output_starts_with_reasoning_boundary(
                current_text, self._reasoning_parser, allow_prefix=True
            )
            and not output_starts_with_reasoning_boundary(
                current_text, self._reasoning_parser
            )
        ):
            state.previous_text = current_text
            state.previous_token_ids = current_token_ids
            return None

        if not state.reasoning_ended and self._machine_output_starts_in_content_phase(
            request,
            current_text,
            allow_prefix=False,
        ):
            state.reasoning_ended = True
            machine_output_started_content_phase = True

        # Reasoning extraction
        if self._in_reasoning_phase(state):
            reasoning_end_token_delta_index = self._reasoning_end_token_delta_index(
                delta_token_ids
            )
            if state.pending_content_start_boundary is not None:
                boundary = state.pending_content_start_boundary
                boundary_index = self._delayed_content_start_boundary_index(
                    delta_text, delta_token_ids, boundary
                )
                if boundary_index is None and not finished:
                    state.pending_content_start_buffer += delta_text
                    state.pending_content_start_token_ids.extend(delta_token_ids)
                    return None
                if boundary_index is None:
                    content_delta = (
                        boundary + state.pending_content_start_buffer + delta_text
                    )
                else:
                    content_delta = (
                        delta_text[boundary_index : boundary_index + len(boundary)]
                        + state.pending_content_start_buffer
                        + delta_text[:boundary_index]
                        + delta_text[boundary_index + len(boundary) :]
                    )
                delta_message = DeltaMessage(
                    content=content_delta or None,
                )
                reasoning_ended = True
                current_text = content_delta
                current_token_ids = (
                    state.pending_content_start_token_ids + delta_token_ids
                )
                delta_token_ids = current_token_ids
                delta_text = content_delta
                state.pending_content_start_boundary = None
                state.pending_content_start_buffer = ""
                state.pending_content_start_token_ids = []

            if (
                self._reasoning_parser is not None
                and not state.pending_reasoning_end
                and state.pending_content_start_boundary is None
                and delta_message is None
                and (
                    reasoning_end_token_delta_index == len(delta_token_ids)
                    or self._reasoning_parser.is_reasoning_end_streaming(
                        current_token_ids, delta_token_ids
                    )
                )
            ):
                reasoning_end_delta_index = self._reasoning_end_delta_index(
                    current_token_ids,
                    delta_token_ids,
                )
                reasoning_delta: str | None = None
                content_delta: str | None = None
                pending_boundary_finished = False
                if reasoning_end_delta_index is not None and delta_text:
                    pending_boundary = self._pending_content_start_boundary(
                        delta_text, delta_token_ids, reasoning_end_delta_index
                    )
                    if pending_boundary is not None:
                        reasoning_delta, content_delta = (
                            self._split_hidden_reasoning_end_delta(
                                delta_text,
                                delta_token_ids,
                                reasoning_end_delta_index,
                            )
                        )
                        if finished:
                            content_delta = pending_boundary + (content_delta or "")
                            delta_message = DeltaMessage(
                                reasoning=reasoning_delta or None,
                                content=content_delta or None,
                            )
                            reasoning_ended = True
                            current_text = content_delta
                            current_token_ids = delta_token_ids[
                                reasoning_end_delta_index:
                            ]
                            delta_token_ids = current_token_ids
                            delta_text = content_delta
                            pending_boundary_finished = True
                        else:
                            state.pending_content_start_boundary = pending_boundary
                            state.pending_content_start_buffer += content_delta
                            state.pending_content_start_token_ids.extend(
                                delta_token_ids[reasoning_end_delta_index:]
                            )
                            current_token_ids = (
                                state.previous_token_ids
                                + delta_token_ids[:reasoning_end_delta_index]
                            )
                            cleaned_delta = (
                                self._extract_reasoning_delta_with_reasoning_parser(
                                    state=state,
                                    delta_text=reasoning_delta,
                                    delta_token_ids=delta_token_ids[
                                        :reasoning_end_delta_index
                                    ],
                                )
                                if reasoning_delta
                                else None
                            )
                            state.previous_text = state.previous_text + (
                                reasoning_delta or ""
                            )
                            state.previous_token_ids = current_token_ids
                            if reasoning_delta:
                                if (
                                    cleaned_delta is not None
                                    and cleaned_delta.content is None
                                ):
                                    return cleaned_delta
                                return DeltaMessage(reasoning=reasoning_delta)
                            return None
                    if not pending_boundary_finished:
                        reasoning_delta, content_delta = (
                            self._split_hidden_reasoning_end_delta(
                                delta_text,
                                delta_token_ids,
                                reasoning_end_delta_index,
                            )
                        )

                if (
                    not pending_boundary_finished
                    and reasoning_delta is not None
                    and self._reasoning_end_text_visible(reasoning_delta)
                ):
                    reasoning_end_delta_index = None
                    reasoning_delta = None
                    content_delta = None
                elif not pending_boundary_finished:
                    state.pending_reasoning_end = True

                if pending_boundary_finished:
                    pass
                elif state.pending_reasoning_end and reasoning_end_delta_index is None:
                    state.pending_reasoning_end_content_start = len(current_text)
                    current_token_ids = state.previous_token_ids
                elif state.pending_reasoning_end and delta_text:
                    content_delta_started_with_visible_boundary = False
                    visible_content_split = (
                        self._split_primary_reasoning_end_text(content_delta)
                        if content_delta
                        else None
                    )
                    if (
                        visible_content_split is not None
                        and visible_content_split[0] == ""
                    ):
                        content_delta = visible_content_split[1]
                        content_delta_started_with_visible_boundary = True
                    hidden_end_delta_message = (
                        self._extract_hidden_end_with_reasoning_parser(
                            state=state,
                            reasoning_delta=reasoning_delta,
                            content_delta=content_delta,
                            delta_text=delta_text,
                            delta_token_ids=delta_token_ids,
                            reasoning_end_delta_index=reasoning_end_delta_index,
                        )
                    )
                    if hidden_end_delta_message is not None:
                        delta_message = hidden_end_delta_message
                        reasoning_ended = (
                            bool(delta_message.content)
                            or content_delta_started_with_visible_boundary
                        )
                    else:
                        delta_message = DeltaMessage(
                            reasoning=reasoning_delta or None,
                            content=content_delta or None,
                        )
                        reasoning_ended = (
                            bool(content_delta)
                            or content_delta_started_with_visible_boundary
                        )
                    if (
                        content_delta
                        and not finished
                        and self._is_reasoning_end_prefix(content_delta)
                    ):
                        buffered_reasoning_delta = (
                            hidden_end_delta_message.reasoning
                            if hidden_end_delta_message is not None
                            else reasoning_delta
                        )
                        delta_message = (
                            DeltaMessage(reasoning=buffered_reasoning_delta)
                            if buffered_reasoning_delta
                            else None
                        )
                        reasoning_ended = False
                        state.pending_reasoning_end_content_start = len(
                            current_text
                        ) - len(content_delta)
                    elif content_delta or content_delta_started_with_visible_boundary:
                        state.pending_reasoning_end = False
                        state.pending_reasoning_end_content_start = None
                    else:
                        state.pending_reasoning_end_content_start = len(current_text)
                        current_token_ids = (
                            state.previous_token_ids
                            + delta_token_ids[: max(reasoning_end_delta_index - 1, 0)]
                        )
                elif state.pending_reasoning_end:
                    state.pending_reasoning_end_content_start = len(current_text)
                    current_token_ids = state.previous_token_ids
                    state.previous_text = current_text
                    state.previous_token_ids = current_token_ids
                    return None

            if delta_message is None:
                reasoning_end_split = None
                content_start = state.pending_reasoning_end_content_start
                if state.pending_reasoning_end:
                    pending_content = (
                        current_text[content_start:]
                        if content_start is not None
                        else current_text
                    )
                    pending_split = self._split_primary_reasoning_end_text(
                        pending_content
                    )
                    if pending_split is not None and pending_split[0] == "":
                        reasoning_text = (
                            current_text[:content_start]
                            if content_start is not None
                            else ""
                        )
                        reasoning_end_split = (reasoning_text, pending_split[1])
                if state.pending_reasoning_end and reasoning_end_split is None:
                    pending_content = (
                        current_text[content_start:]
                        if content_start is not None
                        else current_text
                    )
                    if not finished and self._is_reasoning_end_prefix(pending_content):
                        state.previous_text = current_text
                        state.previous_token_ids = current_token_ids
                        return None
                    content_delta = (
                        current_text[content_start:]
                        if content_start is not None
                        else delta_text
                    )
                    delta_message = (
                        self._extract_after_hidden_end_with_reasoning_parser(
                            state=state,
                            content_delta=content_delta,
                            delta_token_ids=delta_token_ids,
                        )
                        or DeltaMessage(content=content_delta or None)
                    )
                    reasoning_ended = True
                    state.pending_reasoning_end = False
                    state.pending_reasoning_end_content_start = None
                elif reasoning_end_split is not None:
                    reasoning_text, content = reasoning_end_split
                    new_reasoning = (
                        reasoning_text[len(state.previous_text) :]
                        if reasoning_text.startswith(state.previous_text)
                        else ""
                    )
                    delta_message = (
                        self._extract_after_hidden_end_with_reasoning_parser(
                            state=state,
                            content_delta=content,
                            delta_token_ids=delta_token_ids,
                        )
                        if not new_reasoning
                        else None
                    ) or DeltaMessage(
                        reasoning=new_reasoning or None, content=content or None
                    )
                    reasoning_ended = True
                    state.pending_reasoning_end = False
                    state.pending_reasoning_end_content_start = None
                else:
                    delta_message = self.extract_reasoning_streaming(
                        previous_text=state.previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=state.previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                    )
                    reasoning_ended = self.is_reasoning_end(
                        current_token_ids
                    ) or self._reasoning_end_text_visible(current_text)
            if delta_message is not None and reasoning_ended:
                state.reasoning_ended = True
            if (
                delta_message is None
                and finished
                and self._reasoning_parser is not None
                and output_starts_with_reasoning_boundary(
                    current_text, self._reasoning_parser, allow_prefix=True
                )
                and not output_starts_with_reasoning_boundary(
                    current_text, self._reasoning_parser
                )
            ):
                delta_message = DeltaMessage(reasoning=current_text)
            # Hand off remaining content to tool parser
            if (
                self._tool_parser
                and self._in_tool_call_phase(state)
                and (not machine_output_contract or tool_parser_output_contract)
                and delta_message is not None
                and state.reasoning_ended
            ):
                current_token_ids = (
                    self.extract_content_ids(current_token_ids)
                    if self.is_reasoning_end(current_token_ids)
                    else delta_token_ids
                )
                if delta_message and delta_message.content:
                    current_text = delta_message.content
                    delta_message.content = None
                else:
                    current_text = ""

        # Tool call extraction
        tool_call_phase_active = self._in_tool_call_phase(state) and (
            not machine_output_contract or tool_parser_output_contract
        )
        if tool_call_phase_active:
            if not state.tool_call_text_started:
                state.tool_call_text_started = True
                state.previous_text = ""
                state.previous_token_ids = []
                delta_text = current_text
                delta_token_ids = current_token_ids

            delta_message, state.function_name_returned = (
                self._extract_tool_calls_streaming(
                    previous_text=state.previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=state.previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                    request=request,  # type: ignore[arg-type]
                    tool_call_idx=state.history_tool_call_cnt,
                    tool_call_id_type=state.tool_call_id_type,
                    function_name_returned=state.function_name_returned,
                )
            )
            if (
                delta_message
                and delta_message.tool_calls
                and delta_message.tool_calls[0].id is not None
            ):
                state.history_tool_call_cnt += 1

        # No phase active: pass through as content
        if (
            delta_message is None
            and not self._in_reasoning_phase(state)
            and not tool_call_phase_active
        ):
            delta_message = DeltaMessage(
                content=current_text
                if machine_output_started_content_phase
                else delta_text
            )

        state.previous_text = current_text
        state.previous_token_ids = current_token_ids
        return delta_message


class _WrappedParser(DelegatingParser):
    """
    A DelegatingParser subclass that instantiates parsers from class attributes.

    This class is used to dynamically create a parser that wraps individual
    ReasoningParser and ToolParser classes. The class attributes
    `reasoning_parser_cls` and `tool_parser_cls` should be set before
    instantiation.

    Usage:
        _WrappedParser.reasoning_parser_cls = MyReasoningParser
        _WrappedParser.tool_parser_cls = MyToolParser
        parser = _WrappedParser(tokenizer)
    """

    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(
        self, tokenizer: TokenizerLike, tools: list[Tool] | None = None, **kwargs
    ):
        super().__init__(tokenizer)
        # Instantiate the underlying parsers from class attributes
        if self.__class__.reasoning_parser_cls is not None:
            self._reasoning_parser = self.__class__.reasoning_parser_cls(
                tokenizer, **kwargs
            )
        if self.__class__.tool_parser_cls is not None:
            self._tool_parser = self.__class__.tool_parser_cls(tokenizer, tools)
