# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)

_PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR = "_vllm_machine_output_contract"
_PRESERVED_STRUCTURED_TEXT_OUTPUT_CONTRACT_ATTR = (
    "_vllm_structured_text_output_contract"
)
_PRESERVED_STRUCTURED_OUTPUT_CHOICES_ATTR = "_vllm_structured_output_choices"


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _has_non_structural_output_constraint(structured_outputs: Any) -> bool:
    return structured_outputs is not None and any(
        _get_field(structured_outputs, field) is not None
        for field in ("json", "regex", "choice", "grammar", "json_object")
    )


def _structured_output_choices(request: Any) -> tuple[str, ...]:
    preserved = getattr(request, _PRESERVED_STRUCTURED_OUTPUT_CHOICES_ATTR, None)
    if preserved is not None:
        return preserved

    structured_outputs = getattr(request, "structured_outputs", None)
    choices = _get_field(structured_outputs, "choice")
    if choices is None:
        return ()
    if isinstance(choices, str):
        return (choices,)
    try:
        return tuple(choice for choice in choices if isinstance(choice, str))
    except TypeError:
        return ()


def request_has_structured_text_output_contract(request: Any) -> bool:
    """Return True when the request requires structured text content."""
    if getattr(request, _PRESERVED_STRUCTURED_TEXT_OUTPUT_CONTRACT_ATTR, False):
        return True

    response_format = getattr(request, "response_format", None)
    if response_format is not None and _get_field(response_format, "type") not in (
        "text",
        "structural_tag",
    ):
        return True

    if _has_non_structural_output_constraint(
        getattr(request, "structured_outputs", None)
    ):
        return True

    text_config = getattr(request, "text", None)
    text_format = _get_field(text_config, "format")
    return text_format is not None and _get_field(text_format, "type") not in (
        "text",
        "structural_tag",
    )


def request_has_generation_structured_text_contract(request: Any) -> bool:
    """Return True when generation should constrain output before reasoning.

    Tool parsers may rewrite a forced-tool request into structured_outputs, but
    that grammar should still wait until reasoning ends when reasoning is
    enabled. Use the preserved pre-adjustment structured-text flag to avoid
    mistaking tool-parser grammar for caller-requested JSON/schema text.
    """
    if getattr(request, _PRESERVED_STRUCTURED_TEXT_OUTPUT_CONTRACT_ATTR, False):
        return True
    if getattr(request, _PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR, False):
        return False
    return request_has_structured_text_output_contract(request)


def request_has_machine_output_contract(request: Any) -> bool:
    """Return True when the request requires content/tool output.

    These request shapes are incompatible with treating untagged model output
    as implicit reasoning: the caller requested JSON/structured text or a
    forced tool payload.
    """
    if request_has_structured_text_output_contract(request):
        return True

    return request_requires_tool_output(request)


def preserve_request_machine_output_contract(request: Any) -> None:
    """Remember pre-adjustment machine-output semantics on mutable requests."""
    choices = _structured_output_choices(request)
    if choices:
        setattr(request, _PRESERVED_STRUCTURED_OUTPUT_CHOICES_ATTR, choices)
    has_structured_text = request_has_structured_text_output_contract(request)
    if has_structured_text:
        setattr(request, _PRESERVED_STRUCTURED_TEXT_OUTPUT_CONTRACT_ATTR, True)
    if has_structured_text or request_requires_tool_output(request):
        setattr(request, _PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR, True)


def request_requires_tool_output(request: Any) -> bool:
    """Return True when the request requires a tool-call payload."""
    tool_choice = getattr(request, "tool_choice", None)
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "function"
    return tool_choice == "required" or isinstance(
        tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam)
    )


def output_starts_with_reasoning_boundary(
    text: str,
    reasoning_parser: Any,
    *,
    allow_prefix: bool = False,
) -> bool:
    """Return True when output begins with a visible reasoning boundary."""
    stripped = text.lstrip()
    if not stripped:
        return False

    for boundary in (
        getattr(reasoning_parser, "reasoning_start_str", None),
        getattr(reasoning_parser, "reasoning_end_str", None),
    ):
        if boundary and (
            stripped.startswith(boundary)
            or (allow_prefix and boundary.startswith(stripped))
        ):
            return True
    return False


def output_starts_with_complete_reasoning_boundary(
    text: str,
    reasoning_parser: Any,
) -> bool:
    """Return True when output starts with a complete reasoning region."""
    stripped = text.lstrip()
    start = getattr(reasoning_parser, "reasoning_start_str", None)
    if not start or not stripped.startswith(start):
        return False

    remainder = stripped[len(start) :]
    end = getattr(reasoning_parser, "reasoning_end_str", None)
    if end and end in remainder:
        return True

    implicit_ends = getattr(reasoning_parser, "reasoning_implicit_end_strs", ())
    return any(boundary and boundary in remainder for boundary in implicit_ends)


def _output_matches_reasoning_prefixed_choice(
    text: str,
    request: Any,
    reasoning_parser: Any,
) -> bool:
    """Return True when output is a prefix/match of a literal choice.

    Structured choices may legitimately contain strings such as
    ``"<think>literal"``. Only this constrained case can safely disambiguate a
    complete visible reasoning delimiter from literal structured content.
    """
    stripped = text.lstrip()
    if not stripped:
        return False

    for choice in _structured_output_choices(request):
        normalized = choice.lstrip()
        if not output_starts_with_reasoning_boundary(normalized, reasoning_parser):
            continue
        if normalized.startswith(stripped) or stripped.startswith(normalized):
            return True
    return False


def output_starts_with_machine_output_contract(
    text: str,
    request: Any,
    reasoning_parser: Any,
    *,
    allow_prefix: bool = False,
) -> bool:
    """Return True when output should leave implicit reasoning immediately."""
    if not request_has_machine_output_contract(request):
        return False
    if output_starts_with_reasoning_boundary(
        text, reasoning_parser, allow_prefix=allow_prefix
    ):
        if not allow_prefix and output_is_exact_reasoning_boundary(
            text, reasoning_parser
        ):
            return False
        if not allow_prefix and output_starts_with_complete_reasoning_boundary(
            text, reasoning_parser
        ):
            return False
        if _output_matches_reasoning_prefixed_choice(text, request, reasoning_parser):
            return not allow_prefix
        return False

    stripped = text.lstrip()
    if not stripped:
        return False

    if request_requires_tool_output(request) and not (
        request_has_generation_structured_text_contract(request)
    ):
        tool_payload_prefixes = (
            "[",
            "{",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
        )
        return any(
            stripped.startswith(prefix)
            or (allow_prefix and prefix.startswith(stripped))
            for prefix in tool_payload_prefixes
        )

    return True


def output_is_exact_reasoning_boundary(text: str, reasoning_parser: Any) -> bool:
    """Return True when output is exactly one visible reasoning delimiter."""
    stripped = text.strip()
    if not stripped:
        return False

    for boundary in (
        getattr(reasoning_parser, "reasoning_start_str", None),
        getattr(reasoning_parser, "reasoning_end_str", None),
    ):
        if boundary and stripped == boundary:
            return True
    return False


def extract_reasoning_with_machine_output_contract(
    *,
    model_output: str,
    request: Any,
    reasoning_parser: Any,
) -> tuple[str | None, str | None]:
    """Extract reasoning while preserving untagged machine-output content."""
    if request_has_machine_output_contract(request):
        if output_is_exact_reasoning_boundary(model_output, reasoning_parser):
            return None, model_output
        if output_starts_with_machine_output_contract(
            model_output, request, reasoning_parser
        ):
            return None, model_output
    return reasoning_parser.extract_reasoning(model_output, request)
