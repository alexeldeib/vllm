# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)

_PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR = "_vllm_machine_output_contract"


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _has_non_structural_output_constraint(structured_outputs: Any) -> bool:
    return structured_outputs is not None and any(
        _get_field(structured_outputs, field) is not None
        for field in ("json", "regex", "choice", "grammar", "json_object")
    )


def request_has_machine_output_contract(request: Any) -> bool:
    """Return True when the request contract requires content/tool output.

    These request shapes are incompatible with treating untagged model output as
    implicit reasoning: the caller requested JSON/structured text or a forced
    tool payload.
    """
    if getattr(request, _PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR, False):
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
    if text_format is not None and _get_field(text_format, "type") not in (
        "text",
        "structural_tag",
    ):
        return True

    return request_requires_tool_output(request)


def preserve_request_machine_output_contract(request: Any) -> None:
    """Remember pre-adjustment machine-output semantics on mutable requests."""
    if request_has_machine_output_contract(request):
        setattr(request, _PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR, True)


def request_requires_tool_output(request: Any) -> bool:
    """Return True when the request requires a tool-call payload."""
    tool_choice = getattr(request, "tool_choice", None)
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "function"
    return tool_choice == "required" or isinstance(
        tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam)
    )


def request_allows_auto_tool_output(request: Any) -> bool:
    """Return True when an automatic tool parser may emit tool calls."""
    if getattr(request, _PRESERVED_MACHINE_OUTPUT_CONTRACT_ATTR, False):
        return False

    tool_choice = getattr(request, "tool_choice", None)
    tools = getattr(request, "tools", None)
    if not tools or (tool_choice is not None and tool_choice != "auto"):
        return False

    structured_outputs = getattr(request, "structured_outputs", None)
    if _has_non_structural_output_constraint(structured_outputs):
        return False

    response_format = getattr(request, "response_format", None)
    if response_format is not None and _get_field(response_format, "type") not in (
        "text",
        "structural_tag",
    ):
        return False

    text_config = getattr(request, "text", None)
    text_format = _get_field(text_config, "format")
    return text_format is None or _get_field(text_format, "type") in (
        "text",
        "structural_tag",
    )


def output_starts_with_reasoning_boundary(
    text: str,
    reasoning_parser: Any,
    *,
    allow_prefix: bool = False,
) -> bool:
    """Return True when output begins with a visible reasoning boundary.

    Machine-output requests still need reasoning parsing when the model emits an
    explicit reasoning delimiter. For untagged output, the API contract wins and
    the parser should start in content/tool mode. Streaming callers may set
    ``allow_prefix`` to hold back partial delimiter prefixes until the next
    delta disambiguates them.
    """
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
        if not output_starts_with_reasoning_boundary(model_output, reasoning_parser):
            return None, model_output
    return reasoning_parser.extract_reasoning(model_output, request)
