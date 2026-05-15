# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.request_utils import (
    output_is_exact_reasoning_boundary,
    output_starts_with_reasoning_boundary,
    request_has_machine_output_contract,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class KimiK2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi K2 model.

    The Kimi K2 model uses <think>...</think> tokens to denote reasoning text,
    and may implicitly end reasoning by starting a tool call section using
    <|tool_calls_section_begin|>.
    Thinking may also begin without a </think> token.

    Kimi's thinking mode can be disabled via chat_template_kwargs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # Check if thinking is disabled via chat_template_kwargs
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", True))

        # If thinking is not enabled, use identity parser to fall through
        self._identity_parser: IdentityReasoningParser | None
        if not thinking:
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._identity_parser = None

        # Token definitions
        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        # Get token IDs
        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_id = self.vocab.get(
            self._tool_section_start_token
        )

        if self._start_token_id is None or self._end_token_id is None:
            raise RuntimeError(
                "KimiK2ReasoningParser could not locate think start/end "
                "tokens in the tokenizer!"
            )

    @property
    def reasoning_start_str(self) -> str | None:
        return self._start_token

    @property
    def reasoning_end_str(self) -> str | None:
        return self._end_token

    @property
    def reasoning_end_strs(self) -> tuple[str, ...]:
        return (self._end_token, self._tool_section_start_token)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Reasoning ends when we see either:
        1. The end token (</think>)
        2. The tool section start token (<|tool_calls_section_begin|>)
        """
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)

        start_token_id = self._start_token_id
        end_token_id = self._end_token_id
        tool_section_start_token_id = self._tool_section_start_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
            # Implicit reasoning end via tool call section
            if (
                tool_section_start_token_id is not None
                and input_ids[i] == tool_section_start_token_id
            ):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a decode step.
        """
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        # Materialize iterable for membership checks
        delta_ids_set = set(delta_ids)

        # Check for explicit end token or implicit tool section start in delta
        if self._end_token_id in delta_ids_set:
            return True
        return (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_ids_set
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        if self._end_token_id in input_ids:
            end_token_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._end_token_id)
            )

            if end_token_index != -1:
                return input_ids[end_token_index + 1 :]

        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in input_ids
        ):
            tool_section_index = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self._tool_section_start_token_id)
            )

            if tool_section_index != -1:
                return input_ids[tool_section_index:]

        # still reasoning (no content)
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning(model_output, request)

        if request_has_machine_output_contract(request):
            if output_is_exact_reasoning_boundary(model_output, self):
                return None, model_output
            if not output_starts_with_reasoning_boundary(model_output, self):
                return None, model_output

        # thinking does not require a think start token but consume it if present
        start_token_index = self._strip_start_token_for_full_output(model_output)
        end_token_index = model_output.find(self._end_token)

        if end_token_index != -1:
            return (
                model_output[start_token_index:end_token_index],
                model_output[end_token_index + len(self._end_token) :] or None,
            )

        tool_section_index = model_output.find(self._tool_section_start_token)
        if tool_section_index != -1:
            return (
                model_output[start_token_index:tool_section_index],
                model_output[tool_section_index:] or None,
            )

        # still reasoning (no content)
        return (
            model_output[start_token_index:],
            None,
        )

    def _strip_start_token_for_full_output(self, model_output: str) -> int:
        start_token_index = model_output.find(self._start_token)
        if start_token_index == -1:
            return 0
        if model_output[:start_token_index].strip():
            return 0
        return start_token_index + len(self._start_token)

    def _strip_start_token_from_stream_delta(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> str | None:
        previous_stripped = previous_text.lstrip()
        current_stripped = current_text.lstrip()

        if not current_stripped:
            return delta_text

        if self._start_token.startswith(current_stripped):
            return None

        if not current_stripped.startswith(self._start_token):
            if previous_stripped and self._start_token.startswith(previous_stripped):
                return previous_stripped + delta_text
            return delta_text

        current_reasoning = current_stripped[len(self._start_token) :]
        if previous_stripped.startswith(self._start_token):
            previous_reasoning = previous_stripped[len(self._start_token) :]
            if current_reasoning.startswith(previous_reasoning):
                return current_reasoning[len(previous_reasoning) :]

        if self._start_token.startswith(previous_stripped) or not previous_stripped:
            return current_reasoning

        return delta_text

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
        Extract reasoning content from a delta message during streaming.
        """
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # If reasoning has already ended in previous tokens, this is content
        if self.is_reasoning_end(previous_token_ids):
            return DeltaMessage(content=delta_text)

        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] in [
            self._start_token_id,
            self._end_token_id,
        ]:
            return None

        delta_text = self._strip_start_token_from_stream_delta(
            previous_text, current_text, delta_text
        )
        if delta_text is None:
            return None

        if self._end_token in delta_text:
            end_index = delta_text.find(self._end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning, content=content if content else None
            )

        if self._tool_section_start_token in delta_text:
            tool_index = delta_text.find(self._tool_section_start_token)
            reasoning = delta_text[:tool_index]
            content = delta_text[tool_index:]
            return DeltaMessage(reasoning=reasoning, content=content)

        if self._end_token_id in delta_token_ids:
            if self._end_token not in delta_text:
                # Token ID arrived before text was flushed (stop-sequence buffering).
                # Wait for the next delta when the text becomes visible.
                return None
            end_index = delta_text.find(self._end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning, content=content if content else None
            )

        if self._tool_section_start_token_id in delta_token_ids:
            if self._tool_section_start_token not in delta_text:
                # Token ID arrived before text was flushed (stop-sequence buffering).
                return None
            tool_index = delta_text.find(self._tool_section_start_token)
            reasoning = delta_text[:tool_index]
            content = delta_text[tool_index:]
            return DeltaMessage(reasoning=reasoning, content=content)

        # still reasoning (no end token)
        return DeltaMessage(reasoning=delta_text)
