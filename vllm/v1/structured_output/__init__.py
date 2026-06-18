# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())

                # Speculative decoding can place the reasoning-end marker
                # (e.g. </think>) and the first answer tokens inside a SINGLE
                # draft block. apply_bitmask is evaluated once per step, so the
                # answer positions after the marker would inherit
                # apply_bitmask=False and be sampled with a full (unconstrained)
                # mask. Find where the answer begins among the scheduled draft
                # tokens and switch the grammar on for those positions, within
                # this same block. grammar_advance_tokens() keeps the FSM in sync
                # across the boundary on the committed step; this keeps the
                # boundary block's own answer positions masked.
                #
                # Boundary is located with extract_content_ids() -- a pure
                # function of the token sequence -- rather than by walking
                # is_reasoning_end_streaming() over the draft tokens: drafts are
                # speculative and may be rejected, and some parsers (e.g.
                # Step3p5) mutate state inside the streaming check, which a
                # rollback would not undo. The hypothetical sequence includes
                # every draft token, so a marker completed by the current draft
                # token is visible (parsers that scan input_ids, e.g. GPT-OSS).
                reasoner = self._get_reasoner(request)
                content_start = None  # index in req_tokens where the answer begins
                if (
                    self.vllm_config.speculative_config is not None
                    and reasoner is not None
                    and not self.enable_in_reasoning
                    and not apply_bitmask
                    and structured_output_request.structured_output_key[0]
                    != StructuredOutputOptions.STRUCTURAL_TAG
                    and req_tokens
                ):
                    # Locate the answer start within this draft block robustly.
                    # Per-draft is_reasoning_end_streaming on the running sequence
                    # handles every alignment -- marker mid-block, marker as the
                    # last draft (content_start == len(req_tokens) -> bonus row),
                    # and a prior </think> in history / multiple think blocks
                    # (multi-turn) -- which the global last-marker length math of
                    # extract_content_ids does NOT (it skipped masking on those,
                    # leaving the answer unconstrained -> grammar-reject 500s).
                    # Snapshot the reasoner: is_reasoning_end_streaming mutates
                    # some parsers (e.g. Step3p5) and these drafts are speculative
                    # (may be rejected); restore so the real reasoner is untouched.
                    saved_state = dict(reasoner.__dict__)
                    try:
                        # Pass the full running sequence: is_reasoning_end_streaming
                        # may inspect the whole output, not just the delta -- the
                        # abstract-base default does (is_reasoning_end(input_ids):
                        # cohere_command/granite/hunyuan_a13b/olmo3), and a composed
                        # parser (deepseek_v3) can delegate to it -- so a bounded
                        # tail is NOT universally correct. Build once and append the
                        # drafts incrementally: O(n) per step, not the O(K*n) of
                        # rebuilding the prefix per draft. Runs only during the
                        # reasoning phase of structured spec-decode requests.
                        seq = list(request.all_token_ids)
                        for j, marker_tok in enumerate(req_tokens):
                            seq.append(marker_tok)
                            if reasoner.is_reasoning_end_streaming(seq, [marker_tok]):
                                content_start = j + 1  # answer starts after marker
                                break
                    finally:
                        reasoner.__dict__.clear()
                        reasoner.__dict__.update(saved_state)

                state_advancements = 0
                content_advance = True
                for pos, token in enumerate(itertools.chain(req_tokens, (-1,))):
                    in_content = content_start is not None and pos >= content_start
                    pos_apply = apply_bitmask or in_content
                    self._fill_bitmasks(((grammar, cumulative_index, pos_apply),))
                    cumulative_index += 1
                    if token == -1 or grammar.is_terminated():
                        continue
                    if in_content:
                        # Post-boundary drafts predate grammar activation, so
                        # peek with validate_tokens (no FSM advance, no error
                        # log) and only advance over grammar-valid drafts; once a
                        # stale draft is hit, keep masking from the current state
                        # but stop advancing.
                        if content_advance and grammar.validate_tokens([token]) == [
                            token
                        ]:
                            grammar.accept_tokens(req_id, [token])
                            state_advancements += 1
                        else:
                            content_advance = False
                    elif apply_bitmask:
                        # Steady state: scheduled drafts were already
                        # grammar-filtered, so a rejection is a real bug.
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (
                            token,
                            req_id,
                            scheduled_spec_decode_tokens,
                        )
                        state_advancements += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True

            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                return True

        return False

    def grammar_advance_tokens(
        self, request: "Request", new_token_ids: list[int]
    ) -> list[int]:
        """Tokens the grammar FSM should consume for ``new_token_ids``.

        Returns the exact slice of ``new_token_ids`` to feed to
        ``grammar.accept_tokens`` so the matcher stays in sync when reasoning
        ends mid-step.

        ``should_advance`` returns a bool and defers the *entire* step on the
        token that closes reasoning. That is safe without speculative decoding
        (one token per step), but a speculative step can commit ``</think>``
        together with the first answer tokens; deferring drops those answer
        tokens, so on the next step the matcher is behind the real output and
        never engages -> the rest of the answer is generated unconstrained.

        #42452 fixed this for structural tags only (their grammar models the
        pre-trigger free text, so the whole delta advances the matcher). This
        completes it for JSON/regex/choice/grammar by advancing the matcher
        over just the post-``</think>`` answer tokens on the boundary step.
        Boundary detection uses this step's committed tokens directly, not the
        ``num_computed_tokens`` window the deferred path relied on.
        """
        if not request.use_structured_output:
            return []
        reasoner = self._get_reasoner(request)
        if reasoner is None or self.enable_in_reasoning:
            return new_token_ids
        structured_req = request.structured_output_request
        assert structured_req is not None
        if structured_req.reasoning_ended:
            return new_token_ids
        # Did reasoning close within this step's committed tokens? Use the
        # streaming contract -- full sequence + this step's delta -- not
        # is_reasoning_end(new_token_ids): some parsers (e.g. Mistral) only
        # report end when the start marker is also present, which it is not
        # once the opening tag scrolled out of this step. all_token_ids already
        # includes new_token_ids here (appended in _update_request_with_output),
        # so this matches what should_advance does in the non-spec path. Safe to
        # call on a stateful parser: new_token_ids are committed, not draft.
        if not reasoner.is_reasoning_end_streaming(
            request.all_token_ids, new_token_ids
        ):
            return []
        structured_req.reasoning_ended = True
        if (
            structured_req.structured_output_key[0]
            == StructuredOutputOptions.STRUCTURAL_TAG
        ):
            # The structural-tag grammar consumes the pre-trigger free text, so
            # the whole delta (including the trigger) advances it correctly.
            return new_token_ids
        # The reasoning text is not part of a JSON/regex/choice/grammar schema;
        # advance only over the answer tokens emitted after the end marker.
        # extract_content_ids(all_token_ids) is the answer-so-far for parsers
        # that return post-marker content only (Kimi / Qwen / DeepSeek-R1 /
        # base); at this boundary step that answer is exactly the tail of
        # new_token_ids. Some parsers also return non-answer tokens (e.g. Mistral
        # returns pre-<think> content + post-</think> content), so guard that the
        # extracted content really is this step's suffix; otherwise defer (the
        # non-spec behaviour -- matcher lags one step) rather than feed reasoning
        # tokens to the grammar and trip a spurious rejection.
        content = reasoner.extract_content_ids(list(request.all_token_ids))
        if not (
            content
            and len(content) <= len(new_token_ids)
            and list(content) == list(new_token_ids[-len(content) :])
        ):
            return []
        suffix = list(content)
        # Belt-and-suspenders: these post-</think> tokens were sampled before the
        # bitmask could constrain them. If boundary masking missed this alignment
        # the tokens may be invalid; feeding them to accept_tokens would reject
        # and FINISHED_ERROR (HTTP 500). Peek with validate_tokens (no FSM
        # advance) and only advance if they are grammar-valid; otherwise defer
        # (matcher lags one step -- pre-patch behaviour, no 500).
        grammar = structured_req.grammar
        if grammar is not None and grammar.validate_tokens(suffix) != suffix:
            return []
        return suffix

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
