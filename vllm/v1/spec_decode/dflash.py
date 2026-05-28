# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None
        self._k26_branch_trace_dir = os.environ.get("VLLM_K26_DFLASH_BRANCH_TRACE_DIR")
        self._k26_branch_trace_limit = int(
            os.environ.get("VLLM_K26_DFLASH_BRANCH_TRACE_LIMIT", "16")
        )
        self._k26_branch_k = int(os.environ.get("VLLM_K26_DFLASH_BRANCH_K", "4"))
        self._k26_branch_trace_k = int(
            os.environ.get("VLLM_K26_DFLASH_BRANCH_TRACE_K", "16")
        )
        self._k26_branch_positions = os.environ.get(
            "VLLM_K26_DFLASH_BRANCH_POSITIONS", "0"
        )
        self._k26_branch_trace_count = 0

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        cache_config = base.cache_config
        if cache_config is not None and is_quantized_kv_cache(cache_config.cache_dtype):
            cache_config = replace(
                cache_config,
                cache_dtype="auto",
                calculate_kv_scales=False,
            )
        return replace(
            base,
            cache_config=cache_config,
            attention_config=replace(
                base.attention_config,
                use_non_causal=True,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        pass

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        with record_function_or_nullcontext("dflash: copy_expand_inputs_kernel"):
            copy_and_expand_dflash_inputs_kernel[grid](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
                out_query_slot_mapping_ptr=self._slot_mapping_buffer,
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=cad.block_table_tensor,
                block_table_stride=cad.block_table_tensor.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                num_rejected_tokens_ptr=(
                    num_rejected_tokens_gpu if has_num_rejected else 0
                ),
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=self.block_size,
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_NUM_REJECTED=has_num_rejected,
            )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # Skip num_rejected_tokens (GPU-only); overestimating is fine here.
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        with record_function_or_nullcontext("dflash: build_common_attn_metadata"):
            new_cad = CommonAttentionMetadata(
                query_start_loc=new_query_start_loc,
                seq_lens=effective_seq_lens + num_query_per_req,
                query_start_loc_cpu=(
                    torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                    * num_query_per_req
                ),
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
                num_reqs=cad.num_reqs,
                num_actual_tokens=num_query_total,
                max_query_len=num_query_per_req,
                max_seq_len=cad.max_seq_len + num_query_per_req,
                block_table_tensor=cad.block_table_tensor,
                slot_mapping=query_slot_mapping,
                causal=False,  # Non-causal attention is required for DFlash
            )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
            trace_label="dflash_dummy_forward",
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        # Pre-insert context KVs directly into cache.
        with record_function_or_nullcontext("dflash: precompute_context_kv"):
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states,
                self._context_positions_buffer[:num_context],
                self._context_slot_mapping_buffer[:num_context],
            )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        for layer_name, attn_metadata in per_layer.items():
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def _dflash_trace_topk_payload(
        self, logits: torch.Tensor, k: int
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
        logits_fp32 = logits.float()
        k = min(max(1, k), logits_fp32.shape[-1])
        topk_values, topk_ids = torch.topk(logits_fp32, k=k, dim=-1)
        topk_logprobs = topk_values - torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
        return (
            topk_ids.detach().cpu().tolist(),
            topk_values.detach().cpu().tolist(),
            topk_logprobs.detach().cpu().tolist(),
        )

    def _run_dflash_trace_forward(
        self,
        model_kwargs: dict[str, Any],
        per_layer_attn_metadata: dict[str, object],
        common_attn_metadata: CommonAttentionMetadata,
        num_input_tokens: int,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        slot_mapping_size: int,
    ) -> torch.Tensor:
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
            trace_label="dflash_branch_trace",
        ):
            return self.model(**model_kwargs)

    def _k26_dflash_branch_positions_for_trace(self) -> list[int]:
        positions_env = self._k26_branch_positions.strip().lower()
        if positions_env in ("all", "*"):
            return list(range(self.num_speculative_tokens))

        positions: list[int] = []
        for part in positions_env.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                position = int(part)
            except ValueError:
                logger.warning_once(
                    "Ignoring invalid VLLM_K26_DFLASH_BRANCH_POSITIONS=%r",
                    self._k26_branch_positions,
                )
                continue
            if 0 <= position < self.num_speculative_tokens:
                positions.append(position)
        return sorted(set(positions)) or [0]

    @override
    def _maybe_trace_dflash_branch_prefix(
        self,
        sample_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        model_kwargs: dict[str, Any],
        per_layer_attn_metadata: dict[str, object],
        common_attn_metadata: CommonAttentionMetadata,
        token_indices_to_sample: torch.Tensor,
        num_input_tokens: int,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        slot_mapping_size: int,
    ) -> None:
        trace_dir = self._k26_branch_trace_dir
        if (
            not trace_dir
            or self._k26_branch_trace_count >= self._k26_branch_trace_limit
            or self.num_speculative_tokens <= 1
        ):
            return

        input_ids = model_kwargs.get("input_ids")
        if input_ids is None:
            return

        batch_size = common_attn_metadata.num_reqs
        num_query_per_req = 1 + self.num_speculative_tokens
        num_actual_query_tokens = batch_size * num_query_per_req
        branch_positions = self._k26_dflash_branch_positions_for_trace()
        trace_k = self._k26_branch_trace_k
        branch_k = min(max(1, self._k26_branch_k), trace_k)

        original_query_input_ids = self.input_ids[:num_actual_query_tokens].clone()
        flat_draft_token_ids = draft_token_ids.reshape(-1)
        original_logits = self.model.compute_logits(sample_hidden_states)
        (
            original_topk_ids,
            original_topk_logits,
            original_topk_logprobs,
        ) = self._dflash_trace_topk_payload(original_logits, max(trace_k, branch_k))

        branches: list[dict[str, Any]] = []
        trace_idx = self._k26_branch_trace_count
        try:
            for req_idx in range(batch_size):
                for branch_position in branch_positions:
                    branch_row = req_idx * self.num_speculative_tokens + branch_position
                    candidate_ids = original_topk_ids[branch_row][:branch_k]
                    for candidate_rank, candidate_token_id in enumerate(candidate_ids):
                        self.input_ids[:num_actual_query_tokens].copy_(
                            original_query_input_ids
                        )
                        for prefix_position in range(branch_position):
                            prefix_row = (
                                req_idx * self.num_speculative_tokens + prefix_position
                            )
                            prefix_query_offset = (
                                req_idx * num_query_per_req + 1 + prefix_position
                            )
                            self.input_ids[prefix_query_offset] = int(
                                flat_draft_token_ids[prefix_row].item()
                            )

                        branch_query_offset = (
                            req_idx * num_query_per_req + 1 + branch_position
                        )
                        self.input_ids[branch_query_offset] = int(candidate_token_id)

                        branch_hidden_states = self._run_dflash_trace_forward(
                            model_kwargs,
                            per_layer_attn_metadata,
                            common_attn_metadata,
                            num_input_tokens,
                            num_tokens_across_dp,
                            cudagraph_runtime_mode,
                            slot_mapping_size,
                        )
                        branch_sample_hidden_states = branch_hidden_states[
                            token_indices_to_sample
                        ]
                        branch_logits = self.model.compute_logits(
                            branch_sample_hidden_states
                        )
                        (
                            branch_topk_ids,
                            branch_topk_logits,
                            branch_topk_logprobs,
                        ) = self._dflash_trace_topk_payload(branch_logits, trace_k)
                        branches.append(
                            {
                                "request_index": req_idx,
                                "branch_position": branch_position,
                                "candidate_rank": candidate_rank,
                                "candidate_token_id": int(candidate_token_id),
                                "topk_ids": branch_topk_ids,
                                "topk_logits": branch_topk_logits,
                                "topk_logprobs": branch_topk_logprobs,
                            }
                        )
        except Exception:
            logger.exception("K26 DFlash branch-prefix trace failed")
            return
        finally:
            self.input_ids[:num_actual_query_tokens].copy_(original_query_input_ids)
            try:
                self._run_dflash_trace_forward(
                    model_kwargs,
                    per_layer_attn_metadata,
                    common_attn_metadata,
                    num_input_tokens,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode,
                    slot_mapping_size,
                )
            except Exception:
                logger.exception("K26 DFlash branch-prefix trace restore failed")

        os.makedirs(trace_dir, exist_ok=True)
        payload = {
            "time_ns": time.time_ns(),
            "trace_index": trace_idx,
            "dp_rank": self.dp_rank,
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "all_greedy": bool(sampling_metadata.all_greedy),
            "branch_positions": branch_positions,
            "branch_k": branch_k,
            "trace_k": trace_k,
            "num_reqs": int(batch_size),
            "num_query_per_req": int(num_query_per_req),
            "token_indices_to_sample": token_indices_to_sample.detach().cpu().tolist(),
            "query_input_ids": original_query_input_ids.detach().cpu().tolist(),
            "draft_token_ids": draft_token_ids.detach().cpu().tolist(),
            "original_topk_ids": original_topk_ids,
            "original_topk_logits": original_topk_logits,
            "original_topk_logprobs": original_topk_logprobs,
            "branches": branches,
        }
        filename = (
            f"dflash_branch_dp{self.dp_rank}_{trace_idx:04d}_{payload['time_ns']}.json"
        )
        with open(os.path.join(trace_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self._k26_branch_trace_count += 1
