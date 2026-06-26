# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRT-LLM Ragged backend for MLA prefill."""

from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


class TrtllmRaggedPrefillBackend(MLAPrefillBackend):
    """TRT-LLM Ragged backend for MLA prefill."""

    supported_mla_dimensions: ClassVar[list[MLADimensions]] = [
        MLADimensions(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        MLADimensions(
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
        ),
    ]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_RAGGED"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.prefill import (
                trtllm_ragged_attention_deepseek,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
            (
                (envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,),
                torch.uint8,
            ),
        )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        self._query_seq_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )
        self._query_seq_lens_cpu = prefill_metadata.query_lens_cpu

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        out = torch.empty(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=self._prefill_metadata.output_dtype,
        )

        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=self._query_seq_lens,
            max_q_len=self._prefill_metadata.max_query_len,
            max_kv_len=self._prefill_metadata.max_query_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=self._query_seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            cum_seq_lens_kv=self._prefill_metadata.query_start_loc,
            enable_pdl=False,
            is_causal=True,
            return_lse=return_softmax_lse,
            out=out,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        chunked_context = self._prefill_metadata.chunked_context
        assert chunked_context.seq_lens[chunk_idx] is not None

        out = torch.empty(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=self._prefill_metadata.output_dtype,
        )

        kv_seq_lens_cpu = chunked_context.seq_lens[chunk_idx]
        active_rows_cpu = kv_seq_lens_cpu > 0
        if active_rows_cpu.all().item():
            attn_out, lse = self._run_prefill_context_chunk_kernel(
                q=q,
                k=k,
                v=v,
                seq_lens=kv_seq_lens_cpu,
                max_kv_len=chunked_context.max_seq_lens[chunk_idx],
                batch_size=kv_seq_lens_cpu.shape[0],
                cum_seq_lens_q=self._prefill_metadata.query_start_loc,
                cum_seq_lens_kv=chunked_context.cu_seq_lens[chunk_idx],
                out=out,
            )
            return attn_out, lse.transpose(0, 1).contiguous()

        q_seq_lens_cpu = self._query_seq_lens_cpu
        assert q_seq_lens_cpu is not None
        active_count = int(active_rows_cpu.sum().item())
        q_token_mask_cpu = torch.repeat_interleave(
            active_rows_cpu,
            q_seq_lens_cpu,
            output_size=q.shape[0],
        )

        if active_count == 0:
            lse = torch.full(
                (q.shape[1], q.shape[0]),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )
            return out.zero_(), lse

        kv_token_mask_cpu = torch.repeat_interleave(
            active_rows_cpu,
            kv_seq_lens_cpu,
            output_size=k.shape[0],
        )
        q_token_mask = q_token_mask_cpu.to(device=q.device, non_blocking=True)
        kv_token_mask = kv_token_mask_cpu.to(device=k.device, non_blocking=True)
        compact_q = q[q_token_mask]
        compact_k = k[kv_token_mask]
        compact_v = v[kv_token_mask]

        compact_q_seq_lens_cpu = q_seq_lens_cpu[active_rows_cpu]
        compact_kv_seq_lens_cpu = kv_seq_lens_cpu[active_rows_cpu]
        compact_cu_seq_lens_q_cpu = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(compact_q_seq_lens_cpu, dim=0, dtype=torch.int32),
            ]
        )
        compact_cu_seq_lens_kv_cpu = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(compact_kv_seq_lens_cpu, dim=0, dtype=torch.int32),
            ]
        )

        compact_out = torch.empty(
            compact_q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=self._prefill_metadata.output_dtype,
        )

        compact_attn_out, compact_lse = self._run_prefill_context_chunk_kernel(
            q=compact_q,
            k=compact_k,
            v=compact_v,
            seq_lens=compact_kv_seq_lens_cpu,
            max_kv_len=int(compact_kv_seq_lens_cpu.max().item()),
            batch_size=active_count,
            cum_seq_lens_q=compact_cu_seq_lens_q_cpu.to(
                device=q.device,
                non_blocking=True,
            ),
            cum_seq_lens_kv=compact_cu_seq_lens_kv_cpu.to(
                device=k.device,
                non_blocking=True,
            ),
            out=compact_out,
        )

        out.zero_()
        out[q_token_mask] = compact_attn_out
        lse = torch.full(
            (q.shape[0], q.shape[1]),
            -torch.inf,
            device=q.device,
            dtype=compact_lse.dtype,
        )
        lse[q_token_mask] = compact_lse
        return out, lse.transpose(0, 1).contiguous()

    def _run_prefill_context_chunk_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        attn_out, lse = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=seq_lens,
            max_q_len=self._prefill_metadata.max_query_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=batch_size,
            window_left=-1,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            enable_pdl=False,
            is_causal=False,
            return_lse=True,
            out=out,
        )
        return attn_out, lse
