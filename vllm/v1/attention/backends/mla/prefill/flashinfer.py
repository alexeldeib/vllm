# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer backend for MLA prefill."""

import functools
from inspect import signature
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.utils import (
    PerLayerParameters,
    get_per_layer_parameters,
    infer_global_hyperparameters,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = object  # type: ignore[misc,assignment]

_DEFAULT_NUM_CHUNKS = 32


@functools.cache
def _flashinfer_ragged_run_accepts_scales() -> bool:
    run = getattr(BatchPrefillWithRaggedKVCacheWrapper, "run", None)
    if run is None:
        return False

    try:
        parameters = signature(run).parameters
    except (TypeError, ValueError):
        return False

    # Treat only explicit q/k/v scale parameters as support. Some wrappers
    # accept **kwargs but older kernels still ignore unknown descale arguments,
    # which would silently run scaled FP8 inputs in the encoded domain.
    return all(name in parameters for name in ("q_scale", "k_scale", "v_scale"))


class FlashInferPrefillBackend(MLAPrefillBackend):
    """FlashInfer backend for MLA prefill."""

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer import (
                BatchPrefillWithRaggedKVCacheWrapper,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    @classmethod
    def supports_prefill_query_quantization(cls) -> bool:
        return _flashinfer_ragged_run_accepts_scales()

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

        self._prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = None
        self._prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = []
        self._global_hyperparameters: PerLayerParameters | None = None
        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
        )

    def _ensure_chunks(
        self,
        num_chunks: int,
        workspace_buffer: torch.Tensor,
    ) -> None:
        if len(self._prefill_chunks) < num_chunks:
            for _ in range(len(self._prefill_chunks), num_chunks):
                self._prefill_chunks.append(
                    BatchPrefillWithRaggedKVCacheWrapper(
                        workspace_buffer, "NHD", backend="cutlass"
                    )
                )

    def _resolve_global_hyperparameters(self) -> PerLayerParameters:
        if self._global_hyperparameters is not None:
            return self._global_hyperparameters

        from vllm.model_executor.layers.attention.mla_attention import (
            MLAAttention,
            MLACommonImpl,
        )

        forward_context = self.vllm_config.compilation_config.static_forward_context
        layer_names = [
            name
            for name, layer in forward_context.items()
            if isinstance(layer, MLAAttention)
        ]

        self._global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(
                self.vllm_config,
                layer_names,
                MLACommonImpl,  # type: ignore[type-abstract]
            )
        )
        return self._global_hyperparameters

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        global_hyperparameters = self._resolve_global_hyperparameters()
        qo_indptr = prefill_metadata.query_start_loc
        has_context = prefill_metadata.chunked_context is not None
        if self._prefill_main is None:
            self._prefill_main = BatchPrefillWithRaggedKVCacheWrapper(
                self._workspace_buffer, "NHD", backend="cutlass"
            )
            self._ensure_chunks(_DEFAULT_NUM_CHUNKS, self._workspace_buffer)

        if has_context:
            chunked_context = prefill_metadata.chunked_context
            assert chunked_context is not None
            num_chunks = chunked_context.cu_seq_lens.shape[0]
            self._ensure_chunks(num_chunks, self._workspace_buffer)

        num_qo_heads = self.num_heads
        num_kv_heads = num_qo_heads

        head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        head_dim_vo = self.v_head_dim
        kv_indptr = qo_indptr.clone()

        assert self._prefill_main is not None
        self._prefill_main.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=True,
            sm_scale=global_hyperparameters.sm_scale,
            window_left=global_hyperparameters.window_left,
            logits_soft_cap=global_hyperparameters.logits_soft_cap,
            q_data_type=prefill_metadata.q_data_type,
            o_data_type=prefill_metadata.output_dtype,
        )

        if has_context:
            chunked_context = prefill_metadata.chunked_context
            assert chunked_context is not None
            for i in range(num_chunks):
                kv_indptr_chunk = chunked_context.cu_seq_lens[i]

                self._prefill_chunks[i].plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr_chunk,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    causal=False,
                    sm_scale=global_hyperparameters.sm_scale,
                    window_left=global_hyperparameters.window_left,
                    logits_soft_cap=global_hyperparameters.logits_soft_cap,
                    q_data_type=prefill_metadata.q_data_type,
                    o_data_type=prefill_metadata.output_dtype,
                )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_main is not None
        scale_kwargs = self._scale_kwargs(q_scale, k_scale, v_scale)

        ret = self._prefill_main.run(
            q=q,
            k=k,
            v=v,
            return_lse=return_softmax_lse,
            **scale_kwargs,
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
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_kwargs = self._scale_kwargs(q_scale, k_scale, v_scale)
        attn_out, lse = self._prefill_chunks[chunk_idx].run(
            q=q,
            k=k,
            v=v,
            return_lse=True,
            **scale_kwargs,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()

    @classmethod
    def _scale_kwargs(
        cls,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
    ) -> dict[str, float]:
        if q_scale is None and k_scale is None and v_scale is None:
            return {}
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError(
                "FlashInfer ragged MLA prefill requires q, k, and v scales together"
            )
        if not cls.supports_prefill_query_quantization():
            raise NotImplementedError(
                "FlashInfer ragged MLA prefill does not support scaled FP8 "
                "input descales with this flashinfer version"
            )
        return {"q_scale": q_scale, "k_scale": k_scale, "v_scale": v_scale}
