from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.kv_4bit.config import (
    KV_4BIT_DTYPES,
    KV4BitConfig,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_kv_4bit_decode import (
    triton_kv_4bit_dequant_kv,
    triton_kv_4bit_decode_attention,
)
from vllm.v1.attention.ops.triton_kv_4bit_store import triton_kv_4bit_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


class KV4BitAttentionBackend(AttentionBackend):
    """Attention backend using KV-4BIT packed KV-cache storage."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(KV_4BIT_DTYPES)

    @staticmethod
    def get_name() -> str:
        return "KV_4BIT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["KV4BitAttentionImpl"]:
        return KV4BitAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["KV4BitMetadataBuilder"]:
        return KV4BitMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "kv_4bit",
    ) -> tuple[int, ...]:
        cfg = KV4BitConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype in KV_4BIT_DTYPES

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # The selector passes the effective allocation head size for packed
        # caches, so only reject impossible values here.
        return head_size > 0

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability,
    ) -> str | None:
        if kv_cache_dtype in KV_4BIT_DTYPES:
            try:
                KV4BitConfig.from_cache_dtype(kv_cache_dtype, head_size)
            except ValueError as err:
                return str(err)
        return None


@dataclass
class KV4BitMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0


class KV4BitMetadataBuilder(AttentionMetadataBuilder[KV4BitMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> KV4BitMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )
        return KV4BitMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class KV4BitAttentionImpl(AttentionImpl[KV4BitMetadata]):
    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        if alibi_slopes is not None:
            raise NotImplementedError("KV-4BIT does not support ALiBi yet")
        if sliding_window is not None:
            raise NotImplementedError("KV-4BIT does not support sliding window yet")
        if logits_soft_cap is not None:
            raise NotImplementedError("KV-4BIT does not support logits soft cap yet")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("KV-4BIT only supports decoder attention")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV-4BIT does not support KV sharing yet")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_4bit_config = KV4BitConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.fa_version = get_flash_attn_version(head_size=head_size)

        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        triton_kv_4bit_store(
            k,
            v,
            kv_cache,
            slot_mapping,
            hadamard_order=self.kv_4bit_config.hadamard_order,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KV4BitMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            decode_output = output[:N]
            if decode_output.ndim == 2:
                decode_output = decode_output.view(N, self.num_heads, self.head_size)
            self._decode_attention(
                q,
                kv_cache,
                attn_metadata,
                layer,
                output=decode_output,
            )
            return output
        elif num_decodes == 0:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(q, k, v, kv_cache, attn_metadata, layer)
        else:
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=q.device, dtype=q.dtype
            )

            decode_meta = KV4BitMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            self._decode_attention(
                q[:num_decode_tokens],
                kv_cache,
                decode_meta,
                layer,
                output=attn_out[:num_decode_tokens],
            )

            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            prefill_max_seq = max(prefill_seq_lens.tolist())
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_meta = KV4BitMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                layer,
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KV4BitMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        qsl = attn_metadata.query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()
        num_reqs = len(qsl) - 1
        all_first_chunk = all(
            seq_lens_list[i] == qsl[i + 1] - qsl[i] for i in range(num_reqs)
        )

        if _HAS_FLASH_ATTN and all_first_chunk:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]

            if q_len == seq_len:
                if _HAS_FLASH_ATTN:
                    cu = torch.zeros(2, device=query.device, dtype=torch.int32)
                    cu[1] = q_len
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q_seq.transpose(0, 1).contiguous(),
                        k_seq.transpose(0, 1).contiguous(),
                        v_seq.transpose(0, 1).contiguous(),
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=(self.num_kv_heads < self.num_heads),
                    ).transpose(0, 1)
            else:
                cached_len = seq_len - q_len
                out = self._continuation_prefill(
                    query=q_seq,
                    key_chunk=k_seq,
                    value_chunk=v_seq,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i : i + 1],
                    cached_len=cached_len,
                    seq_len=seq_len,
                )

            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        query: torch.Tensor,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Run SGLang-style prefill over dense cached + current K/V.

        Cached K/V are stored packed in the KV-4BIT cache. For continuation prefill,
        dequantize prior cached tokens to dense tensors, append the current raw
        chunk K/V, then use FlashAttention/SDPA. The dequant kernel
        inverse-rotates cached K so query and current K stay in original space.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        dtype = query.dtype

        if cached_len <= 0:
            if _HAS_FLASH_ATTN:
                if not hasattr(self, "_cu_2"):
                    self._cu_2 = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2[1] = q_len
                return self._flash_attn_varlen(
                    q=query,
                    k=key_chunk,
                    v=value_chunk,
                    cu_seqlens_q=self._cu_2,
                    cu_seqlens_k=self._cu_2,
                    max_seqlen_q=q_len,
                    max_seqlen_k=q_len,
                )
            return F.scaled_dot_product_attention(
                query.transpose(0, 1).contiguous(),
                key_chunk.transpose(0, 1).contiguous(),
                value_chunk.transpose(0, 1).contiguous(),
                is_causal=True,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            ).transpose(0, 1)

        block_size = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / block_size) * block_size
        cache_shape = (1, Hk, alloc_len, D)
        if is_workspace_manager_initialized():
            k_cache_buf, v_cache_buf = current_workspace_manager().get_simultaneous(
                (cache_shape, dtype),
                (cache_shape, dtype),
            )
        else:
            k_cache_buf = torch.empty(cache_shape, dtype=dtype, device=device)
            v_cache_buf = torch.empty(cache_shape, dtype=dtype, device=device)

        triton_kv_4bit_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            k_out=k_cache_buf,
            v_out=v_cache_buf,
            hadamard_order=self.kv_4bit_config.hadamard_order,
        )

        k_full = torch.empty(seq_len, Hk, D, dtype=dtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=dtype, device=device)
        k_full[:cached_len] = k_cache_buf[0, :, :cached_len, :].transpose(0, 1)
        v_full[:cached_len] = v_cache_buf[0, :, :cached_len, :].transpose(0, 1)
        k_full[cached_len:] = key_chunk
        v_full[cached_len:] = value_chunk

        if _HAS_FLASH_ATTN:
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            self._cu_2_q[1] = q_len
            self._cu_2_k[1] = seq_len
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=self._cu_2_q,
                cu_seqlens_k=self._cu_2_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )

        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = k_pos <= q_pos
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=causal_mask,
            scale=self.scale,
            enable_gqa=(Hk < Hq),
        )
        return out[0].transpose(0, 1)

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KV4BitMetadata,
        layer: torch.nn.Module | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = query.shape[0]
        D = self.head_size
        Hq = self.num_heads
        S = self.max_num_kv_splits
        mid_o_buf = output_buf = None
        if is_workspace_manager_initialized():
            workspace = current_workspace_manager()
            if output is None:
                mid_o_buf, output_buf = workspace.get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                )
            else:
                (mid_o_buf,) = workspace.get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                )

        return triton_kv_4bit_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            scale=self.scale,
            hadamard_order=self.kv_4bit_config.hadamard_order,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            output=output,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )
