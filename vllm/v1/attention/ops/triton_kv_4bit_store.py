import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_kv_4bit_decode import (
    get_kv_4bit_scale_zero_views,
)


@triton.jit
def _fwht_blocked(x, width: tl.constexpr, log_order: tl.constexpr):
    idx = tl.arange(0, width)
    for s in tl.static_range(0, log_order):
        stride = 1 << s
        partner = idx ^ stride
        other = tl.gather(x, partner, 0)
        is_low = ((idx >> s) & 1) == 0
        x = tl.where(is_low, x + other, other - x)
    return x


@triton.jit
def _quant_pack_store_vector(
    X_ptr,
    KV_cache_ptr,
    Scale_ptr,
    Zero_ptr,
    base,
    slot_base,
    scale_base,
    data_offset: tl.constexpr,
    dim_full,
    dim_half,
    full_mask,
    half_mask,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    D_HALF_PAD: tl.constexpr,
    ROTATE: tl.constexpr,
    LOG_ORDER: tl.constexpr,
    PRE_SCALE: tl.constexpr,
):
    x = tl.load(X_ptr + base + dim_full, mask=full_mask, other=0.0).to(tl.float32)
    if ROTATE:
        x = _fwht_blocked(x * PRE_SCALE, D_PAD, LOG_ORDER)

    half = D // 2
    safe_lo = tl.where(half_mask, dim_half, 0)
    safe_hi = tl.where(half_mask, dim_half + half, 0)
    vals_lo = tl.gather(x, safe_lo, 0)
    vals_hi = tl.gather(x, safe_hi, 0)

    vals_lo_min = tl.where(half_mask, vals_lo, float("inf"))
    vals_hi_min = tl.where(half_mask, vals_hi, float("inf"))
    vals_lo_max = tl.where(half_mask, vals_lo, -float("inf"))
    vals_hi_max = tl.where(half_mask, vals_hi, -float("inf"))
    x_min = tl.minimum(tl.min(vals_lo_min, axis=0), tl.min(vals_hi_min, axis=0))
    x_max = tl.maximum(tl.max(vals_lo_max, axis=0), tl.max(vals_hi_max, axis=0))
    scale = tl.maximum((x_max - x_min) / 15.0, 1.0e-8)
    zero = -x_min / scale

    q_lo_f = tl.minimum(tl.maximum((vals_lo - x_min) / scale + 0.5, 0.0), 15.0)
    q_hi_f = tl.minimum(tl.maximum((vals_hi - x_min) / scale + 0.5, 0.0), 15.0)
    q_lo = q_lo_f.to(tl.uint8)
    q_hi = q_hi_f.to(tl.uint8)
    packed = q_lo | (q_hi << 4)

    tl.store(
        KV_cache_ptr + slot_base + data_offset + dim_half,
        packed,
        mask=half_mask,
    )
    tl.store(Scale_ptr + scale_base, scale)
    tl.store(Zero_ptr + scale_base, zero)


@triton.jit
def _kv_4bit_store_kernel(
    Key_ptr,
    Value_ptr,
    KV_cache_ptr,
    K_scale_ptr,
    K_zero_ptr,
    V_scale_ptr,
    V_zero_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    stride_ks_block: tl.constexpr,
    stride_ks_pos: tl.constexpr,
    stride_ks_head: tl.constexpr,
    stride_vs_block: tl.constexpr,
    stride_vs_pos: tl.constexpr,
    stride_vs_head: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D_PAD: tl.constexpr,
    D_HALF_PAD: tl.constexpr,
    V_DATA_OFFSET: tl.constexpr,
    ROTATE_K: tl.constexpr,
    LOG_ORDER: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_group = tl.program_id(1)

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    dim_full = tl.arange(0, D_PAD)
    dim_half = tl.arange(0, D_HALF_PAD)
    full_mask = dim_full < D
    half_mask = dim_half < (D // 2)

    for head_offset in tl.static_range(0, HEADS_PER_PROGRAM):
        head_idx = head_group * HEADS_PER_PROGRAM + head_offset
        if head_idx < H:
            slot_base = (
                blk * stride_cache_block
                + off * stride_cache_pos
                + tl.cast(head_idx, tl.int64) * stride_cache_head
            )
            k_scale_base = (
                blk * stride_ks_block
                + off * stride_ks_pos
                + tl.cast(head_idx, tl.int64) * stride_ks_head
            )
            v_scale_base = (
                blk * stride_vs_block
                + off * stride_vs_pos
                + tl.cast(head_idx, tl.int64) * stride_vs_head
            )
            base = (token_idx * H + head_idx) * D

            _quant_pack_store_vector(
                Key_ptr,
                KV_cache_ptr,
                K_scale_ptr,
                K_zero_ptr,
                base,
                slot_base,
                k_scale_base,
                data_offset=0,
                dim_full=dim_full,
                dim_half=dim_half,
                full_mask=full_mask,
                half_mask=half_mask,
                D=D,
                D_PAD=D_PAD,
                D_HALF_PAD=D_HALF_PAD,
                ROTATE=ROTATE_K,
                LOG_ORDER=LOG_ORDER,
                PRE_SCALE=PRE_SCALE,
            )
            _quant_pack_store_vector(
                Value_ptr,
                KV_cache_ptr,
                V_scale_ptr,
                V_zero_ptr,
                base,
                slot_base,
                v_scale_base,
                data_offset=V_DATA_OFFSET,
                dim_full=dim_full,
                dim_half=dim_half,
                full_mask=full_mask,
                half_mask=half_mask,
                D=D,
                D_PAD=D_PAD,
                D_HALF_PAD=D_HALF_PAD,
                ROTATE=False,
                LOG_ORDER=LOG_ORDER,
                PRE_SCALE=PRE_SCALE,
            )


def triton_kv_4bit_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    hadamard_order: int,
) -> None:
    """Quantize and store K/V into the KV-4BIT combined cache."""
    N, H, D = key.shape
    assert value.shape == key.shape
    assert D % 2 == 0

    if hadamard_order < 2 or hadamard_order & (hadamard_order - 1):
        raise ValueError(
            f"hadamard_order must be a power of two >= 2, got {hadamard_order}"
        )
    if D % hadamard_order:
        raise ValueError(
            f"head_dim ({D}) must be divisible by hadamard_order "
            f"({hadamard_order})"
        )

    d_pad = triton.next_power_of_2(D)
    d_half_pad = triton.next_power_of_2(D // 2)
    data_bytes = D // 2
    v_data_offset = data_bytes + 8

    block_size = kv_cache.shape[1]
    log_order = int(math.log2(hadamard_order))
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))
    k_scale, k_zero, v_scale, v_zero = get_kv_4bit_scale_zero_views(kv_cache, D)
    heads_per_program = min(8, H) if d_pad < 512 else 1

    _kv_4bit_store_kernel[(N, triton.cdiv(H, heads_per_program))](
        key.reshape(N * H, D).contiguous(),
        value.reshape(N * H, D).contiguous(),
        kv_cache.view(-1),
        k_scale,
        k_zero,
        v_scale,
        v_zero,
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        stride_ks_block=k_scale.stride(0),
        stride_ks_pos=k_scale.stride(1),
        stride_ks_head=k_scale.stride(2),
        stride_vs_block=v_scale.stride(0),
        stride_vs_pos=v_scale.stride(1),
        stride_vs_head=v_scale.stride(2),
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        D_PAD=d_pad,
        D_HALF_PAD=d_half_pad,
        V_DATA_OFFSET=v_data_offset,
        ROTATE_K=True,
        LOG_ORDER=log_order,
        PRE_SCALE=pre_scale,
        HEADS_PER_PROGRAM=heads_per_program,
        num_warps=4,
        num_stages=1,
    )
