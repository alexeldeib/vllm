import math
from typing import Any

import torch

from vllm.triton_utils import tl, triton


_MIN_BLOCK_KV = 32
_MAX_BLOCK_H = 16
_TARGET_STAGE1_PROGRAMS = 512


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _select_decode_num_splits(
    batch: int,
    head_grid: int,
    max_num_kv_splits: int,
) -> int:
    """Choose the active split count under the cudagraph allocation limit."""
    max_num_kv_splits = max(1, max_num_kv_splits)
    programs_per_split = max(1, batch * head_grid)
    requested_splits = triton.cdiv(_TARGET_STAGE1_PROGRAMS, programs_per_split)
    return min(max_num_kv_splits, _next_power_of_2(requested_splits))


def get_kv_4bit_scale_zero_views(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return float32 views over the inline KV-4BIT scale/zero fields.

    The cache allocation remains a single uint8 tensor, but direct float32 views
    let Triton load/store scale and zero with one fp32 memory op instead of
    reconstructing fp32 values from four byte loads.
    """
    assert kv_cache.dtype == torch.uint8
    data_bytes = head_dim // 2
    k_scale_offset = data_bytes
    k_zero_offset = k_scale_offset + 4
    v_data_offset = data_bytes + 8
    v_scale_offset = v_data_offset + data_bytes
    v_zero_offset = v_scale_offset + 4

    byte_strides = tuple(
        stride * kv_cache.element_size() for stride in kv_cache.stride()
    )
    byte_storage_offset = kv_cache.storage_offset() * kv_cache.element_size()
    offsets = (k_scale_offset, k_zero_offset, v_scale_offset, v_zero_offset)
    align_values = (*byte_strides[:3], byte_storage_offset, *offsets)
    if any(value % 4 != 0 for value in align_values):
        raise ValueError(
            "KV-4BIT optimized scale views require 4-byte aligned cache "
            f"layout; got head_dim={head_dim}, strides={kv_cache.stride()}, "
            f"offsets={offsets}"
        )

    storage = kv_cache.untyped_storage()
    base = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(
        storage,
        0,
        (storage.nbytes() // 4,),
        (1,),
    )
    size = tuple(kv_cache.shape[:3])
    stride = tuple(byte_stride // 4 for byte_stride in byte_strides[:3])
    base_offset = byte_storage_offset // 4

    def make_view(byte_offset: int) -> torch.Tensor:
        return torch.as_strided(
            base,
            size=size,
            stride=stride,
            storage_offset=base_offset + byte_offset // 4,
        )

    return (
        make_view(k_scale_offset),
        make_view(k_zero_offset),
        make_view(v_scale_offset),
        make_view(v_zero_offset),
    )


@triton.jit
def _fwht_blocked_batch(
    x,
    width: tl.constexpr,
    log_order: tl.constexpr,
    block_h: tl.constexpr,
):
    idx = tl.arange(0, width)
    for s in tl.static_range(0, log_order):
        stride = 1 << s
        partner = idx ^ stride
        other = tl.gather(
            x,
            tl.broadcast_to(partner[None, :], [block_h, width]),
            1,
        )
        is_low = ((idx >> s) & 1) == 0
        x = tl.where(is_low[None, :], x + other, other - x)
    return x


@triton.jit
def _fwht_blocked_vector(
    x,
    width: tl.constexpr,
    log_order: tl.constexpr,
):
    idx = tl.arange(0, width)
    for s in tl.static_range(0, log_order):
        stride = 1 << s
        partner = idx ^ stride
        other = tl.gather(x, partner, 0)
        is_low = ((idx >> s) & 1) == 0
        x = tl.where(is_low, x + other, other - x)
    return x


@triton.jit
def _actual_kv_splits(
    seq_len,
    max_kv_splits: tl.constexpr,
    min_block_kv: tl.constexpr,
):
    kv_splits = tl.cdiv(seq_len, min_block_kv)
    kv_splits = tl.maximum(kv_splits, 1)
    return tl.minimum(kv_splits, max_kv_splits)


@triton.jit
def _kv_4bit_decode_stage1_grouped(
    Query_ptr,
    KV_cache_ptr,
    K_scale_ptr,
    K_zero_ptr,
    V_scale_ptr,
    V_zero_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_ks_block,
    stride_ks_pos,
    stride_ks_head,
    stride_vs_block,
    stride_vs_pos,
    stride_vs_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    V_DATA_OFFSET: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROTATE_Q: tl.constexpr,
    LOG_ORDER: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
):
    bid = tl.program_id(0)
    head_group = tl.program_id(1)
    sid = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid)
    kv_splits = _actual_kv_splits(seq_len, NUM_KV_SPLITS, MIN_BLOCK_KV)
    split_len = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    half = HEAD_DIM // 2
    half_offs = tl.arange(0, BLOCK_D // 2)
    half_mask = half_offs < half
    second_offs = half + half_offs
    second_mask = second_offs < HEAD_DIM

    groups_per_kv = tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    kv_head = head_group // groups_per_kv
    group_in_kv = head_group % groups_per_kv
    group_head_offsets = tl.arange(0, BLOCK_H)
    cur_heads = (
        kv_head * KV_GROUP_SIZE + group_in_kv * BLOCK_H + group_head_offsets
    )
    head_mask = (
        (group_in_kv * BLOCK_H + group_head_offsets) < KV_GROUP_SIZE
    ) & (cur_heads < NUM_Q_HEADS)

    out_base = (
        bid * stride_mid_b
        + cur_heads[:, None] * stride_mid_h
        + sid * stride_mid_s
    )

    if split_start >= split_end:
        tl.store(
            Mid_o_ptr + out_base + d_offs[None, :],
            tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32),
            mask=head_mask[:, None] & d_mask[None, :],
        )
        tl.store(
            Mid_o_ptr
            + bid * stride_mid_b
            + cur_heads * stride_mid_h
            + sid * stride_mid_s
            + HEAD_DIM,
            -float("inf"),
            mask=head_mask,
        )
        return

    q_base = bid * stride_qb + cur_heads[:, None] * stride_qh
    q = tl.load(
        Query_ptr + q_base + d_offs[None, :],
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    if ROTATE_Q:
        q = _fwht_blocked_batch(
            q.to(tl.float32) * PRE_SCALE,
            BLOCK_D,
            LOG_ORDER,
            BLOCK_H,
        ).to(q.dtype)

    q_first = tl.gather(
        q,
        tl.broadcast_to(half_offs[None, :], [BLOCK_H, BLOCK_D // 2]),
        1,
    )
    q_second = tl.gather(
        q,
        tl.broadcast_to(second_offs[None, :], [BLOCK_H, BLOCK_D // 2]),
        1,
    )
    q_sum = tl.sum(q_first, 1) + tl.sum(q_second, 1)

    kv_range = tl.arange(0, BLOCK_KV)
    m_prev = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_first = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)
    acc_second = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )
        scale_bases = (
            block_nums * stride_ks_block
            + page_off.to(tl.int64) * stride_ks_pos
            + tl.cast(kv_head, tl.int64) * stride_ks_head
        )

        k_packed = tl.load(
            KV_cache_ptr + slot_bases[None, :] + half_offs[:, None],
            mask=kv_mask[None, :] & half_mask[:, None],
            other=0,
        )
        k_scale = tl.load(K_scale_ptr + scale_bases, mask=kv_mask, other=1.0)
        k_zero = tl.load(K_zero_ptr + scale_bases, mask=kv_mask, other=0.0)
        k_lower = (k_packed & 0x0F).to(q_first.dtype)
        k_upper = ((k_packed >> 4) & 0x0F).to(q_first.dtype)

        scores = tl.dot(q_first, k_lower) + tl.dot(q_second, k_upper)
        scores = (scores - q_sum[:, None] * k_zero[None, :]) * (k_scale[None, :] * ATTN_SCALE)
        scores = tl.where(
            head_mask[:, None] & kv_mask[None, :],
            scores,
            -float("inf"),
        )

        n_e_max = tl.maximum(tl.max(scores, 1), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, None])

        v_packed = tl.load(
            KV_cache_ptr + slot_bases[:, None] + V_DATA_OFFSET + half_offs[None, :],
            mask=kv_mask[:, None] & half_mask[None, :],
            other=0,
        )
        v_scale_bases = (
            block_nums * stride_vs_block
            + page_off.to(tl.int64) * stride_vs_pos
            + tl.cast(kv_head, tl.int64) * stride_vs_head
        )
        v_scale = tl.load(V_scale_ptr + v_scale_bases, mask=kv_mask, other=1.0)
        v_zero = tl.load(V_zero_ptr + v_scale_bases, mask=kv_mask, other=0.0)
        v_lower = (v_packed & 0x0F).to(q_first.dtype)
        v_upper = ((v_packed >> 4) & 0x0F).to(q_first.dtype)
        p_scaled = p * v_scale[None, :]
        p_scaled_dot = p_scaled.to(v_lower.dtype)
        v_bias = tl.sum(p_scaled * v_zero[None, :], 1)

        acc_first = (
            acc_first * re_scale[:, None]
            + tl.dot(p_scaled_dot, v_lower)
            - v_bias[:, None]
        )
        acc_second = acc_second * re_scale[:, None] + tl.dot(
            p_scaled_dot,
            v_upper,
        ) - v_bias[:, None]
        l_prev = l_prev * re_scale + tl.sum(p, 1)
        m_prev = n_e_max

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_o_ptr + out_base + half_offs[None, :],
        acc_first / safe_l[:, None],
        mask=head_mask[:, None] & half_mask[None, :],
    )
    tl.store(
        Mid_o_ptr + out_base + second_offs[None, :],
        acc_second / safe_l[:, None],
        mask=head_mask[:, None] & second_mask[None, :],
    )
    tl.store(
        Mid_o_ptr
        + bid * stride_mid_b
        + cur_heads * stride_mid_h
        + sid * stride_mid_s
        + HEAD_DIM,
        m_prev + tl.log(safe_l),
        mask=head_mask,
    )


@triton.jit
def _kv_4bit_decode_stage2(
    Mid_o_ptr,
    Out_ptr,
    Seq_lens_ptr,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_out_b,
    stride_out_h,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    OUTPUT_FP16: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    seq_len = tl.load(Seq_lens_ptr + bid)
    kv_splits = _actual_kv_splits(seq_len, NUM_KV_SPLITS, MIN_BLOCK_KV)
    split_len = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    offs_v = bid * stride_mid_b + hid * stride_mid_h + d_offs
    offs_lse = bid * stride_mid_b + hid * stride_mid_h + HEAD_DIM

    for sid in range(0, NUM_KV_SPLITS):
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_end > split_start:
            tv = tl.load(
                Mid_o_ptr + offs_v + sid * stride_mid_s,
                mask=d_mask,
                other=0.0,
            )
            tlogic = tl.load(Mid_o_ptr + offs_lse + sid * stride_mid_s)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    result = acc / e_sum
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(
        Out_ptr + bid * stride_out_b + hid * stride_out_h + d_offs,
        result,
        mask=d_mask,
    )


@triton.jit
def _kv_4bit_dequant_kv_kernel(
    KV_cache_ptr,
    K_scale_ptr,
    K_zero_ptr,
    V_scale_ptr,
    V_zero_ptr,
    Block_table_ptr,
    K_out_ptr,
    V_out_ptr,
    stride_ko_h,
    stride_ko_s,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_ks_block,
    stride_ks_pos,
    stride_ks_head,
    stride_vs_block,
    stride_vs_pos,
    stride_vs_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    V_DATA_OFFSET: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTATE_K: tl.constexpr,
    LOG_ORDER: tl.constexpr,
    PRE_SCALE: tl.constexpr,
):
    """Dequantize one cached KV-4BIT token/head into dense K/V tensors."""
    pos = tl.program_id(0)
    hid = tl.program_id(1)

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )
    scale_base = (
        block_num * stride_ks_block
        + tl.cast(page_off, tl.int64) * stride_ks_pos
        + tl.cast(hid, tl.int64) * stride_ks_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    half = HEAD_DIM // 2
    packed_offs = d_offs % half
    is_upper = d_offs >= half

    k_packed = tl.load(
        KV_cache_ptr + slot_base + packed_offs,
        mask=d_mask,
        other=0,
    ).to(tl.int32)
    k_scale = tl.load(K_scale_ptr + scale_base)
    k_zero = tl.load(K_zero_ptr + scale_base)
    k_q = tl.where(is_upper, (k_packed >> 4) & 0x0F, k_packed & 0x0F).to(tl.float32)
    k_vals = (k_q - k_zero) * k_scale
    if ROTATE_K:
        # The cache stores normalized-Hadamard K. Applying the same transform
        # again recovers original-space K for FlashAttention prefill.
        k_vals = _fwht_blocked_vector(
            k_vals * PRE_SCALE,
            BLOCK_D,
            LOG_ORDER,
        )

    v_scale_base = (
        block_num * stride_vs_block
        + tl.cast(page_off, tl.int64) * stride_vs_pos
        + tl.cast(hid, tl.int64) * stride_vs_head
    )
    v_packed = tl.load(
        KV_cache_ptr + slot_base + V_DATA_OFFSET + packed_offs,
        mask=d_mask,
        other=0,
    ).to(tl.int32)
    v_scale = tl.load(V_scale_ptr + v_scale_base)
    v_zero = tl.load(V_zero_ptr + v_scale_base)
    v_q = tl.where(is_upper, (v_packed >> 4) & 0x0F, v_packed & 0x0F).to(tl.float32)
    v_vals = (v_q - v_zero) * v_scale

    ko_base = hid * stride_ko_h + pos * stride_ko_s
    vo_base = hid * stride_vo_h + pos * stride_vo_s
    tl.store(K_out_ptr + ko_base + d_offs, k_vals, mask=d_mask)
    tl.store(V_out_ptr + vo_base + d_offs, v_vals, mask=d_mask)


def triton_kv_4bit_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    *,
    hadamard_order: int,
) -> None:
    """Dequantize cached KV-4BIT K/V for prefill FlashAttention.

    ``k_out`` and ``v_out`` use shape ``[1, num_kv_heads, seq_len, head_dim]``.
    K is inverse-rotated back to original space to match SGLang's flattened
    prefill path, so raw Q/current K can be used directly.
    """
    assert kv_cache.dtype == torch.uint8
    assert k_out.shape == v_out.shape
    assert k_out.ndim == 4 and k_out.shape[0] == 1

    _, num_kv_heads, seq_len, head_dim = k_out.shape
    if seq_len <= 0:
        return

    if hadamard_order < 2 or hadamard_order & (hadamard_order - 1):
        raise ValueError(
            f"hadamard_order must be a power of two >= 2, got {hadamard_order}"
        )
    if head_dim % hadamard_order:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by hadamard_order "
            f"({hadamard_order})"
        )

    k_scale, k_zero, v_scale, v_zero = get_kv_4bit_scale_zero_views(
        kv_cache,
        head_dim,
    )
    block_size = kv_cache.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    v_data_offset = head_dim // 2 + 8
    log_order = int(math.log2(hadamard_order))
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))

    _kv_4bit_dequant_kv_kernel[(seq_len, num_kv_heads)](
        kv_cache.view(-1),
        k_scale,
        k_zero,
        v_scale,
        v_zero,
        block_table,
        k_out,
        v_out,
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(1),
        v_out.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        V_DATA_OFFSET=v_data_offset,
        BLOCK_D=block_d,
        ROTATE_K=True,
        LOG_ORDER=log_order,
        PRE_SCALE=pre_scale,
        num_warps=4,
    )


@triton.jit
def _kv_4bit_gather_k_kernel(
    KV_cache_ptr,
    K_scale_ptr,
    K_zero_ptr,
    Block_table_ptr,
    Cu_seq_lens_ptr,
    Token_to_seq_ptr,
    Seq_starts_ptr,
    K_out_ptr,
    Num_tokens: tl.constexpr,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_ks_block,
    stride_ks_pos,
    stride_ks_head,
    stride_bt_b,
    stride_ko_t,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTATE_K: tl.constexpr,
    LOG_ORDER: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    HAS_SEQ_STARTS: tl.constexpr,
):
    token_id = tl.program_id(0)
    if token_id >= Num_tokens:
        return

    batch_id = tl.load(Token_to_seq_ptr + token_id).to(tl.int64)
    batch_start = tl.load(Cu_seq_lens_ptr + batch_id)
    batch_end = tl.load(Cu_seq_lens_ptr + batch_id + 1)
    if token_id >= batch_end:
        return

    batch_offset = token_id - batch_start
    if HAS_SEQ_STARTS:
        batch_offset += tl.load(Seq_starts_ptr + batch_id)

    table_id = batch_offset // BLOCK_SIZE
    slot_id = batch_offset % BLOCK_SIZE
    block_id = tl.load(Block_table_ptr + batch_id * stride_bt_b + table_id)
    slot_base = block_id * stride_cache_block + slot_id * stride_cache_pos
    scale_base = block_id * stride_ks_block + slot_id * stride_ks_pos

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    half = HEAD_DIM // 2
    packed_offs = d_offs % half
    is_upper = d_offs >= half

    k_packed = tl.load(
        KV_cache_ptr + slot_base + packed_offs,
        mask=d_mask,
        other=0,
    )
    k_scale = tl.load(K_scale_ptr + scale_base)
    k_zero = tl.load(K_zero_ptr + scale_base)
    k_q = tl.where(is_upper, (k_packed >> 4) & 0x0F, k_packed & 0x0F).to(
        tl.float32
    )
    k_vals = (k_q - k_zero) * k_scale
    if ROTATE_K:
        k_vals = _fwht_blocked_vector(k_vals, BLOCK_D, LOG_ORDER) * PRE_SCALE

    tl.store(K_out_ptr + token_id * stride_ko_t + d_offs, k_vals, mask=d_mask)


def triton_kv_4bit_gather_k(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    k_out: torch.Tensor,
    num_tokens: int,
    *,
    hadamard_order: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather and dequantize KV-4BIT K entries into an MLA workspace."""
    assert kv_cache.dtype == torch.uint8
    assert k_out.ndim == 2
    if isinstance(num_tokens, torch.Tensor):
        num_tokens = int(num_tokens.item())
    else:
        num_tokens = int(num_tokens)
    if num_tokens <= 0:
        return

    if kv_cache.ndim == 3:
        kv_cache = kv_cache.unsqueeze(2)
    assert kv_cache.ndim == 4 and kv_cache.shape[2] == 1

    head_dim = k_out.shape[-1]
    if hadamard_order < 2 or hadamard_order & (hadamard_order - 1):
        raise ValueError(
            f"hadamard_order must be a power of two >= 2, got {hadamard_order}"
        )
    if head_dim % hadamard_order:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by hadamard_order "
            f"({hadamard_order})"
        )

    k_scale, k_zero, _, _ = get_kv_4bit_scale_zero_views(kv_cache, head_dim)
    block_d = triton.next_power_of_2(head_dim)
    log_order = int(math.log2(hadamard_order))
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))
    seq_starts_arg = seq_starts if seq_starts is not None else cu_seq_lens

    _kv_4bit_gather_k_kernel[(num_tokens,)](
        kv_cache.view(-1),
        k_scale,
        k_zero,
        block_table,
        cu_seq_lens,
        token_to_seq,
        seq_starts_arg,
        k_out,
        Num_tokens=num_tokens,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        stride_ks_block=k_scale.stride(0),
        stride_ks_pos=k_scale.stride(1),
        stride_ks_head=k_scale.stride(2),
        stride_bt_b=block_table.stride(0),
        stride_ko_t=k_out.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=kv_cache.shape[1],
        BLOCK_D=block_d,
        ROTATE_K=True,
        LOG_ORDER=log_order,
        PRE_SCALE=pre_scale,
        HAS_SEQ_STARTS=seq_starts is not None,
        num_warps=4,
    )


def triton_kv_4bit_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    *,
    hadamard_order: int,
    max_num_kv_splits: int = 32,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    buf_holder: Any = None,
) -> torch.Tensor:
    query = query.contiguous()
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device
    block_d = triton.next_power_of_2(D)
    # BLOCK_H is used as the upper bound of tl.arange() in the Triton
    # kernel, and Triton requires that bound to be a power of two. For
    # non-power-of-two GQA groups, pad the tile and rely on head_mask in
    # the kernel to suppress dummy heads.
    block_h = min(_MAX_BLOCK_H, triton.next_power_of_2(kv_group_size))
    head_grid = Hk * triton.cdiv(kv_group_size, block_h)
    num_splits = _select_decode_num_splits(B, head_grid, max_num_kv_splits)

    if hadamard_order < 2 or hadamard_order & (hadamard_order - 1):
        raise ValueError(
            f"hadamard_order must be a power of two >= 2, got {hadamard_order}"
        )
    if D % hadamard_order:
        raise ValueError(
            f"head_dim ({D}) must be divisible by hadamard_order "
            f"({hadamard_order})"
        )
    log_order = int(math.log2(hadamard_order))
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))

    k_scale, k_zero, v_scale, v_zero = get_kv_4bit_scale_zero_views(kv_cache, D)
    v_data_offset = D // 2 + 8

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= num_splits
    ):
        mid_o = mid_o_buf[:B, :Hq, :num_splits, :]
    else:
        mid_o = torch.empty(
            B, Hq, num_splits, D + 1, dtype=torch.float32, device=device
        )
        if buf_holder is not None:
            buf_holder._kv_4bit_mid_o_buf = mid_o

    _kv_4bit_decode_stage1_grouped[(B, head_grid, num_splits)](
        query,
        kv_cache,
        k_scale,
        k_zero,
        v_scale,
        v_zero,
        block_table,
        seq_lens,
        mid_o,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=num_splits,
        KV_GROUP_SIZE=kv_group_size,
        V_DATA_OFFSET=v_data_offset,
        ATTN_SCALE=scale,
        BLOCK_D=block_d,
        BLOCK_KV=128,
        BLOCK_H=block_h,
        ROTATE_Q=True,
        LOG_ORDER=log_order,
        PRE_SCALE=pre_scale,
        NUM_Q_HEADS=Hq,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        num_warps=4,
        num_stages=2,
    )

    query_dtype = query.dtype
    if output is not None:
        if output.ndim == 2:
            out = output.view(B, Hq, D)
        else:
            out = output
    elif (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == query_dtype
    ):
        out = output_buf[:B, :Hq, :D]
    else:
        out = torch.empty(B, Hq, D, dtype=query_dtype, device=device)
        if buf_holder is not None:
            buf_holder._kv_4bit_output_buf = out
    out_dtype = out.dtype

    _kv_4bit_decode_stage2[(B, Hq)](
        mid_o,
        out,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        out.stride(0),
        out.stride(1),
        NUM_KV_SPLITS=num_splits,
        BLOCK_D=block_d,
        HEAD_DIM=D,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )
    return out
