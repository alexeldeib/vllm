import math

import pytest
import torch

from vllm.model_executor.layers.quantization.kv_4bit.config import KV4BitConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_kv_4bit_decode import (
    triton_kv_4bit_dequant_kv,
    triton_kv_4bit_decode_attention,
)
from vllm.v1.attention.ops.triton_kv_4bit_store import triton_kv_4bit_store

DEVICE_TYPE = current_platform.device_type
pytestmark = pytest.mark.skipif(
    DEVICE_TYPE != "cuda" or not torch.cuda.is_available(),
    reason="KV-4BIT Triton kernels require CUDA",
)


def _hadamard(order: int, device: torch.device) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < order:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(order)


def _rotate_blocked(x: torch.Tensor, order: int) -> torch.Tensor:
    h = _hadamard(order, x.device)
    return (x.float().reshape(*x.shape[:-1], -1, order) @ h).reshape(x.shape)


def _quant_params_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    x_min = x.amin(dim=-1)
    x_max = x.amax(dim=-1)
    scale = ((x_max - x_min) / 15.0).clamp(min=1.0e-8)
    zero = -x_min / scale
    return scale, zero


def _slot_mapping(batch: int, seq_len: int, block_size: int, device: str):
    pages_per_req = cdiv(seq_len, block_size)
    block_table = torch.arange(
        batch * pages_per_req,
        device=device,
        dtype=torch.int64,
    ).view(batch, pages_per_req)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64)
    slots = (
        block_table[:, positions // block_size] * block_size
        + positions[None, :] % block_size
    )
    return block_table, slots.reshape(-1)


def _dequant_slots_with_params(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    data_bytes = head_dim // 2
    k_scale_offset = data_bytes
    v_data_offset = data_bytes + 8
    v_scale_offset = v_data_offset + data_bytes

    blocks = slot_mapping // block_size
    offsets = slot_mapping % block_size
    slots = kv_cache[blocks, offsets]

    def load_fp32(byte_slice: torch.Tensor) -> torch.Tensor:
        return byte_slice.contiguous().view(torch.float32).squeeze(-1)

    def unpack(data_offset: int, scale_offset: int) -> torch.Tensor:
        packed = slots[..., data_offset : data_offset + data_bytes]
        lo = (packed & 0xF).float()
        hi = ((packed >> 4) & 0xF).float()
        q = torch.cat([lo, hi], dim=-1)
        scale = load_fp32(slots[..., scale_offset : scale_offset + 4])
        zero = load_fp32(slots[..., scale_offset + 4 : scale_offset + 8])
        return (q - zero.unsqueeze(-1)) * scale.unsqueeze(-1)

    k = unpack(0, k_scale_offset)
    v = unpack(v_data_offset, v_scale_offset)
    k_scale = load_fp32(slots[..., k_scale_offset : k_scale_offset + 4])
    k_zero = load_fp32(slots[..., k_scale_offset + 4 : k_scale_offset + 8])
    v_scale = load_fp32(slots[..., v_scale_offset : v_scale_offset + 4])
    v_zero = load_fp32(slots[..., v_scale_offset + 4 : v_scale_offset + 8])
    return k, v, k_scale, k_zero, v_scale, v_zero


def _dequant_slots(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k, v, _, _, _, _ = _dequant_slots_with_params(
        kv_cache,
        slot_mapping,
        block_size,
        head_dim,
    )
    return k, v


def _assert_quantized_within_half_step(
    actual: torch.Tensor,
    source: torch.Tensor,
    stored_scale: torch.Tensor,
    stored_zero: torch.Tensor,
) -> None:
    ref_scale, ref_zero = _quant_params_ref(source)
    torch.testing.assert_close(stored_scale, ref_scale, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stored_zero, ref_zero, atol=1e-4, rtol=1e-5)

    # Values exactly on a quantization threshold may round to either adjacent
    # INT4 code depending on tiny arithmetic differences between Triton and
    # PyTorch. Both choices are correct nearest-neighbor quantization, so test
    # the quantizer contract directly: dequantization error is <= half a bin.
    err = (actual - source.float()).abs()
    bound = stored_scale.unsqueeze(-1) * 0.5 + 1e-3
    assert torch.all(err <= bound), (
        f"max quantization error {err.max().item()} exceeds "
        f"half-step bound {bound.max().item()}"
    )


def test_kv_4bit_store_matches_reference():
    torch.manual_seed(0)
    batch = 2
    seq_len = 11
    num_kv_heads = 3
    head_dim = 64
    block_size = 8
    hadamard_order = 32
    cfg = KV4BitConfig(
        head_dim=head_dim,
        hadamard_order=hadamard_order,
    )

    device = DEVICE_TYPE
    block_table, slot_mapping = _slot_mapping(batch, seq_len, block_size, device)
    num_blocks = int(block_table.numel())
    key = torch.randn(
        batch * seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    kv_cache = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )

    triton_kv_4bit_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        hadamard_order=hadamard_order,
    )

    cached_k, cached_v, k_scale, k_zero, v_scale, v_zero = (
        _dequant_slots_with_params(kv_cache, slot_mapping, block_size, head_dim)
    )
    key_ref = _rotate_blocked(key, hadamard_order)

    _assert_quantized_within_half_step(cached_k, key_ref, k_scale, k_zero)
    _assert_quantized_within_half_step(cached_v, value, v_scale, v_zero)


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [
        pytest.param(4, 2, id="power_of_two_gqa_group"),
        pytest.param(6, 2, id="non_power_of_two_gqa_group"),
    ],
)
def test_kv_4bit_decode_matches_quantized_reference(
    num_heads: int,
    num_kv_heads: int,
):
    torch.manual_seed(1)
    batch = 2
    seq_len = 19
    head_dim = 64
    block_size = 8
    hadamard_order = 32
    scale = 1.0 / math.sqrt(head_dim)
    cfg = KV4BitConfig(
        head_dim=head_dim,
        hadamard_order=hadamard_order,
    )

    device = DEVICE_TYPE
    block_table, slot_mapping = _slot_mapping(batch, seq_len, block_size, device)
    num_blocks = int(block_table.numel())
    key = torch.randn(
        batch * seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    query = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int64)
    kv_cache = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )

    triton_kv_4bit_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        hadamard_order=hadamard_order,
    )
    output = torch.empty_like(query)
    actual = triton_kv_4bit_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        hadamard_order=hadamard_order,
        max_num_kv_splits=4,
        output=output,
    )
    assert actual.data_ptr() == output.data_ptr()

    cached_k, cached_v = _dequant_slots(kv_cache, slot_mapping, block_size, head_dim)
    cached_k = cached_k.view(batch, seq_len, num_kv_heads, head_dim)
    cached_v = cached_v.view(batch, seq_len, num_kv_heads, head_dim)
    query_ref = _rotate_blocked(query, hadamard_order)

    expected = torch.empty_like(actual, dtype=torch.float32)
    kv_group_size = num_heads // num_kv_heads
    for b in range(batch):
        for h in range(num_heads):
            kv_h = h // kv_group_size
            scores = (cached_k[b, :, kv_h] @ query_ref[b, h]) * scale
            probs = torch.softmax(scores.float(), dim=0)
            expected[b, h] = probs @ cached_v[b, :, kv_h].float()

    torch.testing.assert_close(actual.float(), expected, atol=7e-2, rtol=7e-2)


def test_kv_4bit_dequant_kv_matches_reference():
    torch.manual_seed(2)
    seq_len = 17
    num_kv_heads = 2
    head_dim = 64
    block_size = 8
    hadamard_order = 32
    cfg = KV4BitConfig(
        head_dim=head_dim,
        hadamard_order=hadamard_order,
    )

    device = DEVICE_TYPE
    block_table, slot_mapping = _slot_mapping(1, seq_len, block_size, device)
    num_blocks = int(block_table.numel())
    key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    kv_cache = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )

    triton_kv_4bit_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        hadamard_order=hadamard_order,
    )

    alloc_len = cdiv(seq_len, block_size) * block_size
    k_out = torch.empty(
        1,
        num_kv_heads,
        alloc_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_out = torch.empty_like(k_out)
    triton_kv_4bit_dequant_kv(
        kv_cache,
        block_table,
        k_out,
        v_out,
        hadamard_order=hadamard_order,
    )

    cached_k, cached_v = _dequant_slots(kv_cache, slot_mapping, block_size, head_dim)
    cached_k = _rotate_blocked(cached_k, hadamard_order)
    expected_k = cached_k.view(seq_len, num_kv_heads, head_dim)
    expected_v = cached_v.view(seq_len, num_kv_heads, head_dim)

    actual_k = k_out[0, :, :seq_len, :].transpose(0, 1).float()
    actual_v = v_out[0, :, :seq_len, :].transpose(0, 1).float()
    torch.testing.assert_close(actual_k, expected_k, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(actual_v, expected_v, atol=5e-2, rtol=5e-2)
