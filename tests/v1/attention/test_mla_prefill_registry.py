# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend registry."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.v1.attention.backends.mla.prefill.base import (
    MLAPrefillBackend,
    flashinfer_log2_lse_to_natural_lse,
)
from vllm.v1.attention.backends.mla.prefill.flashinfer import FlashInferPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)
from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import (
    TrtllmRaggedPrefillBackend,
)


class CustomMLAPrefillBackend(MLAPrefillBackend):
    """Mock custom MLA prefill backend for testing."""

    supported_dtypes = [torch.bfloat16, torch.float16]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
        raise NotImplementedError

    def run_prefill_context_chunk(self, chunk_idx, q, k, v):
        raise NotImplementedError


class FakeFlashInferWrapper:
    def __init__(self, ret: object) -> None:
        self.ret = ret
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.ret


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clear any overrides after each test."""
    yield
    for member in MLAPrefillBackendEnum:
        member.clear_override()


def test_custom_is_not_alias_of_any_backend():
    all_backends = list(MLAPrefillBackendEnum)

    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is MLAPrefillBackendEnum.CUSTOM:
            aliases.append(backend.name)

    assert len(aliases) == 0, (
        f"BUG! CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(MLAPrefillBackendEnum.CUSTOM.value)}\n"
        f"All MLA prefill backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )

    assert MLAPrefillBackendEnum.CUSTOM.name == "CUSTOM"


def test_custom_unregistered_raises():
    with pytest.raises(ValueError, match="must be registered before use"):
        MLAPrefillBackendEnum.CUSTOM.get_path()


def test_register_custom_backend_with_class_path():
    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.CUSTOM,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )

    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()

    class_path = MLAPrefillBackendEnum.CUSTOM.get_path()
    assert class_path == (
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
    )

    backend_cls = MLAPrefillBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM"


def test_register_custom_backend_as_decorator():
    @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
    class DecoratedPrefillBackend(MLAPrefillBackend):
        supported_dtypes = [torch.bfloat16]

        @staticmethod
        def get_name() -> str:
            return "DECORATED"

        def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
            raise NotImplementedError

        def run_prefill_context_chunk(self, chunk_idx, q, k, v):
            raise NotImplementedError

    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()
    assert "DecoratedPrefillBackend" in MLAPrefillBackendEnum.CUSTOM.get_path()


def test_override_existing_backend():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.FLASH_ATTN,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )

    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() != original_path

    backend_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
    assert backend_cls.get_name() == "CUSTOM"


def test_clear_override():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.FLASH_ATTN,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )
    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()

    MLAPrefillBackendEnum.FLASH_ATTN.clear_override()
    assert not MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() == original_path


def test_unknown_backend_name_raises():
    with pytest.raises(ValueError, match="Unknown MLA prefill backend"):
        MLAPrefillBackendEnum["NONEXISTENT"]


def test_flashinfer_log2_lse_to_natural_lse_converts_base2_to_natural_log():
    flashinfer_lse = torch.tensor([[0.0, 1.0, 8.0, -float("inf")]], dtype=torch.float32)

    softmax_lse = flashinfer_log2_lse_to_natural_lse(flashinfer_lse)

    expected = torch.log(torch.tensor(2.0)) * flashinfer_lse
    torch.testing.assert_close(softmax_lse, expected)


def _expected_natural_lse(raw_lse: torch.Tensor) -> torch.Tensor:
    return (raw_lse * torch.log(torch.tensor(2.0))).transpose(0, 1).contiguous()


def test_flashinfer_new_tokens_converts_lse_at_backend_boundary():
    raw_lse = torch.tensor([[1.0, 2.0, -float("inf")]], dtype=torch.float32)
    attn_out = torch.ones((1, 3, 4), dtype=torch.float32)
    wrapper = FakeFlashInferWrapper((attn_out, raw_lse))
    backend = FlashInferPrefillBackend.__new__(FlashInferPrefillBackend)
    backend._prefill_main = wrapper  # type: ignore[assignment]

    out, lse = backend.run_prefill_new_tokens(
        torch.empty((1, 3, 5)),
        torch.empty((1, 3, 5)),
        torch.empty((1, 3, 4)),
        return_softmax_lse=True,
    )

    assert out is attn_out
    assert wrapper.calls[0]["return_lse"] is True
    torch.testing.assert_close(lse, _expected_natural_lse(raw_lse))


def test_flashinfer_context_chunk_converts_lse_at_backend_boundary():
    raw_lse = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    attn_out = torch.ones((1, 2, 4), dtype=torch.float32)
    wrapper = FakeFlashInferWrapper((attn_out, raw_lse))
    backend = FlashInferPrefillBackend.__new__(FlashInferPrefillBackend)
    backend._prefill_chunks = [wrapper]  # type: ignore[list-item]

    out, lse = backend.run_prefill_context_chunk(
        0,
        torch.empty((1, 2, 5)),
        torch.empty((1, 2, 5)),
        torch.empty((1, 2, 4)),
    )

    assert out is attn_out
    assert wrapper.calls[0]["return_lse"] is True
    torch.testing.assert_close(lse, _expected_natural_lse(raw_lse))


def _install_fake_trtllm_ragged(
    monkeypatch: pytest.MonkeyPatch,
    raw_lse: torch.Tensor,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    flashinfer_module = ModuleType("flashinfer")
    prefill_module = ModuleType("flashinfer.prefill")

    def trtllm_ragged_attention_deepseek(
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        return kwargs["out"], raw_lse

    prefill_module.trtllm_ragged_attention_deepseek = (  # type: ignore[attr-defined]
        trtllm_ragged_attention_deepseek
    )
    flashinfer_module.prefill = prefill_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill_module)
    return calls


def _make_trtllm_ragged_backend() -> TrtllmRaggedPrefillBackend:
    backend = TrtllmRaggedPrefillBackend.__new__(TrtllmRaggedPrefillBackend)
    backend.scale = 0.125
    backend._workspace_buffer = torch.empty(0, dtype=torch.uint8)
    backend._query_seq_lens = torch.tensor([1], dtype=torch.int32)
    backend._query_seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
    backend._prefill_metadata = SimpleNamespace(
        output_dtype=torch.float32,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        chunked_context=SimpleNamespace(
            seq_lens=[torch.tensor([2], dtype=torch.int32)],
            max_seq_lens=[2],
            cu_seq_lens=[torch.tensor([0, 2], dtype=torch.int32)],
        ),
    )
    return backend


def test_trtllm_ragged_new_tokens_converts_lse_at_backend_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    raw_lse = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    calls = _install_fake_trtllm_ragged(monkeypatch, raw_lse)
    backend = _make_trtllm_ragged_backend()

    _, lse = backend.run_prefill_new_tokens(
        torch.empty((1, 2, 5)),
        torch.empty((1, 2, 5)),
        torch.empty((1, 2, 4)),
        return_softmax_lse=True,
    )

    assert calls[0]["return_lse"] is True
    assert calls[0]["is_causal"] is True
    torch.testing.assert_close(
        cast(torch.Tensor, calls[0]["q_seq_lens_cpu"]),
        torch.tensor([1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cast(torch.Tensor, calls[0]["kv_seq_lens_cpu"]),
        torch.tensor([1], dtype=torch.int32),
    )
    torch.testing.assert_close(lse, _expected_natural_lse(raw_lse))


def test_trtllm_ragged_context_chunk_converts_lse_at_backend_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    raw_lse = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
    calls = _install_fake_trtllm_ragged(monkeypatch, raw_lse)
    backend = _make_trtllm_ragged_backend()

    _, lse = backend.run_prefill_context_chunk(
        0,
        torch.empty((1, 2, 5)),
        torch.empty((2, 2, 5)),
        torch.empty((2, 2, 4)),
    )

    assert calls[0]["return_lse"] is True
    assert calls[0]["is_causal"] is False
    torch.testing.assert_close(
        cast(torch.Tensor, calls[0]["q_seq_lens_cpu"]),
        torch.tensor([1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cast(torch.Tensor, calls[0]["kv_seq_lens_cpu"]),
        torch.tensor([2], dtype=torch.int32),
    )
    torch.testing.assert_close(lse, _expected_natural_lse(raw_lse))


def _merge_two_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    prefix_exp = torch.exp(prefix_lse - max_lse)
    suffix_exp = torch.exp(suffix_lse - max_lse)
    denom = prefix_exp + suffix_exp

    prefix_scale = (prefix_exp / denom).transpose(0, 1).unsqueeze(-1)
    suffix_scale = (suffix_exp / denom).transpose(0, 1).unsqueeze(-1)
    output = prefix_output * prefix_scale + suffix_output * suffix_scale
    lse = torch.log(denom) + max_lse
    return output, lse


def test_empty_attention_state_is_neutral_for_lse_merge():
    suffix_output = torch.tensor([[[0.25, -0.5, 1.25]]], dtype=torch.float32)
    suffix_lse = torch.zeros((1, 1), dtype=torch.float32)
    stale_empty_output = torch.full_like(suffix_output, 1024.0)

    neutral_output, neutral_lse = _merge_two_attention_states(
        stale_empty_output,
        torch.full_like(suffix_lse, -float("inf")),
        suffix_output,
        suffix_lse,
    )

    torch.testing.assert_close(neutral_output, suffix_output)
    torch.testing.assert_close(neutral_lse, suffix_lse)


def test_finite_empty_attention_state_reproduces_corrupted_lse_merge():
    valid_output = torch.tensor([[[0.25, -0.5, 1.25]]], dtype=torch.float32)
    valid_lse = torch.zeros((1, 1), dtype=torch.float32)
    stale_empty_output = torch.full_like(valid_output, 1024.0)

    stale_output, stale_lse = _merge_two_attention_states(
        stale_empty_output,
        torch.zeros_like(valid_lse),
        valid_output,
        valid_lse,
    )

    expected_stale_output = (stale_empty_output + valid_output) / 2
    expected_stale_lse = torch.full_like(valid_lse, torch.log(torch.tensor(2.0)).item())

    torch.testing.assert_close(stale_output, expected_stale_output)
    torch.testing.assert_close(stale_lse, expected_stale_lse)
    assert not torch.allclose(stale_output, valid_output)


def test_flashinfer_lse_base_conversion_preserves_merge_weights():
    prefix_output = torch.zeros((1, 1, 1), dtype=torch.float32)
    suffix_output = torch.ones((1, 1, 1), dtype=torch.float32)
    flashinfer_prefix_lse = torch.tensor([[4.0]], dtype=torch.float32)
    suffix_lse = torch.zeros((1, 1), dtype=torch.float32)

    merged_output, _ = _merge_two_attention_states(
        prefix_output,
        flashinfer_log2_lse_to_natural_lse(flashinfer_prefix_lse),
        suffix_output,
        suffix_lse,
    )

    torch.testing.assert_close(merged_output, torch.tensor([[[1.0 / 17.0]]]))

    wrongly_scaled_output, _ = _merge_two_attention_states(
        prefix_output,
        flashinfer_prefix_lse,
        suffix_output,
        suffix_lse,
    )

    assert not torch.allclose(wrongly_scaled_output, merged_output)
