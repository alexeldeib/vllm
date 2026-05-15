# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend selector."""

import sys
import types
from inspect import signature
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention import mla_attention as mla_attention_module
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLACommonImpl,
    MLACommonMetadataBuilder,
    backend_supports_prefill_query_quantization,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill import (
    flashinfer as flashinfer_prefill_module,
)
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.flashinfer import (
    FlashInferPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.mla.prefill.selector import (
    MLAPrefillSelectorConfig,
    _auto_select_mla_prefill_backend,
    get_mla_prefill_backend,
    is_deepseek_r1_mla_compatible,
)
from vllm.v1.attention.backends.mla.prefill.tokenspeed_mla import (
    TokenspeedMLAPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import (
    TrtllmRaggedPrefillBackend,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _auto_select_mla_prefill_backend.cache_clear()


def _make_mock_model_config(
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> ModelConfig:
    mock_config = MagicMock(spec=ModelConfig)
    mock_config.dtype = dtype
    mock_config.hf_text_config = MagicMock()
    mock_config.hf_text_config.qk_nope_head_dim = qk_nope_head_dim
    mock_config.hf_text_config.qk_rope_head_dim = qk_rope_head_dim
    mock_config.hf_text_config.v_head_dim = v_head_dim
    return mock_config


def _make_vllm_config(
    model_config: ModelConfig | None = None,
    mla_prefill_backend: MLAPrefillBackendEnum | None = None,
) -> VllmConfig:
    if model_config is None:
        model_config = _make_mock_model_config()

    attention_config = AttentionConfig(mla_prefill_backend=mla_prefill_backend)
    mock_vllm_config = MagicMock(spec=VllmConfig)
    mock_vllm_config.model_config = model_config
    mock_vllm_config.attention_config = attention_config
    mock_vllm_config.cache_config = types.SimpleNamespace(
        cache_dtype="auto",
        calculate_kv_scales=False,
    )
    return mock_vllm_config


class _ConcreteMLACommonImpl(MLACommonImpl):
    def forward_mqa(self, *args, **kwargs):
        raise NotImplementedError


class _RecordingPrefillBackend:
    def __init__(self):
        self.calls = []

    def run_prefill_new_tokens(
        self,
        q,
        k,
        v,
        return_softmax_lse,
        q_scale=None,
        k_scale=None,
        v_scale=None,
    ):
        self.calls.append(
            {
                "kind": "new",
                "q": q,
                "k": k,
                "v": v,
                "q_scale": q_scale,
                "k_scale": k_scale,
                "v_scale": v_scale,
            }
        )
        return torch.ones(q.shape[0], q.shape[1], v.shape[-1], dtype=torch.bfloat16)

    def run_prefill_context_chunk(
        self,
        chunk_idx,
        q,
        k,
        v,
        q_scale=None,
        k_scale=None,
        v_scale=None,
    ):
        self.calls.append(
            {
                "kind": "context",
                "chunk_idx": chunk_idx,
                "q": q,
                "k": k,
                "v": v,
                "q_scale": q_scale,
                "k_scale": k_scale,
                "v_scale": v_scale,
            }
        )
        return (
            torch.ones(q.shape[0], q.shape[1], v.shape[-1], dtype=torch.bfloat16),
            torch.ones(q.shape[1], q.shape[0], dtype=torch.float32),
        )


class TestGetMLAPrefillBackend:
    """Tests for get_mla_prefill_backend (public API)."""

    def test_no_device_capability_returns_flash_attn(self):
        vllm_config = _make_vllm_config()

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = None

            backend = get_mla_prefill_backend(vllm_config)
            assert backend.get_name() == "FLASH_ATTN"

    def test_explicit_flash_attn_selection(self):
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")
            return

        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASH_ATTN,
        )

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                assert backend.get_name() == "FLASH_ATTN"

    def test_explicit_backend_invalid_raises_error(self):
        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASHINFER,
        )

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with pytest.raises(ValueError, match="is not valid"):
                get_mla_prefill_backend(vllm_config)

    def test_explicit_backend_import_error_raises(self):
        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.TRTLLM_RAGGED,
        )

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=10, minor=0
            )

            with (
                patch.object(
                    MLAPrefillBackendEnum.TRTLLM_RAGGED,
                    "get_class",
                    side_effect=ImportError("trtllm not installed"),
                ),
                pytest.raises(ValueError, match="is not valid"),
            ):
                get_mla_prefill_backend(vllm_config)

    def test_auto_selection_on_hopper(self):
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")
            return

        vllm_config = _make_vllm_config()

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                assert backend.get_name() == "FLASH_ATTN"


class TestPrefillQueryQuantization:
    @staticmethod
    def _mock_backend(name: str, supports_prefill_query_quantization: bool):
        backend = MagicMock()
        backend.get_name.return_value = name
        backend.supports_prefill_query_quantization.return_value = (
            supports_prefill_query_quantization
        )
        return backend

    @pytest.mark.parametrize(
        ("backend_name", "expected"),
        [
            ("TRTLLM_RAGGED", True),
            ("FLASHINFER", True),
            ("TOKENSPEED_MLA", True),
            ("FLASH_ATTN", False),
        ],
    )
    def test_backend_supports_scaled_fp8_prefill_backends(self, backend_name, expected):
        with (
            set_current_vllm_config(_make_vllm_config()),
            patch.object(
                mla_attention_module.current_platform,
                "is_device_capability_family",
                return_value=True,
            ),
            patch(
                "vllm.v1.attention.backends.mla.prefill.get_mla_prefill_backend",
                return_value=self._mock_backend(backend_name, expected),
            ),
        ):
            assert backend_supports_prefill_query_quantization() is expected

    def test_backend_support_requires_sm100(self):
        with patch.object(
            mla_attention_module.current_platform,
            "is_device_capability_family",
            return_value=False,
        ):
            assert backend_supports_prefill_query_quantization() is False

    def test_backend_support_uses_current_config_each_call(self):
        first_config = _make_vllm_config()
        second_config = _make_vllm_config()

        with (
            patch.object(
                mla_attention_module.current_platform,
                "is_device_capability_family",
                return_value=True,
            ),
            patch(
                "vllm.v1.attention.backends.mla.prefill.get_mla_prefill_backend",
                side_effect=[
                    self._mock_backend("FLASH_ATTN", False),
                    self._mock_backend("TRTLLM_RAGGED", True),
                ],
            ) as get_backend,
        ):
            assert backend_supports_prefill_query_quantization(first_config) is False
            assert backend_supports_prefill_query_quantization(second_config) is True

        assert get_backend.call_count == 2

    @pytest.mark.parametrize(
        ("cache_dtype", "backend_support", "expected_dtype"),
        [
            ("auto", True, torch.bfloat16),
            ("fp8_e4m3", True, torch.float8_e4m3fn),
        ],
    )
    def test_prefill_query_quantization_requires_fp8_cache_and_backend(
        self,
        cache_dtype,
        backend_support,
        expected_dtype,
    ):
        vllm_config = _make_vllm_config()
        vllm_config.cache_config = types.SimpleNamespace(
            cache_dtype=cache_dtype,
            calculate_kv_scales=False,
        )
        vllm_config.attention_config.use_prefill_query_quantization = True

        with (
            patch(
                "vllm.model_executor.layers.attention.mla_attention."
                "backend_supports_prefill_query_quantization",
                return_value=backend_support,
            ),
            patch(
                "vllm.model_executor.layers.attention.mla_attention."
                "current_platform.fp8_dtype",
                return_value=torch.float8_e4m3fn,
            ),
        ):
            assert (
                MLACommonMetadataBuilder.determine_prefill_query_data_type(
                    vllm_config,
                    torch.bfloat16,
                )
                == expected_dtype
            )

    def test_prefill_query_quantization_rejects_unsupported_fp8_backend(self):
        vllm_config = _make_vllm_config()
        vllm_config.cache_config = types.SimpleNamespace(
            cache_dtype="fp8_e4m3",
            calculate_kv_scales=False,
        )
        vllm_config.attention_config.use_prefill_query_quantization = True

        with (
            patch(
                "vllm.model_executor.layers.attention.mla_attention."
                "backend_supports_prefill_query_quantization",
                return_value=False,
            ),
            pytest.raises(ValueError, match="requires GB200/SM100"),
        ):
            MLACommonMetadataBuilder.determine_prefill_query_data_type(
                vllm_config,
                torch.bfloat16,
            )

    def test_prefill_query_quantization_rejects_dynamic_kv_scales(self):
        vllm_config = _make_vllm_config()
        vllm_config.cache_config = types.SimpleNamespace(
            cache_dtype="fp8_e4m3",
            calculate_kv_scales=True,
        )
        vllm_config.attention_config.use_prefill_query_quantization = True

        with (
            patch(
                "vllm.model_executor.layers.attention.mla_attention."
                "backend_supports_prefill_query_quantization",
                return_value=True,
            ),
            pytest.raises(ValueError, match="requires static calibrated"),
        ):
            MLACommonMetadataBuilder.determine_prefill_query_data_type(
                vllm_config,
                torch.bfloat16,
            )

    def test_trtllm_ragged_bmm_scales_include_fp8_input_descales(self):
        backend = object.__new__(TrtllmRaggedPrefillBackend)
        backend.scale = 0.125

        assert backend._get_bmm_scales(None, None, None) == (0.125, 1.0)
        assert backend._get_bmm_scales(2.0, 3.0, 5.0) == (0.75, 5.0)

        with pytest.raises(ValueError, match="requires q, k, and v scales"):
            backend._get_bmm_scales(2.0, None, 5.0)

    def test_tokenspeed_bmm_scales_include_fp8_input_descales(self):
        backend = object.__new__(TokenspeedMLAPrefillBackend)
        backend.scale = 0.125

        assert backend._get_bmm_scales(None, None, None) == (0.125, 1.0)
        assert backend._get_bmm_scales(2.0, 3.0, 5.0) == (0.75, 5.0)

        with pytest.raises(ValueError, match="requires q, k, and v scales"):
            backend._get_bmm_scales(2.0, None, 5.0)

    def test_scaled_fp8_prefill_input_uses_supplied_scale(self):
        layer = object.__new__(MLAAttention)
        x = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4).transpose(0, 1)
        scale = torch.tensor(0.25)
        captured = {}

        def fake_quant_fp8(x_2d, scale_arg):
            captured["shape"] = x_2d.shape
            captured["is_contiguous"] = x_2d.is_contiguous()
            captured["scale"] = scale_arg
            return torch.ones_like(x_2d), scale_arg

        layer._quant_fp8_op = fake_quant_fp8

        out = MLAAttention._scaled_fp8_prefill_input(layer, x, scale)

        assert out.shape == x.shape
        assert captured == {
            "shape": torch.Size((6, 4)),
            "is_contiguous": True,
            "scale": scale,
        }

    def test_scaled_fp8_prefill_input_uses_scalar_descale(self):
        layer = object.__new__(MLAAttention)
        x = torch.ones(2, 3, 4, dtype=torch.float32)
        scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
        captured = {}

        def fake_quant_fp8(x_2d, scale_arg):
            captured["scale"] = scale_arg
            return torch.ones_like(x_2d), scale_arg

        layer._quant_fp8_op = fake_quant_fp8

        out = MLAAttention._scaled_fp8_prefill_input(layer, x, scale, scale_float=0.5)

        assert out.shape == x.shape
        assert captured["scale"].shape == torch.Size([])
        assert captured["scale"].item() == 0.5

    def test_scaled_fp8_prefill_input_rejects_vector_scale_without_descale(self):
        layer = object.__new__(MLAAttention)
        x = torch.ones(2, 3, 4, dtype=torch.float32)

        with pytest.raises(ValueError, match="requires scalar"):
            MLAAttention._scaled_fp8_prefill_input(layer, x, torch.tensor([0.25, 0.5]))

    def test_scaled_fp8_prefill_descales_match_reference_attention(self):
        layer = object.__new__(MLAAttention)

        def fake_quant_fp8(x_2d, scale_arg):
            return x_2d / scale_arg, scale_arg

        layer._quant_fp8_op = fake_quant_fp8
        q = torch.tensor([[[1.0, -2.0]]])
        k = torch.tensor([[[0.5, 1.0], [2.0, -1.0]]])
        v = torch.tensor([[[3.0, -1.0], [0.5, 2.0]]])
        attn_scale = 0.5
        q_scale = torch.tensor(2.0)
        k_scale = torch.tensor(4.0)
        v_scale = torch.tensor(8.0)

        q_fp8 = MLAAttention._scaled_fp8_prefill_input(layer, q, q_scale)
        k_fp8 = MLAAttention._scaled_fp8_prefill_input(layer, k, k_scale)
        v_fp8 = MLAAttention._scaled_fp8_prefill_input(layer, v, v_scale)

        reference = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * attn_scale,
            dim=-1,
        ).matmul(v)
        scaled_fp8_output = (
            torch.softmax(
                torch.matmul(q_fp8, k_fp8.transpose(-1, -2))
                * (attn_scale * q_scale.item() * k_scale.item()),
                dim=-1,
            ).matmul(v_fp8)
            * v_scale.item()
        )
        unscaled_fp8_output = torch.softmax(
            torch.matmul(q_fp8, k_fp8.transpose(-1, -2)) * attn_scale,
            dim=-1,
        ).matmul(v_fp8)

        torch.testing.assert_close(scaled_fp8_output, reference)
        assert not torch.allclose(unscaled_fp8_output, reference, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("q_data_type", [torch.bfloat16, torch.float8_e4m3fn])
    def test_dcp_chunked_context_rejects_fp8_kv_cache(self, q_data_type):
        impl = object.__new__(_ConcreteMLACommonImpl)
        impl.dcp_world_size = 2
        impl.kv_cache_dtype = "fp8_e4m3"
        impl.num_heads = 1
        impl.qk_nope_head_dim = 2
        impl.qk_rope_head_dim = 1
        impl.v_head_dim = 2
        impl._use_flashinfer_concat_mla_k = False
        impl.kv_b_proj = MagicMock(return_value=(torch.ones(1, 4), None))

        layer = types.SimpleNamespace(
            _q_scale=torch.tensor(1.0),
            _k_scale=torch.tensor(1.0),
            _v_scale=torch.tensor(1.0),
            _q_scale_float=1.0,
            _k_scale_float=1.0,
            _v_scale_float=1.0,
            _scaled_fp8_prefill_input=lambda x, scale, scale_float=None: x,
        )
        prefill_backend = MagicMock()
        prefill_backend.run_prefill_new_tokens.return_value = (
            torch.ones(1, 1, 2),
            torch.ones(1, 1),
        )
        attn_metadata = types.SimpleNamespace(
            prefill=types.SimpleNamespace(
                q_data_type=q_data_type,
                prefill_backend=prefill_backend,
                chunked_context=types.SimpleNamespace(),
            )
        )

        with (
            patch(
                "vllm.model_executor.layers.attention.mla_attention."
                "current_platform.fp8_dtype",
                return_value=torch.float8_e4m3fn,
            ),
            pytest.raises(NotImplementedError, match="DCP MLA chunked-context"),
        ):
            impl.forward_mha(
                q=torch.ones(1, 1, 3),
                kv_c_normed=torch.ones(1, 1),
                k_pe=torch.ones(1, 1, 1),
                kv_c_and_k_pe_cache=torch.empty(0),
                attn_metadata=attn_metadata,
                k_scale=torch.tensor(1.0),
                output=torch.empty(1, 2),
                layer=layer,
            )
        prefill_backend.run_prefill_new_tokens.assert_not_called()
        impl.kv_b_proj.assert_not_called()

    def test_scaled_fp8_prefill_passes_descales_to_backend(self):
        impl = object.__new__(_ConcreteMLACommonImpl)
        impl.dcp_world_size = 1
        impl.num_heads = 1
        impl.qk_nope_head_dim = 2
        impl.qk_rope_head_dim = 1
        impl.v_head_dim = 2
        impl.kv_lora_rank = 2
        impl._use_flashinfer_concat_mla_k = False
        impl.kv_cache_dtype = "fp8_e4m3"

        class QuantizedKvBProj:
            params_dtype = torch.bfloat16

            def __init__(self):
                self.input_dtype = None

            def __call__(self, x):
                self.input_dtype = x.dtype
                return torch.ones(1, 4, dtype=torch.bfloat16), None

        kv_b_proj = QuantizedKvBProj()
        impl.kv_b_proj = kv_b_proj

        q_scale = torch.tensor(2.0)
        k_scale = torch.tensor(3.0)
        v_scale = torch.tensor(5.0)
        quant_calls = []

        def fake_scaled_fp8_prefill_input(x, scale, scale_float=None):
            quant_calls.append((scale, scale_float))
            return x.to(torch.float8_e4m3fn)

        layer = types.SimpleNamespace(
            _q_scale=q_scale,
            _k_scale=k_scale,
            _v_scale=v_scale,
            _q_scale_float=2.0,
            _k_scale_float=3.0,
            _v_scale_float=5.0,
            _scaled_fp8_prefill_input=fake_scaled_fp8_prefill_input,
        )
        backend = _RecordingPrefillBackend()
        prefill = types.SimpleNamespace(
            q_data_type=torch.float8_e4m3fn,
            prefill_backend=backend,
            chunked_context=None,
        )
        attn_metadata = types.SimpleNamespace(prefill=prefill)

        with patch(
            "vllm.model_executor.layers.attention.mla_attention."
            "current_platform.fp8_dtype",
            return_value=torch.float8_e4m3fn,
        ):
            impl.forward_mha(
                q=torch.ones(1, 1, 3, dtype=torch.bfloat16),
                kv_c_normed=torch.ones(1, 2, dtype=torch.bfloat16),
                k_pe=torch.ones(1, 1, 1, dtype=torch.bfloat16),
                kv_c_and_k_pe_cache=torch.empty(0),
                attn_metadata=attn_metadata,
                k_scale=k_scale,
                output=torch.empty(1, 2, dtype=torch.bfloat16),
                layer=layer,
            )

        call = backend.calls[-1]
        assert call["kind"] == "new"
        assert call["q"].dtype == torch.float8_e4m3fn
        assert call["k"].dtype == torch.float8_e4m3fn
        assert call["v"].dtype == torch.float8_e4m3fn
        assert call["q_scale"] == 2.0
        assert call["k_scale"] == 3.0
        assert call["v_scale"] == 5.0
        assert kv_b_proj.input_dtype == torch.bfloat16
        assert [scale.item() for scale, _ in quant_calls] == [2.0, 3.0, 5.0]
        assert [scale_float for _, scale_float in quant_calls] == [2.0, 3.0, 5.0]

    def test_scaled_fp8_context_prefill_passes_descales_to_backend(self, monkeypatch):
        impl = object.__new__(_ConcreteMLACommonImpl)
        impl.num_heads = 1
        impl.qk_nope_head_dim = 2
        impl.qk_rope_head_dim = 1
        impl.v_head_dim = 2
        impl.kv_lora_rank = 2
        impl._use_flashinfer_concat_mla_k = False
        impl.kv_cache_dtype = "fp8_e4m3"
        kv_b_proj = MagicMock(
            return_value=(torch.ones(1, 4, dtype=torch.bfloat16), None)
        )
        kv_b_proj.weight = torch.empty(0, dtype=torch.bfloat16)
        impl.kv_b_proj = kv_b_proj

        backend = _RecordingPrefillBackend()
        workspace = torch.zeros(1, 3, dtype=torch.bfloat16)
        chunked_context = types.SimpleNamespace(
            seq_tot=[1],
            workspace=workspace,
            cu_seq_lens=[torch.tensor([0, 1], dtype=torch.int32)],
            token_to_seq=[torch.tensor([0], dtype=torch.int32)],
            chunk_total_token=[1],
            starts=[torch.tensor([0], dtype=torch.int32)],
        )
        prefill = types.SimpleNamespace(
            q_data_type=torch.float8_e4m3fn,
            prefill_backend=backend,
            chunked_context=chunked_context,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        attn_metadata = types.SimpleNamespace(prefill=prefill)
        layer = types.SimpleNamespace(
            _q_scale=torch.tensor(2.0),
            _k_scale=torch.tensor(3.0),
            _v_scale=torch.tensor(5.0),
            _q_scale_float=2.0,
            _k_scale_float=3.0,
            _v_scale_float=5.0,
            _scaled_fp8_prefill_input=lambda x, scale, scale_float=None: x.to(
                torch.float8_e4m3fn
            ),
        )

        def fake_gather_and_maybe_dequant_cache(**kwargs):
            kwargs["dst"][:1].copy_(torch.tensor([[1.0, 2.0, 3.0]]))

        monkeypatch.setattr(
            mla_attention_module.ops,
            "gather_and_maybe_dequant_cache",
            fake_gather_and_maybe_dequant_cache,
        )

        with patch(
            "vllm.model_executor.layers.attention.mla_attention."
            "current_platform.fp8_dtype",
            return_value=torch.float8_e4m3fn,
        ):
            q = torch.ones(1, 1, 3, dtype=torch.float8_e4m3fn)
            impl._compute_prefill_context(
                q,
                torch.empty(0),
                attn_metadata,
                torch.tensor(3.0),
                layer,
            )

        call = backend.calls[-1]
        assert call["kind"] == "context"
        assert call["q"].dtype == torch.float8_e4m3fn
        assert call["k"].dtype == torch.float8_e4m3fn
        assert call["v"].dtype == torch.float8_e4m3fn
        assert call["q_scale"] == 2.0
        assert call["k_scale"] == 3.0
        assert call["v_scale"] == 5.0

    def test_rocm_aiter_forward_mha_accepts_layer_argument(self):
        try:
            from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
                AiterMLAImpl,
            )
        except ImportError as exc:
            pytest.skip(f"ROCm AITER MLA backend unavailable: {exc}")

        assert "layer" in signature(AiterMLAImpl.forward_mha).parameters

    def test_flashinfer_prefill_backend_forwards_fp8_input_descales(self):
        backend = object.__new__(FlashInferPrefillBackend)
        q = k = v = torch.empty(1)
        backend._prefill_main = MagicMock()
        backend._prefill_main.run.return_value = torch.empty(1)
        backend._prefill_chunks = [MagicMock()]
        backend._prefill_chunks[0].run.return_value = (
            torch.empty(1, 1),
            torch.empty(1, 1),
        )

        with patch.object(
            FlashInferPrefillBackend,
            "supports_prefill_query_quantization",
            return_value=True,
        ):
            backend.run_prefill_new_tokens(
                q,
                k,
                v,
                return_softmax_lse=False,
                q_scale=2.0,
                k_scale=3.0,
                v_scale=5.0,
            )
            backend.run_prefill_context_chunk(
                0,
                q,
                k,
                v,
                q_scale=7.0,
                k_scale=11.0,
                v_scale=13.0,
            )
        assert backend._prefill_main.run.call_args.kwargs["q_scale"] == 2.0
        assert backend._prefill_main.run.call_args.kwargs["k_scale"] == 3.0
        assert backend._prefill_main.run.call_args.kwargs["v_scale"] == 5.0
        assert backend._prefill_chunks[0].run.call_args.kwargs["q_scale"] == 7.0
        assert backend._prefill_chunks[0].run.call_args.kwargs["k_scale"] == 11.0
        assert backend._prefill_chunks[0].run.call_args.kwargs["v_scale"] == 13.0

    def test_flashinfer_prefill_backend_omits_empty_scale_kwargs(self):
        backend = object.__new__(FlashInferPrefillBackend)
        q = k = v = torch.empty(1)
        backend._prefill_main = MagicMock()
        backend._prefill_main.run.return_value = torch.empty(1)

        backend.run_prefill_new_tokens(
            q,
            k,
            v,
            return_softmax_lse=False,
        )

        assert "q_scale" not in backend._prefill_main.run.call_args.kwargs
        assert "k_scale" not in backend._prefill_main.run.call_args.kwargs
        assert "v_scale" not in backend._prefill_main.run.call_args.kwargs

    def test_flashinfer_prefill_backend_rejects_missing_scale_support(self):
        backend = object.__new__(FlashInferPrefillBackend)
        backend._prefill_main = MagicMock()
        q = k = v = torch.empty(1)

        with (
            patch.object(
                FlashInferPrefillBackend,
                "supports_prefill_query_quantization",
                return_value=False,
            ),
            pytest.raises(NotImplementedError, match="does not support scaled FP8"),
        ):
            backend.run_prefill_new_tokens(
                q,
                k,
                v,
                return_softmax_lse=False,
                q_scale=2.0,
                k_scale=3.0,
                v_scale=5.0,
            )
        backend._prefill_main.run.assert_not_called()

    def test_flashinfer_prefill_backend_validation_rejects_missing_scale_support(
        self,
    ):
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=True,
            use_prefill_query_quantization=True,
        )

        with (
            patch.object(FlashInferPrefillBackend, "is_available", return_value=True),
            patch.object(
                FlashInferPrefillBackend,
                "supports_prefill_query_quantization",
                return_value=False,
            ),
        ):
            reasons = FlashInferPrefillBackend.validate_configuration(
                DeviceCapability(major=10, minor=0),
                selector_config,
            )

        assert "backend does not support prefill query quantization" in reasons

    @pytest.mark.parametrize(
        ("signature_kind", "accepts_scales"),
        [
            ("explicit", True),
            ("missing", False),
            ("kwargs", False),
        ],
    )
    def test_flashinfer_reports_installed_scale_support(
        self, monkeypatch, signature_kind, accepts_scales
    ):
        if signature_kind == "explicit":

            class Wrapper:
                def run(self, q, k, v, q_scale=None, k_scale=None, v_scale=None):
                    pass

        elif signature_kind == "missing":

            class Wrapper:
                def run(self, q, k, v):
                    pass

        else:

            class Wrapper:
                def run(self, q, k, v, **kwargs):
                    pass

        monkeypatch.setattr(
            flashinfer_prefill_module,
            "BatchPrefillWithRaggedKVCacheWrapper",
            Wrapper,
        )
        flashinfer_prefill_module._flashinfer_ragged_run_accepts_scales.cache_clear()

        try:
            assert (
                FlashInferPrefillBackend.supports_prefill_query_quantization()
                is accepts_scales
            )
        finally:
            flashinfer_prefill_module._flashinfer_ragged_run_accepts_scales.cache_clear()

    def test_tokenspeed_prefill_backend_applies_equivalent_bmm_scales(
        self, monkeypatch
    ):
        calls = []
        fake_tokenspeed = types.ModuleType("tokenspeed_mla")

        def fake_tokenspeed_mla_prefill(**kwargs):
            calls.append(kwargs)
            out = torch.ones(2, 1, 4)
            if kwargs["return_lse"]:
                return out, torch.full((2, 1), 7.0)
            return out

        fake_tokenspeed.tokenspeed_mla_prefill = fake_tokenspeed_mla_prefill
        monkeypatch.setitem(sys.modules, "tokenspeed_mla", fake_tokenspeed)

        backend = object.__new__(TokenspeedMLAPrefillBackend)
        backend.scale = 0.125
        backend._query_seq_lens = torch.tensor([2], dtype=torch.int32)
        backend._prefill_metadata = types.SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            max_query_len=2,
            chunked_context=types.SimpleNamespace(
                seq_lens=[torch.tensor([4], dtype=torch.int32)],
                cu_seq_lens=[torch.tensor([0, 4], dtype=torch.int32)],
                max_seq_lens=[4],
            ),
        )
        q = k = v = torch.ones(2, 1, 4)

        out = backend.run_prefill_new_tokens(
            q,
            k,
            v,
            return_softmax_lse=False,
            q_scale=2.0,
            k_scale=3.0,
            v_scale=5.0,
        )
        assert torch.equal(out, torch.full_like(out, 5.0))
        assert calls[-1]["softmax_scale"] == 0.75

        out, lse = backend.run_prefill_context_chunk(
            0,
            q,
            k,
            v,
            q_scale=2.0,
            k_scale=3.0,
            v_scale=5.0,
        )
        assert torch.equal(out, torch.full_like(out, 5.0))
        assert torch.equal(lse, torch.full((1, 2), 7.0))
        assert calls[-1]["softmax_scale"] == 0.75

    def test_flash_attn_prefill_backend_rejects_scaled_fp8_inputs(self):
        backend = object.__new__(FlashAttnPrefillBackend)
        q = k = v = torch.empty(1)

        with pytest.raises(NotImplementedError, match="does not support scaled FP8"):
            backend.run_prefill_new_tokens(
                q,
                k,
                v,
                return_softmax_lse=False,
                q_scale=2.0,
                k_scale=3.0,
                v_scale=5.0,
            )
        with pytest.raises(NotImplementedError, match="does not support scaled FP8"):
            backend.run_prefill_context_chunk(
                0,
                q,
                k,
                v,
                q_scale=2.0,
                k_scale=3.0,
                v_scale=5.0,
            )


class TestAutoSelectMLAPrefillBackend:
    """Tests for fallback and error paths in auto-selection."""

    def test_blackwell_query_quantization_skips_flash_attn(self):
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=True,
            use_prefill_query_quantization=True,
        )

        try:
            trtllm_cls = MLAPrefillBackendEnum.TRTLLM_RAGGED.get_class()
        except ImportError:
            pytest.skip("TRTLLM_RAGGED backend not available")
            return

        with (
            patch.object(
                MLAPrefillBackendEnum.FLASH_ATTN,
                "get_class",
                return_value=FlashAttnPrefillBackend,
            ),
            patch.object(trtllm_cls, "is_available", return_value=True),
        ):
            backend = _auto_select_mla_prefill_backend(
                capability,
                selector_config,
            )

        assert backend.get_name() == "TRTLLM_RAGGED"

    def test_blackwell_falls_back_to_trtllm(self):
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
        )

        try:
            trtllm_cls = MLAPrefillBackendEnum.TRTLLM_RAGGED.get_class()
        except ImportError:
            pytest.skip("TRTLLM_RAGGED backend not available")
            return

        with (
            patch.object(
                MLAPrefillBackendEnum.FLASH_ATTN,
                "get_class",
                side_effect=ImportError("FLASH_ATTN not available"),
            ),
            patch.object(trtllm_cls, "validate_configuration", return_value=[]),
        ):
            backend = _auto_select_mla_prefill_backend(
                capability,
                selector_config,
            )
            assert backend.get_name() == "TRTLLM_RAGGED"

    def test_all_fail_raises_error(self):
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
        )

        def mock_get_class(backend_enum):  # noqa: ARG001
            cls = MagicMock()
            cls.validate_configuration.return_value = ["not available"]
            return cls

        with patch.object(MLAPrefillBackendEnum, "get_class", mock_get_class):
            _auto_select_mla_prefill_backend.cache_clear()
            with pytest.raises(ValueError, match="No valid MLA"):
                _auto_select_mla_prefill_backend(
                    capability,
                    selector_config,
                )


class TestBackendValidation:
    """Tests for backend validation logic."""

    def test_r1_dimension_requirement(self):
        try:
            from vllm.v1.attention.backends.mla.prefill.flashinfer import (
                FlashInferPrefillBackend,
            )
        except ImportError:
            pytest.skip("FlashInfer prefill backend not available")
            return

        assert FlashInferPrefillBackend.requires_r1_mla_dimensions is True

        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                capability,
                selector_config,
            )
            assert len(invalid_reasons) == 0

        vllm_config_invalid = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        selector_config_invalid = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config_invalid),
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                capability,
                selector_config_invalid,
            )
            assert len(invalid_reasons) == 1
            assert "DeepSeek R1 MLA dimensions" in invalid_reasons[0]


class TestMLAPrefillBackendParsing:
    """Tests for string-based mla_prefill_backend parsing from CLI args."""

    def test_valid_string_parses_to_enum(self):
        config = AttentionConfig(
            mla_prefill_backend="FLASH_ATTN",  # type: ignore[arg-type]
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.FLASH_ATTN

    def test_invalid_string_raises_error(self):
        with pytest.raises(ValueError, match="Unknown MLA prefill backend"):
            AttentionConfig(
                mla_prefill_backend="NONEXISTENT",  # type: ignore[arg-type]
            )


class TestMLAPrefillBackendConfig:
    """Tests for mla_prefill_backend configuration in AttentionConfig."""

    def test_default_backend_is_none(self):
        config = AttentionConfig()
        assert config.mla_prefill_backend is None

    def test_explicit_flash_attn_backend(self):
        config = AttentionConfig(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASH_ATTN,
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.FLASH_ATTN

    def test_explicit_trtllm_ragged_backend(self):
        config = AttentionConfig(
            mla_prefill_backend=MLAPrefillBackendEnum.TRTLLM_RAGGED,
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.TRTLLM_RAGGED
