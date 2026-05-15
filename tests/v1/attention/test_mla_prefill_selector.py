# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend selector."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention import mla_attention as mla_attention_module
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLACommonMetadataBuilder,
    backend_supports_prefill_query_quantization,
)
from vllm.platforms.interface import DeviceCapability
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
    backend_supports_prefill_query_quantization.cache_clear()


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
    mock_vllm_config.cache_config = MagicMock()
    mock_vllm_config.cache_config.cache_dtype = "auto"
    mock_vllm_config.cache_config.calculate_kv_scales = False
    mock_vllm_config.parallel_config = MagicMock()
    mock_vllm_config.parallel_config.decode_context_parallel_size = 1
    return mock_vllm_config


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
    def _mock_backend(name: str):
        backend = MagicMock()
        backend.get_name.return_value = name
        return backend

    @pytest.mark.parametrize(
        ("backend_name", "expected"),
        [
            ("TRTLLM_RAGGED", True),
            ("FLASHINFER", False),
            ("TOKENSPEED_MLA", False),
            ("FLASH_ATTN", False),
        ],
    )
    def test_backend_supports_only_trtllm_ragged(self, backend_name, expected):
        with (
            set_current_vllm_config(_make_vllm_config()),
            patch.object(
                mla_attention_module.current_platform,
                "is_device_capability_family",
                return_value=True,
            ),
            patch(
                "vllm.v1.attention.backends.mla.prefill.get_mla_prefill_backend",
                return_value=self._mock_backend(backend_name),
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

    @pytest.mark.parametrize(
        ("cache_dtype", "backend_support", "expected_dtype"),
        [
            ("auto", True, torch.bfloat16),
            ("fp8_e4m3", False, torch.bfloat16),
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
        vllm_config.cache_config = MagicMock()
        vllm_config.cache_config.cache_dtype = cache_dtype
        vllm_config.cache_config.calculate_kv_scales = False
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

    @pytest.mark.parametrize(
        "config_attr",
        ["decode_context_parallel_size", "calculate_kv_scales"],
    )
    def test_prefill_query_quantization_disabled_for_unsupported_modes(
        self,
        config_attr,
    ):
        vllm_config = _make_vllm_config()
        vllm_config.cache_config.cache_dtype = "fp8_e4m3"
        vllm_config.attention_config.use_prefill_query_quantization = True
        if config_attr == "decode_context_parallel_size":
            vllm_config.parallel_config.decode_context_parallel_size = 2
        else:
            vllm_config.cache_config.calculate_kv_scales = True

        with patch(
            "vllm.model_executor.layers.attention.mla_attention."
            "backend_supports_prefill_query_quantization",
            return_value=True,
        ):
            assert (
                MLACommonMetadataBuilder.determine_prefill_query_data_type(
                    vllm_config,
                    torch.bfloat16,
                )
                is torch.bfloat16
            )

    def test_trtllm_ragged_bmm_scales_include_fp8_input_descales(self):
        backend = object.__new__(TrtllmRaggedPrefillBackend)
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

    @pytest.mark.parametrize(
        "backend_cls",
        [
            FlashAttnPrefillBackend,
            FlashInferPrefillBackend,
            TokenspeedMLAPrefillBackend,
        ],
    )
    def test_non_trtllm_prefill_backends_reject_scaled_fp8_inputs(self, backend_cls):
        backend = object.__new__(backend_cls)
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


class TestAutoSelectMLAPrefillBackend:
    """Tests for fallback and error paths in auto-selection."""

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
