# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("flashinfer")

from vllm.v1.attention.backends.flashinfer import FlashInferImpl  # noqa: E402


def _make_vllm_config(num_qo_heads: int, dcp_size: int = 1):
    model_config = MagicMock()
    model_config.get_num_attention_heads.return_value = num_qo_heads
    attention_config = MagicMock()
    attention_config.disable_flashinfer_q_quantization = False

    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        attention_config=attention_config,
    )


@pytest.mark.parametrize(
    ("metadata_num_qo_heads", "expected_support"),
    [
        (4, True),
        (1, False),
    ],
)
def test_flashinfer_query_quantization_matches_metadata_head_count(
    metadata_num_qo_heads: int,
    expected_support: bool,
):
    vllm_config = _make_vllm_config(metadata_num_qo_heads)

    with (
        patch(
            "vllm.v1.attention.backends.flashinfer.get_current_vllm_config_or_none",
            return_value=vllm_config,
        ),
        patch(
            "vllm.v1.attention.backends.flashinfer.can_use_trtllm_attention",
            side_effect=lambda num_qo_heads, num_kv_heads: (
                num_qo_heads % num_kv_heads == 0
            ),
        ) as can_use_trtllm_attention,
    ):
        impl = FlashInferImpl(
            num_heads=4,
            head_size=128,
            scale=1.0,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_e4m3",
        )

    assert impl.support_trtllm_attn is expected_support
    assert impl.supports_quant_query_input is expected_support
    can_use_trtllm_attention.assert_called_once_with(metadata_num_qo_heads, 2)


def test_flashinfer_query_quantization_disabled_for_dcp():
    vllm_config = _make_vllm_config(num_qo_heads=4, dcp_size=2)

    with (
        patch(
            "vllm.v1.attention.backends.flashinfer.get_current_vllm_config_or_none",
            return_value=vllm_config,
        ),
        patch(
            "vllm.v1.attention.backends.flashinfer.can_use_trtllm_attention",
            return_value=True,
        ) as can_use_trtllm_attention,
    ):
        impl = FlashInferImpl(
            num_heads=4,
            head_size=128,
            scale=1.0,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_e4m3",
        )

    assert impl.support_trtllm_attn is True
    assert impl.supports_quant_query_input is False
    can_use_trtllm_attention.assert_called_once_with(4, 2)
