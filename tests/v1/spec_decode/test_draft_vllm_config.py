# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


@dataclass
class _ParallelConfig:
    tensor_parallel_size: int
    rank: int
    decode_context_parallel_size: int = 1
    prefill_context_parallel_size: int = 1


@dataclass
class _ModelConfig:
    name: str


@dataclass
class _KernelConfig:
    moe_backend: str


@dataclass
class _AttentionConfig:
    backend: object | None


@dataclass
class _CacheConfig:
    cache_dtype: str


@dataclass
class _SpeculativeConfig:
    draft_parallel_config: _ParallelConfig
    draft_model_config: _ModelConfig
    moe_backend: str | None = None
    attention_backend: object | None = None
    draft_kv_cache_dtype: str | None = None


@dataclass
class _VllmConfig:
    parallel_config: _ParallelConfig
    model_config: _ModelConfig
    kernel_config: _KernelConfig
    attention_config: _AttentionConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig


def _create_draft_config(vllm_config: _VllmConfig) -> _VllmConfig:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.vllm_config = vllm_config
    proposer.speculative_config = vllm_config.speculative_config
    return proposer._create_draft_vllm_config()


def test_eagle_draft_config_uses_draft_parallel_config():
    target_parallel_config = _ParallelConfig(
        tensor_parallel_size=4,
        rank=3,
        decode_context_parallel_size=2,
        prefill_context_parallel_size=2,
    )
    draft_parallel_config = _ParallelConfig(tensor_parallel_size=4, rank=0)
    draft_model_config = _ModelConfig(name="draft")
    target_config = _VllmConfig(
        parallel_config=target_parallel_config,
        model_config=_ModelConfig(name="target"),
        kernel_config=_KernelConfig(moe_backend="auto"),
        attention_config=_AttentionConfig(backend="TRITON_MLA"),
        cache_config=_CacheConfig(cache_dtype="fp8_e4m3"),
        speculative_config=_SpeculativeConfig(
            draft_parallel_config=draft_parallel_config,
            draft_model_config=draft_model_config,
        ),
    )

    draft_config = _create_draft_config(target_config)

    assert draft_config.model_config is draft_model_config
    assert draft_config.parallel_config.tensor_parallel_size == 4
    assert draft_config.parallel_config.rank == target_parallel_config.rank
    assert draft_config.parallel_config.decode_context_parallel_size == 1
    assert draft_config.parallel_config.prefill_context_parallel_size == 1


def test_eagle_draft_config_keeps_draft_kernel_overrides():
    target_config = _VllmConfig(
        parallel_config=_ParallelConfig(tensor_parallel_size=1, rank=0),
        model_config=_ModelConfig(name="target"),
        kernel_config=_KernelConfig(moe_backend="flashinfer_trtllm"),
        attention_config=_AttentionConfig(backend="TRITON_MLA"),
        cache_config=_CacheConfig(cache_dtype="fp8_e4m3"),
        speculative_config=_SpeculativeConfig(
            draft_parallel_config=_ParallelConfig(tensor_parallel_size=1, rank=0),
            draft_model_config=_ModelConfig(name="draft"),
            moe_backend="triton",
            attention_backend=None,
        ),
    )

    draft_config = _create_draft_config(target_config)

    assert draft_config.kernel_config.moe_backend == "triton"
    assert draft_config.attention_config.backend is None


def test_eagle_draft_config_uses_draft_kv_cache_dtype_override():
    target_config = _VllmConfig(
        parallel_config=_ParallelConfig(tensor_parallel_size=1, rank=0),
        model_config=_ModelConfig(name="target"),
        kernel_config=_KernelConfig(moe_backend="auto"),
        attention_config=_AttentionConfig(backend=None),
        cache_config=_CacheConfig(cache_dtype="fp8_e4m3"),
        speculative_config=_SpeculativeConfig(
            draft_parallel_config=_ParallelConfig(tensor_parallel_size=1, rank=0),
            draft_model_config=_ModelConfig(name="draft"),
            draft_kv_cache_dtype="bfloat16",
        ),
    )

    draft_config = _create_draft_config(target_config)

    assert target_config.cache_config.cache_dtype == "fp8_e4m3"
    assert draft_config.cache_config.cache_dtype == "bfloat16"
