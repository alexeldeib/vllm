# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.attention.backend import (
    AttentionImplBase,
    resolve_effective_cp_state_for_vllm_config,
)


@dataclass
class _ParallelConfig:
    decode_context_parallel_size: int
    prefill_context_parallel_size: int


@dataclass
class _VllmConfig:
    parallel_config: _ParallelConfig


class _AttentionImpl(AttentionImplBase):
    can_return_lse_for_decode = True


def _impl_with_cp(dcp_size: int, pcp_size: int) -> _AttentionImpl:
    impl = object.__new__(_AttentionImpl)
    impl.dcp_world_size = dcp_size
    impl.dcp_rank = dcp_size - 1
    impl.pcp_world_size = pcp_size
    impl.pcp_rank = pcp_size - 1
    impl.total_cp_world_size = dcp_size * pcp_size
    impl.total_cp_rank = impl.pcp_rank * dcp_size + impl.dcp_rank
    impl.need_to_return_lse_for_decode = dcp_size > 1
    return impl


def test_attention_impl_clears_cp_state_for_draft_config():
    impl = _impl_with_cp(dcp_size=2, pcp_size=2)

    impl.maybe_override_cp_for_vllm_config(
        _VllmConfig(
            parallel_config=_ParallelConfig(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            )
        )
    )

    assert impl.dcp_world_size == 1
    assert impl.dcp_rank == 0
    assert impl.pcp_world_size == 1
    assert impl.pcp_rank == 0
    assert impl.total_cp_world_size == 1
    assert impl.total_cp_rank == 0
    assert not impl.need_to_return_lse_for_decode


def test_attention_impl_keeps_cp_state_for_cp_config():
    impl = _impl_with_cp(dcp_size=2, pcp_size=2)

    impl.maybe_override_cp_for_vllm_config(
        _VllmConfig(
            parallel_config=_ParallelConfig(
                decode_context_parallel_size=2,
                prefill_context_parallel_size=2,
            )
        )
    )

    assert impl.dcp_world_size == 2
    assert impl.dcp_rank == 1
    assert impl.pcp_world_size == 2
    assert impl.pcp_rank == 1
    assert impl.total_cp_world_size == 4
    assert impl.total_cp_rank == 3
    assert impl.need_to_return_lse_for_decode


def test_resolve_effective_cp_state_clears_builder_dcp_for_draft_config():
    (
        dcp_world_size,
        dcp_rank,
        pcp_world_size,
        pcp_rank,
        total_cp_world_size,
        total_cp_rank,
    ) = resolve_effective_cp_state_for_vllm_config(
        dcp_world_size=2,
        dcp_rank=1,
        pcp_world_size=1,
        pcp_rank=0,
        vllm_config=_VllmConfig(
            parallel_config=_ParallelConfig(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            )
        ),
    )

    assert dcp_world_size == 1
    assert dcp_rank == 0
    assert pcp_world_size == 1
    assert pcp_rank == 0
    assert total_cp_world_size == 1
    assert total_cp_rank == 0
