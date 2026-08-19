# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


def _make_rank_input(shape, dtype, device, rank, iteration):
    values = torch.arange(
        math.prod(shape),
        dtype=torch.float32,
        device=device,
    ).reshape(shape)
    return ((values % 32) + rank * 64 + (iteration % 4) * 256).to(dtype)


def _supports_multimem():
    device_capability = current_platform.get_device_capability()
    return (
        current_platform.is_cuda()
        and device_capability is not None
        and device_capability.major >= 9
    )


@ray.remote(num_gpus=1, max_calls=1)
def mnnvl_lamport_collectives(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        fa = get_tp_group().device_communicator.ca_comm
        assert fa is not None and not fa.disabled
        assert fa.mnnvl_multicast_ptr, (
            "This test must exercise the MNNVL Lamport path, but symmetric "
            "memory did not provide an NVLS multicast mapping."
        )

        shapes = [(1, 128), (7, 128), (64, 512), (257, 128)]
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for iteration in range(64):
            for shape in shapes:
                for dtype in dtypes:
                    inp = _make_rank_input(shape, dtype, device, rank, iteration)
                    expected = torch.cat(
                        [
                            _make_rank_input(shape, dtype, device, src_rank, iteration)
                            for src_rank in range(tp_size)
                        ]
                    )
                    out = fa.custom_all_gather(inp)
                    assert out is not None
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

                    rs_inp = torch.cat(
                        [inp + dst_rank * 8 for dst_rank in range(tp_size)]
                    )
                    expected_rs = sum(
                        _make_rank_input(shape, dtype, device, src_rank, iteration)
                        + rank * 8
                        for src_rank in range(tp_size)
                    )
                    out = fa.custom_reduce_scatter(rs_inp)
                    assert out is not None
                    torch.testing.assert_close(out, expected_rs, rtol=0, atol=0)

        static_inp = _make_rank_input(
            (64, 512), torch.bfloat16, device, rank, iteration=0
        )
        with graph_capture(device=device) as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                graph_out = fa.custom_all_gather(static_inp)
                assert graph_out is not None

        for iteration in range(128):
            static_inp.copy_(
                _make_rank_input(
                    static_inp.shape,
                    static_inp.dtype,
                    device,
                    rank,
                    iteration,
                )
            )
            graph.replay()
            expected = torch.cat(
                [
                    _make_rank_input(
                        static_inp.shape,
                        static_inp.dtype,
                        device,
                        src_rank,
                        iteration,
                    )
                    for src_rank in range(tp_size)
                ]
            )
            torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)


@pytest.mark.skipif(
    not _supports_multimem(),
    reason="MNNVL Lamport collectives require an SM90 or newer NVIDIA GPU.",
)
def test_mnnvl_lamport_collectives(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Need at least two GPUs to run the test.")
    multi_process_parallel(monkeypatch, 2, 1, mnnvl_lamport_collectives)
