# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the TRTLLM_RAGGED MLA prefill backend."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import (
    TrtllmRaggedPrefillBackend,
)


def test_context_chunk_empty_kv_rows_do_not_leak_invalid_outputs(monkeypatch):
    calls = []

    def fake_trtllm_ragged_attention_deepseek(**kwargs):
        calls.append(kwargs)
        q = kwargs["query"]
        out = kwargs["out"]
        lse = torch.empty(q.shape[0], q.shape[1], device=q.device)

        out.zero_()
        lse.zero_()
        for row, kv_len in enumerate(kwargs["seq_lens"].tolist()):
            q_start = int(kwargs["cum_seq_lens_q"][row].item())
            q_end = int(kwargs["cum_seq_lens_q"][row + 1].item())
            if kv_len == 0:
                out[q_start:q_end] = torch.nan
                lse[q_start:q_end] = torch.nan
                continue

            for token in range(q_start, q_end):
                out[token] = torch.arange(
                    token * q.shape[1] * q.shape[2],
                    (token + 1) * q.shape[1] * q.shape[2],
                    device=q.device,
                    dtype=out.dtype,
                ).reshape(q.shape[1], q.shape[2])
                lse[token] = torch.arange(
                    token * q.shape[1] + 1,
                    (token + 1) * q.shape[1] + 1,
                    device=q.device,
                    dtype=lse.dtype,
                )
        return out, lse

    prefill_module = ModuleType("flashinfer.prefill")
    prefill_module.trtllm_ragged_attention_deepseek = (  # type: ignore[attr-defined]
        fake_trtllm_ragged_attention_deepseek
    )
    flashinfer_module = ModuleType("flashinfer")
    flashinfer_module.prefill = prefill_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill_module)

    backend = TrtllmRaggedPrefillBackend.__new__(TrtllmRaggedPrefillBackend)
    backend.scale = 1.0
    backend._workspace_buffer = torch.empty(1, dtype=torch.uint8)
    backend._query_seq_lens = torch.tensor([2, 3, 1], dtype=torch.int32)
    backend._query_seq_lens_cpu = torch.tensor([2, 3, 1], dtype=torch.int32)
    backend._prefill_metadata = SimpleNamespace(
        max_query_len=3,
        output_dtype=torch.float32,
        query_start_loc=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
        chunked_context=SimpleNamespace(
            seq_lens=torch.tensor([[4, 0, 2]], dtype=torch.int32),
            max_seq_lens=[4],
            cu_seq_lens=torch.tensor([[0, 4, 4, 6]], dtype=torch.int32),
        ),
    )

    q = torch.ones(6, 2, 3)
    k = torch.ones(6, 1, 3)
    v = torch.ones(6, 1, 3)
    out, lse = backend.run_prefill_context_chunk(0, q, k, v)

    assert len(calls) == 1
    assert calls[0]["batch_size"] == 2
    assert calls[0]["seq_lens"].tolist() == [4, 2]
    assert calls[0]["cum_seq_lens_q"].tolist() == [0, 2, 3]
    assert calls[0]["cum_seq_lens_kv"].tolist() == [0, 4, 6]
    assert out.shape == (6, 2, 3)
    assert torch.isfinite(out).all()
    assert not torch.isnan(lse).any()
    assert torch.equal(
        out[:2],
        torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    )
    assert torch.count_nonzero(out[2:5]) == 0
    assert torch.equal(out[5], torch.arange(12, 18, dtype=torch.float32).reshape(2, 3))
    assert lse.shape == (2, 6)
    assert torch.equal(lse[:, :2], torch.tensor([[1.0, 3.0], [2.0, 4.0]]))
    assert torch.isneginf(lse[:, 2:5]).all()
    assert torch.equal(lse[:, 5], torch.tensor([5.0, 6.0]))
