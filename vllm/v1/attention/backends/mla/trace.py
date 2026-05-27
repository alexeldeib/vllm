# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

_TRACE_COUNT = 0


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _device(value: Any) -> str | None:
    device = getattr(value, "device", None)
    if device is None:
        return None
    return str(device)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cuda_is_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _trace_path() -> str | None:
    return os.environ.get("VLLM_K26_MLA_TRACE_DIR")


def _trace_limit() -> int:
    return int(os.environ.get("VLLM_K26_MLA_TRACE_LIMIT", "4096"))


def _trace_sync() -> bool:
    return os.environ.get("VLLM_K26_MLA_TRACE_SYNC", "0") == "1"


def _forward_summary() -> dict[str, Any]:
    if not is_forward_context_available():
        return {"trace_label": "no_forward_context"}

    context = get_forward_context()
    batch_descriptor = context.batch_descriptor
    return {
        "trace_label": context.additional_kwargs.get("trace_label", "unlabeled"),
        "cudagraph_runtime_mode": str(context.cudagraph_runtime_mode),
        "batch_num_tokens": (
            None if batch_descriptor is None else batch_descriptor.num_tokens
        ),
        "batch_num_reqs": None if batch_descriptor is None else batch_descriptor.num_reqs,
        "batch_uniform": None if batch_descriptor is None else batch_descriptor.uniform,
        "skip_compiled": context.skip_compiled,
    }


def _metadata_summary(attn_metadata: Any) -> dict[str, Any]:
    decode = getattr(attn_metadata, "decode", None)
    num_decodes = _safe_int(getattr(attn_metadata, "num_decodes", None))
    num_decode_tokens = _safe_int(getattr(attn_metadata, "num_decode_tokens", None))
    q_len_per_decode = None
    if num_decodes:
        q_len_per_decode = num_decode_tokens // num_decodes

    return {
        "num_reqs": _safe_int(getattr(attn_metadata, "num_reqs", None)),
        "max_query_len": _safe_int(getattr(attn_metadata, "max_query_len", None)),
        "max_seq_len": _safe_int(getattr(attn_metadata, "max_seq_len", None)),
        "num_actual_tokens": _safe_int(
            getattr(attn_metadata, "num_actual_tokens", None)
        ),
        "num_decodes": num_decodes,
        "num_decode_tokens": num_decode_tokens,
        "q_len_per_decode": q_len_per_decode,
        "num_prefills": _safe_int(getattr(attn_metadata, "num_prefills", None)),
        "decode_block_table_shape": (
            None if decode is None else _shape(getattr(decode, "block_table", None))
        ),
        "decode_seq_lens_shape": (
            None if decode is None else _shape(getattr(decode, "seq_lens", None))
        ),
        "decode_seq_lens_device": (
            None if decode is None else _device(getattr(decode, "seq_lens", None))
        ),
    }


def _write_trace(trace_dir: str, payload: dict[str, Any]) -> None:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"mla_trace_rank{rank}.jsonl")
    with open(path, "a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(payload, sort_keys=True) + "\n")


@contextmanager
def maybe_trace_mla_forward(
    backend: str,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    layer: Any,
):
    trace_dir = _trace_path()
    if not trace_dir:
        with nullcontext():
            yield
        return

    global _TRACE_COUNT
    if _TRACE_COUNT >= _trace_limit():
        with nullcontext():
            yield
        return

    _TRACE_COUNT += 1
    capturing = _cuda_is_capturing()
    use_events = _trace_sync() and torch.cuda.is_available() and not capturing

    payload = {
        "backend": backend,
        "event_index": _TRACE_COUNT,
        "pid": os.getpid(),
        "time_unix_ns": time.time_ns(),
        "cuda_stream_capturing": capturing,
        "q_shape": _shape(q),
        "q_dtype": _dtype(q),
        "q_device": _device(q),
        "kv_shape": _shape(kv_cache),
        "kv_dtype": _dtype(kv_cache),
        "kv_device": _device(kv_cache),
        "layer_type": type(layer).__name__,
    }
    payload.update(_forward_summary())
    payload.update(_metadata_summary(attn_metadata))

    start_event = end_event = None
    if use_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    try:
        yield
    finally:
        if use_events and start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            payload["elapsed_ms"] = float(start_event.elapsed_time(end_event))
        _write_trace(trace_dir, payload)
