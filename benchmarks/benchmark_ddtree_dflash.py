# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Budget-sweep benchmark for DFlash vs DDTree+DFlash.

This is intentionally a narrow diagnostic harness. It runs the same offline
workload through target-only, DFlash, and DDTree+DFlash, optionally enabling the
runner's SPEC_DECODE_DEBUG_METRICS JSONL stream so acceptance lengths and stage
timers can be aggregated without changing normal production behavior.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


PROMPT_SEEDS = [
    "Explain why speculative decoding must preserve exact greedy outputs.",
    "Write a Python function that counts vowels in a string.",
    "Give three practical tips for debugging CUDA kernels.",
    "Summarize how prefix caching changes prefill latency.",
    "Describe the tradeoff between draft accuracy and draft cost.",
    "Write a Python function that reverses words in a sentence.",
    "List three metrics to track during model serving load tests.",
    "Compare online serving traces with offline replay benchmarks.",
    "Explain how paged KV cache allocation affects serving throughput.",
    "Write a compact Python function that clamps a value to a range.",
    "List three signs that a GPU kernel is memory bandwidth bound.",
    "Compare request concurrency and batch size in inference serving.",
    "Explain why batching matters for speculative decoding in two sentences.",
    "Write a short Python function that returns the square of an integer.",
    "Give three ways to reduce warmup noise in GPU benchmarks.",
    "Summarize the difference between latency and throughput.",
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def make_prompts(count: int) -> list[str]:
    prompts = []
    for index, prompt in zip(range(count), itertools.cycle(PROMPT_SEEDS)):
        prompts.append(f"{prompt}\nRequest id: {index}.")
    return prompts


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_lengths: list[int] = []
    draft_tokens: list[int] = []
    generated_lengths: list[int] = []
    timers: dict[str, list[float]] = {}
    tree_nodes: list[int] = []
    tree_depths: list[int] = []
    tree_build: dict[str, list[float]] = {
        "tree_topk_ms": [],
        "tree_transfer_ms": [],
        "tree_cpu_build_ms": [],
    }

    for event in events:
        accepted_lengths.extend(int(v) for v in event.get("accepted_lengths", []))
        draft_tokens.extend(int(v) for v in event.get("draft_tokens", []))
        generated_lengths.extend(int(v) for v in event.get("generated_lengths", []))
        tree_nodes.extend(int(v) for v in event.get("tree_nodes", []))
        tree_depths.extend(int(v) for v in event.get("tree_max_depths", []))

        for name, value in event.get("timers_ms", {}).items():
            timers.setdefault(name, []).append(float(value))

        propose_metrics = event.get("ddtree_propose") or {}
        for name in tree_build:
            if name in propose_metrics:
                tree_build[name].append(float(propose_metrics[name]))

    num_drafts = len(accepted_lengths)
    accepted = sum(accepted_lengths)
    drafted = sum(draft_tokens)
    result: dict[str, Any] = {
        "debug_events": len(events),
        "num_drafts": num_drafts,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "mean_accepted_tokens": accepted / num_drafts if num_drafts else None,
        "mean_acceptance_length": 1 + accepted / num_drafts if num_drafts else None,
        "acceptance_rate": accepted / drafted if drafted else None,
        "accepted_lengths": {
            "min": min(accepted_lengths) if accepted_lengths else None,
            "max": max(accepted_lengths) if accepted_lengths else None,
            "median": statistics.median(accepted_lengths)
            if accepted_lengths
            else None,
        },
        "generated_lengths": {
            "min": min(generated_lengths) if generated_lengths else None,
            "max": max(generated_lengths) if generated_lengths else None,
            "median": statistics.median(generated_lengths)
            if generated_lengths
            else None,
        },
        "timers_ms_mean": {
            name: mean(values) for name, values in sorted(timers.items())
        },
    }
    if tree_nodes:
        result["tree_nodes"] = {
            "min": min(tree_nodes),
            "max": max(tree_nodes),
            "median": statistics.median(tree_nodes),
        }
        result["tree_max_depths"] = {
            "min": min(tree_depths),
            "max": max(tree_depths),
            "median": statistics.median(tree_depths),
        }
    if any(tree_build.values()):
        result["ddtree_propose_ms_mean"] = {
            name: mean(values) for name, values in tree_build.items() if values
        }
    return result


def run_case(
    *,
    name: str,
    common_kwargs: dict[str, Any],
    speculative_config: dict[str, Any] | None,
    prompts: list[str],
    warmup_prompts: list[str],
    sampling: SamplingParams,
    metrics_dir: Path,
    debug_metrics: bool,
) -> dict[str, Any]:
    metrics_path = metrics_dir / f"{name}.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    old_debug = os.environ.get("SPEC_DECODE_DEBUG_METRICS")
    old_metrics_file = os.environ.get("SPEC_DECODE_DEBUG_METRICS_FILE")
    if debug_metrics:
        os.environ["SPEC_DECODE_DEBUG_METRICS"] = "1"
        os.environ["SPEC_DECODE_DEBUG_METRICS_FILE"] = str(metrics_path)
    else:
        os.environ.pop("SPEC_DECODE_DEBUG_METRICS", None)
        os.environ.pop("SPEC_DECODE_DEBUG_METRICS_FILE", None)

    try:
        llm = LLM(**common_kwargs, speculative_config=speculative_config)
        llm.generate(warmup_prompts, sampling, use_tqdm=False)
        if metrics_path.exists():
            metrics_path.unlink()

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start

        output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        result = {
            "name": name,
            "elapsed_s": elapsed,
            "output_tokens": output_tokens,
            "output_tps": output_tokens / elapsed if elapsed else None,
            "metrics": aggregate_metrics(read_metrics(metrics_path)),
        }

        del llm
        gc.collect()
        return result
    finally:
        if old_debug is None:
            os.environ.pop("SPEC_DECODE_DEBUG_METRICS", None)
        else:
            os.environ["SPEC_DECODE_DEBUG_METRICS"] = old_debug
        if old_metrics_file is None:
            os.environ.pop("SPEC_DECODE_DEBUG_METRICS_FILE", None)
        else:
            os.environ["SPEC_DECODE_DEBUG_METRICS_FILE"] = old_metrics_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--budgets", default="16,32,64,128,256")
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--attention-backend", default="TREE_ATTN")
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no-debug-metrics", action="store_true")
    args = parser.parse_args()

    prompts = make_prompts(args.concurrency * args.batches)
    warmup_prompts = make_prompts(args.concurrency)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    budgets = parse_csv_ints(args.budgets)

    common_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "enforce_eager": True,
        "max_num_seqs": args.concurrency,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": args.attention_backend,
        "disable_log_stats": True,
        "async_scheduling": args.async_scheduling,
    }

    cases: list[tuple[str, dict[str, Any] | None]] = [
        ("target_treeattn", None),
        (
            "dflash_treeattn",
            {
                "model": args.draft_model,
                "method": "dflash",
                "num_speculative_tokens": args.num_speculative_tokens,
            },
        ),
    ]
    for budget in budgets:
        cases.append(
            (
                f"ddtree_budget_{budget}",
                {
                    "model": args.draft_model,
                    "method": "ddtree",
                    "num_speculative_tokens": args.num_speculative_tokens,
                    "ddtree_tree_budget": budget,
                },
            )
        )

    results = []
    with tempfile.TemporaryDirectory(prefix="ddtree_bench_") as tmpdir:
        metrics_dir = Path(tmpdir)
        for name, spec in cases:
            result = run_case(
                name=name,
                common_kwargs=common_kwargs,
                speculative_config=spec,
                prompts=prompts,
                warmup_prompts=warmup_prompts,
                sampling=sampling,
                metrics_dir=metrics_dir,
                debug_metrics=not args.no_debug_metrics and spec is not None,
            )
            results.append(result)
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

    by_name = {result["name"]: result for result in results}
    dflash_tps = by_name["dflash_treeattn"]["output_tps"]
    target_tps = by_name["target_treeattn"]["output_tps"]
    print("\nSummary")
    print("name,output_tps,speedup_vs_target,delta_vs_dflash,mean_acceptance_length")
    for result in results:
        metrics = result.get("metrics", {})
        output_tps = result["output_tps"]
        speedup_vs_target = output_tps / target_tps if target_tps else None
        delta_vs_dflash = (
            output_tps / dflash_tps - 1
            if dflash_tps and result["name"].startswith("ddtree")
            else None
        )
        mean_acceptance_length = metrics.get("mean_acceptance_length")
        print(
            f"{result['name']},{output_tps:.2f},"
            f"{speedup_vs_target if speedup_vs_target is not None else ''},"
            f"{delta_vs_dflash if delta_vs_dflash is not None else ''},"
            f"{mean_acceptance_length if mean_acceptance_length is not None else ''}"
        )


if __name__ == "__main__":
    main()
