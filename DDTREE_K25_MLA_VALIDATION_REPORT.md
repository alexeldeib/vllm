# DDTree + DFlash Kimi K2.5 MLA Validation Report

Date: May 4, 2026

## Executive Summary

This report covers the current Kimi K2.5 DDTree+DFlash implementation and
monkeypatch validation in `cw4637-dev-us-e-01a`, namespace `ace-inference`, on a
single GB200 node.

The K2.5 path now has the production-critical pieces that were missing:

- Native MLA DDTree verification in the target attention path.
- Value-width MLA verifier output, so the RoPE key channels are no longer
  accumulated and then discarded.
- CUDA-graph-compatible DDTree attention-bias workspace for normal decode tree
  windows.
- Accepted-KV compaction for Kimi/K2.5 MLA KV cache, including a direct Triton
  row-copy path and an allocation-reusing fallback.
- Adaptive tree budgeting controls and a linear-DFlash bypass when the tree
  policy collapses to a top-1 chain.
- Production-safe fallback for non-greedy or logprob requests: those batches use
  normal linear DFlash verification instead of crashing the engine.

The earlier bad result did not reflect DDTree's intended economics. True DDTree
was forcing eager execution and spending about 150-170 ms p50 in target forward
on the short C=8 smoke. After the graph-safe bias path and value-width MLA
verifier, the same class of DDTree step runs around 18-24 ms p50 target forward.

Current best short-smoke result:

- DDTree+DFlash budget 8, C=8, greedy: **741.8 output tok/s** warmed.
- Accepted tokens exactly matched compacted KV tokens in the final smoke:
  **25,068 accepted, 25,068 compacted**.
- Target-forward p50: **18.6 ms** on the warmed short smoke.
- KV compaction p50: **7.6 ms** on the warmed short smoke.

Current full no-delay Rebench/AIPerf C=8 DDTree result:

- Workload: 2,025-turn Rebench trace, 48 sessions, no delay, sequential,
  streaming, greedy `temperature=0`.
- Successful requests: **1,507**.
- Context-limit rejections: **517** HTTP 400 responses where the accumulated
  prompt plus requested output crossed K2.5's 32,768-token max context.
- Other client/server error: **1**.
- Output throughput over successful requests: **294.5 output tok/s**.
- TTFT p50/p95: **738 ms / 1,343 ms**.
- ITL p50/p95: **23.3 ms / 40.2 ms**.
- Engine remained alive; no DDTree verifier crash or EngineDead failure.

A matching DFlash-only C=8 no-delay Rebench run used a disposable pod with the
same target weights, same DFlash head, same image, TP=4, and `max_num_seqs=8`,
changing only `method=ddtree` to `method=dflash`.

That apples-to-apples run shows DDTree is still behind DFlash on the long-context
Rebench trace:

- DFlash-only: **352.3 output tok/s**.
- DDTree+DFlash budget 8: **294.5 output tok/s**.
- Delta: **-16.4% output tok/s** for DDTree.

The remaining gap is concentrated in the MLA verifier on long-context turns:
DFlash target-forward p95 was **25.7 ms**, while DDTree target-forward p95 was
**511 ms**. That points to the current unified DDTree verifier's long-prefix
cost.

The split MLA verifier path was tested next. It is fast when the batch runs in
`FULL` CUDA graph mode: target-forward p50/p95 were **16.0 ms / 16.7 ms** for
those rows on the short smoke. A hybrid policy that used split for `FULL` and
unified for `PIECEWISE`/`NONE` was then validated on the full C=8 Rebench trace.
It fixed normal decode verifier latency but regressed end-to-end throughput:

- Hybrid DDTree+DFlash: **268.0 output tok/s**.
- Unified DDTree+DFlash: **294.5 output tok/s**.
- DFlash-only: **352.3 output tok/s**.

The hybrid run reduced target-forward p95 from **511 ms** to **26.3 ms**, but
acceptance dropped from **34.6%** to **28.7%**, increasing the number of target
steps enough to lose overall. Therefore the split verifier remains opt-in for
debugging and follow-up optimization. The production default is the unified MLA
verifier.

## Branch And Scope

Branch:

- `alex/ddtree-glm51-scaffold`

Main files changed:

- `vllm/config/speculative.py`
- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/spec_decode/ddtree.py`
- `vllm/v1/spec_decode/llm_base_proposer.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `tests/v1/spec_decode/test_ddtree.py`

## Implementation Details

### MLA Tree Verifier

`MLAAttention.forward_impl()` now routes DDTree verification through an MLA-aware
tree verifier when the batch carries DDTree attention metadata.

The default verifier uses the Triton unified attention path with MLA-specific
K/V handling:

- Query uses `[q_nope, q_rope]`.
- Key uses `[c_kv, k_rope]`.
- Value uses only `c_kv`.
- Output shape is `[num_tokens, num_heads, kv_lora_rank]`, not full key width.

This removes the previous wasted work of accumulating RoPE key channels as value
channels and slicing them off afterward.

A split verifier reuses the existing MLA decode kernel for prefix attention and
separately computes the small tree suffix mask, then merges prefix and suffix
states by log-sum-exp. On the K2.5 smoke it is fast under `FULL` graph capture,
but it is not default-ready after the full Rebench acceptance regression.

The default policy is now `VLLM_DDTREE_MLA_SPLIT_VERIFIER=unified`: unified
verifier for all graph modes. Operators can still force the experimental
behaviors:

- `VLLM_DDTREE_MLA_SPLIT_VERIFIER=always` forces split verifier everywhere.
- `VLLM_DDTREE_MLA_SPLIT_VERIFIER=off` or `unified` uses unified verifier.
- `auto`, `full`, `full_only`, and `hybrid` use split only under
  `CUDAGraphMode.FULL` and unified otherwise.

Debug events now include `ddtree_mla_verifier_policy` and
`ddtree_mla_verifier_expected` so a smoke or Rebench run can confirm whether
`FULL` rows are expected to use split and `PIECEWISE`/`NONE` rows are expected
to use unified.

### CUDA Graph Compatibility

The initial DDTree implementation forced eager execution whenever speculative
metadata was present. That was the main reason the early C=8 numbers were far
below DFlash.

The current runner stores DDTree attention bias in a stable GPU workspace and
only forces eager when the query window has a tree offset that is not yet graph
safe. Normal decode tree windows now run with CUDA graph modes:

- Short C=8 final smoke: mostly `FULL`, with some `PIECEWISE`.
- Full Rebench C=8: `FULL` for 62,244 debug events, `PIECEWISE` for 220, and
  `NONE` for 5,372 long/prefill-heavy events.

### Accepted-KV Compaction

K2.5 DDTree verification writes KV for every scheduled tree node, but only the
accepted path can remain in canonical request order. The runner now compacts the
accepted tree path into canonical KV slots after sampling.

The implementation handles:

- MLA 3D KV cache layout.
- Standard key/value split cache layout.
- `UniformTypeKVCacheSpecs` wrapping full-attention specs.
- Reusable fallback workspaces for overlap-safe `index_select` + `index_copy_`.
- Direct Triton row copy for contiguous CUDA cache rows.

The direct Triton copy walks accepted rows in reverse path order. That preserves
correctness when a lower canonical destination overlaps a deeper accepted source
row that has not been copied yet.

### Adaptive Tree Budgeting

Speculative config now supports:

- `ddtree_min_path_probability`
- `ddtree_min_branch_gain_per_node`

The tree builder records expected acceptance and branch gain per extra tree node.
If the tree collapses to a linear top-1 chain for every request in the batch, the
proposer emits normal DFlash draft tokens and clears DDTree metadata. The
scheduler then uses the standard linear speculative verifier, avoiding DDTree
target-verifier overhead.

### Non-Greedy Safety

True DDTree verification is exact for greedy sampling because it follows target
argmax through a tree. It is not exact for random sampling or logprob-producing
requests.

The current implementation detects those batches and falls back to normal linear
DFlash tokens with no DDTree metadata. This keeps production traffic safe:

- Greedy requests use DDTree when the tree policy emits tree metadata.
- Non-greedy requests use DFlash and the existing rejection sampler.
- Logprob requests use DFlash and the existing verifier path.

The previous behavior was an engine-killing runtime error. The fallback was
validated in-cluster with a stochastic chat request; the engine stayed alive and
debug metrics reported `fallback_reason=non_greedy` with `metadata_bypass=true`.

## Correctness Evidence

Local checks:

```text
python3 -m py_compile \
  vllm/model_executor/layers/attention/mla_attention.py \
  vllm/v1/attention/backend.py \
  vllm/v1/attention/ops/triton_unified_attention.py \
  vllm/v1/spec_decode/llm_base_proposer.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/spec_decode/ddtree.py \
  vllm/config/speculative.py \
  tests/v1/spec_decode/test_ddtree.py
```

Result: passed after the final reverse-copy compaction hardening.

Additional local checks after the split-verifier policy and debug-metric
instrumentation:

```text
python3 -m py_compile \
  vllm/model_executor/layers/attention/mla_attention.py \
  vllm/v1/worker/gpu_model_runner.py \
  tests/v1/spec_decode/test_ddtree.py

git diff --check
```

Result: passed.

`pytest -q tests/v1/spec_decode/test_ddtree.py` could not be run in the current
shell because `pytest` is not installed. A temporary `/tmp` venv with
`pytest`/`tblib` got past the test runner dependency, but the local checkout is
not a full vLLM dev environment and then failed importing project dependencies
such as `pydantic`. The focused test still needs to be run in the CI/dev image.

In-cluster functional checks:

- Kimi K2.5 NVFP4 target weights loaded on GB200.
- Drafter loaded from `z-lab/Kimi-K2.5-DFlash`.
- Attention backend selected `FLASHINFER_MLA` with HND cache layout.
- KV cache dtype was `fp8_e4m3`.
- Greedy arithmetic probe `19 + 23 =` returned `42`.
- Stochastic chat request did not crash; it used linear DFlash fallback.
- C=8 short smoke completed with real DDTree metadata and no eager-forced
  verifier.
- Full no-delay Rebench C=8 completed without EngineDead.
- Split MLA verifier smoke completed correctly, but only `FULL` graph rows were
  fast enough to be promising.

Accepted/compacted invariant:

| Workload | Accepted token sum | Compacted token sum | Result |
| --- | ---: | ---: | --- |
| Final warmed C=8 short smoke | 25,068 | 25,068 | Exact match |
| Full no-delay Rebench C=8 | 1,216,688 | 1,216,688 | Exact match |

The debug stream is emitted per tensor-parallel worker, so absolute sums are
rank-repeated. Equality is the invariant that matters: every accepted DDTree node
emitted by the sampler was also compacted into canonical KV.

## Stacked Performance

Short-smoke workload:

- API: `/v1/completions`
- Client concurrency: 8
- Requests: 32
- `max_tokens`: 64
- Sampling: greedy, `temperature=0`
- Prompt shape: short arithmetic prompts

| Step | Mode | Output tok/s | Target fwd p50 ms | KV compact p50 ms | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| A | Initial native MLA verifier, budget 32, eager | 169.6 | ~167 | 8.6 | Correct but graph-hostile |
| B | Adaptive threshold 0.05, mixed tree/linear | 133.7 | ~169 | ~8 | Worse tree economics |
| C | Forced-linear DDTree/DFlash bypass | 669.6 | 22.2 | ~0 | Proxy for linear DFlash path |
| D | Budget 16, graph-safe bias, workspace compaction | 631.3 | 27.7 | 8.4 | True DDTree active |
| E | Budget 16, direct Triton compaction, warmed | 642.7 | 27.5 | 7.8 | True DDTree active |
| F | Budget 8, graph-safe + direct compaction, warmed | 718.6 | 18.5 | 7.5 | True DDTree active |
| G | Final budget 8 with non-greedy fallback, warmed | 741.8 | 18.6 | 7.6 | True DDTree active |

Final warmed C=8 short-smoke details:

- Output throughput: **741.8 output tok/s**.
- Output tokens: **8,169**.
- RPS: **5.81**.
- Mean latency: **1.324 s**.
- P50 latency: **1.253 s**.
- P95 latency: **2.002 s**.
- Acceptance rate: **38.7%**.
- Mean accepted tokens per DDTree step: **3.00**.
- Mean generated tokens per DDTree step: **4.00**.
- Tree nodes per request step: **8**.

Full no-delay Rebench/AIPerf C=8 comparison:

| Metric | DDTree+DFlash budget 8 | DFlash-only | DDTree delta |
| --- | ---: | ---: | ---: |
| Attempted turns | 2,025 | 2,025 | same |
| Successful requests | 1,507 | 1,507 | same |
| Context-limit HTTP 400s | 517 | 517 | same |
| Other errors | 1 | 1 | same |
| Benchmark duration | 1,371.8 s | 1,158.3 s | +18.4% |
| Output throughput | 294.5 tok/s | 352.3 tok/s | -16.4% |
| Request throughput | 1.10 req/s | 1.30 req/s | -15.6% |
| Total output tokens | 403,989 | 408,040 | -1.0% |
| Mean input length | 16,443 tokens | 16,443 tokens | same |
| P50 input length | 16,477 tokens | 16,477 tokens | same |
| P95 input length | 30,307 tokens | 30,307 tokens | same |
| Mean output length | 268.1 tokens | 270.8 tokens | -1.0% |
| TTFT p50 / p95 | 738 ms / 1,343 ms | 442 ms / 837 ms | +67% / +60% |
| ITL p50 / p95 | 23.3 ms / 40.2 ms | 17.8 ms / 31.2 ms | +31% / +29% |
| Request latency p50 / p95 | 5.20 s / 17.37 s | 3.65 s / 12.48 s | +42% / +39% |

Server-side debug for the full Rebench runs:

| Metric | DDTree+DFlash budget 8 | DFlash-only |
| --- | ---: | ---: |
| Acceptance rate | 34.6% | not recorded in DFlash debug |
| Mean accepted tokens | 2.73 | not recorded in DFlash debug |
| Mean generated tokens | 3.73 | not recorded in DFlash debug |
| Mean expected acceptance | 2.87 | n/a |
| Tree nodes | 8 | n/a |
| CUDA graph FULL events | 62,244 | 123,732 |
| CUDA graph NONE events | 5,372 | 5,360 |
| Target forward p50 / p95 | 23.6 ms / 511.3 ms | 18.9 ms / 25.7 ms |
| KV compact p50 / p95 | 7.69 ms / 7.96 ms | 0.03 ms / 0.04 ms |
| Draft total p50 / p95 | 5.21 ms / 6.13 ms | 4.44 ms / 5.04 ms |

The high DDTree target-forward p95 is not present in DFlash under the same trace.
That makes it a DDTree verifier issue, not a general K2.5 workload issue.

Split verifier warmed short-smoke timer split:

| CUDA graph mode | Rows | Target fwd p50 | Target fwd p95 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `FULL` | 296 | 16.0 ms | 16.7 ms | Better than unified verifier |
| `PIECEWISE` | 192 | 207.2 ms | 237.3 ms | Not production-ready |

The split path proves the prefix/suffix formulation can beat the unified
verifier when captured as a full graph, but the mixed graph path is currently too
expensive. The next verifier optimization should focus on making this path graph
stable for all common padded decode sizes, or replacing the Python/Triton split
with a native fused MLA tree verifier.

Hybrid verifier full C=8 Rebench result:

| Metric | Hybrid split-on-FULL | Unified DDTree | DFlash-only |
| --- | ---: | ---: | ---: |
| Output throughput | 268.0 tok/s | 294.5 tok/s | 352.3 tok/s |
| Request throughput | 0.973 req/s | 1.099 req/s | 1.301 req/s |
| Benchmark duration | 1,549.5 s | 1,371.8 s | 1,158.3 s |
| Successful requests | 1,507 | 1,507 | 1,507 |
| Context-limit errors | 518 | 517 | 517 |
| Other errors | 0 | 1 | 1 |
| Total output tokens | 415,290 | 403,989 | 408,040 |
| TTFT p50 / p95 | 727 ms / 1,356 ms | 738 ms / 1,343 ms | 442 ms / 837 ms |
| ITL p50 / p95 | 22.2 ms / 38.5 ms | 23.3 ms / 40.2 ms | 17.8 ms / 31.2 ms |

Hybrid verifier server-side debug:

| Metric | Hybrid split-on-FULL |
| --- | ---: |
| Acceptance rate | 28.7% |
| Accepted token sum | 1,171,708 |
| Compacted token sum | 1,171,708 |
| CUDA graph FULL events | 106,432 |
| CUDA graph PIECEWISE events | 200 |
| CUDA graph NONE events | 5,368 |
| `FULL` target-forward p50 / p95 | 15.0 ms / 25.2 ms |
| `PIECEWISE` target-forward p50 / p95 | 31.4 ms / 565.0 ms |
| `NONE` target-forward p50 / p95 | 531.7 ms / 624.6 ms |
| Overall target-forward p50 / p95 | 19.8 ms / 26.3 ms |

This shows the split path solved the normal `FULL` verifier latency problem, but
not the end-to-end performance problem. The next optimization target is
acceptance parity and the remaining `NONE`/long-context verifier path, not a
container build.

## Apples-To-Apples Baselines

The earlier target-only and DFlash-only numbers were from a short arithmetic
workload, not the full no-delay Rebench trace:

| Mode | Output tok/s | Notes |
| --- | ---: | --- |
| Target-only | 503.8 | Short arithmetic smoke |
| DFlash-only | 765.5 | Short arithmetic smoke |
| DDTree+DFlash final | 741.8 | Short arithmetic smoke, warmed |

On that smoke, final DDTree is about **+47% vs target-only** and about **-3% vs
the older DFlash-only reference**. This is close enough that Rebench is the
right next comparison; the short prompt is not representative of the no-delay
multi-turn workload.

The matching DFlash-only Rebench comparison is now complete. It shows that the
short smoke was too favorable to DDTree; the full no-delay Rebench trace exposes
long-prefix MLA verifier cost that the short prompt did not exercise.

## Why The Early Result Did Not Match DDTree Headline Gains

The earlier roughly flat or single-digit result was measuring our implementation
bottlenecks more than DDTree:

- True DDTree was forcing eager execution. That alone made target forward about
  150-170 ms p50 on the short C=8 smoke.
- The original MLA verifier accumulated full key width and discarded the RoPE
  value channels. The current verifier accumulates only `c_kv`.
- Budget 32 over-expanded the tree for the observed acceptance. Budget 8 has
  better economics for the current K2.5 DFlash head and C=8 decode shape.
- K2.5 DFlash is already a strong baseline. DDTree has to beat optimized DFlash,
  not only autoregressive target decode.
- Full Rebench contains long-context turns where prefill and graph misses dominate
  p95. Those are not comparable to the short arithmetic smoke or to DDTree
  headline plots unless the baseline uses the same trace and limits.

After stacking graph-safe verification, value-width MLA output, direct
compaction, and budget 8, short-smoke throughput moved from **169.6 tok/s** to
**741.8 tok/s**, a **4.4x** improvement over the initial correct implementation.

## Architecture Limits

Current Kimi/K2.5 support:

- Supported: Kimi K2.5 NVFP4 target weights with `fp8_e4m3` MLA KV cache.
- Supported: DFlash head from `z-lab/Kimi-K2.5-DFlash`.
- Supported: TP=4 on GB200, HND MLA cache layout.
- Supported: greedy DDTree verification.
- Supported: production-safe non-greedy/logprob fallback to linear DFlash.

Current limitations:

- True DDTree verification is greedy-only. Non-greedy and logprob requests are
  safe, but they do not use tree verification.
- Packed `fp8_ds_mla` KV cache is not supported.
- NVFP4 KV cache is not supported. The tested deployment uses NVFP4 weights, not
  NVFP4 KV cache.
- MLA decode context parallelism with DDTree is not supported.
- Query windows with nonzero tree offsets still force eager. The common decode
  tree window is graph-safe; chunked/catch-up tree windows need a graph-safe
  offset-bias path.
- The direct compaction kernel assumes accepted path order and now copies in
  reverse to avoid lower-slot overlap. Randomized/property tests should be added
  before removing the fallback path.
- First-use Triton JIT can still produce compaction outliers. Production should
  warm the compaction kernel during capture/startup.
- Current in-cluster validation pod used `max_num_seqs=8`. C=16/32/48 Rebench
  requires a production-shaped pod with `max_num_seqs` around 50.

Kimi K2.6 support should be straightforward if it keeps the same Kimi MLA cache
shape and has a compatible DFlash head. The likely work is packaging and
validation: model config checks, tokenizer/reasoning parser wiring, and the same
Rebench/AIPerf sweep.

GLM-5.1 is a larger stretch goal. It needs a model-specific verifier and KV
compaction review because GLM's attention/cache layout is not guaranteed to
match Kimi MLA. The safe plan is to stack it as a separate PR after Kimi
stabilizes: add model gating, implement the verifier/cache adapter, keep the
non-greedy fallback, and run correctness tests without changing Kimi behavior.

## Production Readiness

Ready enough for code review:

- Correct Kimi K2.5 MLA DDTree verification.
- Correct accepted-KV compaction on the tested MLA cache layout.
- No engine crash for stochastic traffic.
- CUDA graph execution for normal decode tree windows.
- C=8 short smoke positive.
- C=8 no-delay Rebench DDTree run completed.

Not ready to merge as a production default until:

- The split/native MLA verifier closes the long-context Rebench gap to DFlash
  without reducing acceptance or regressing `PIECEWISE`/`NONE` graph modes.
- Full Rebench sweep runs at C=1,2,4,8,16,32,48 on a pod configured for
  `max_num_seqs >= 48`.
- Compaction kernel warmup is added.
- Local/unit tests run in a working vLLM dev environment.
- The monkeypatch is replaced by a real multi-arch `vllm-tensorizer` image build
  from `/Users/aeldeib/code/ml-containers`.

## Next Actions

1. Make the split/native MLA verifier graph-stable outside `FULL` mode, or
   replace it with a fused MLA tree verifier that avoids the `PIECEWISE` 200 ms
   path.
2. Rerun DDTree C=8 no-delay Rebench once the verifier p95 is in the same range
   as DFlash's 25.7 ms target-forward p95.
3. Run a production-shaped sweep at C=1,2,4,8,16,32,48 on a pod configured for
   `max_num_seqs >= 48`.
4. Commit and push `alex/ddtree-glm51-scaffold` to `alexeldeib/vllm`.
5. Start the multi-arch `vllm-tensorizer` build from `ml-containers` only after
   DDTree is neutral or positive versus DFlash on the same Rebench trace.
