# DDTree + DFlash Kimi K2.5 MLA Validation Report

Date: May 4, 2026

## Summary

This report covers the current Kimi K2.5 DDTree+DFlash monkeypatch validation in
`cw4637-dev-us-e-01a`, namespace `ace-inference`, on GB200. The implementation
now includes the two production-critical K2.5 pieces that were previously
missing:

- MLA tree verification for DDTree target verification.
- Accepted-KV compaction for Kimi/K2.5 MLA KV cache.

Correctness is now meaningfully exercised in-cluster: DDTree verifies through the
MLA target path, emits accepted tree nodes, and compacts exactly those accepted
nodes into canonical KV slots. The current result is still not build-ready
because performance is worse than both target-only and DFlash-only on the C=8
K2.5 serving-shaped smoke. I did not trigger a new `ml-containers` build from
this state.

## Implementation State

Current vLLM branch: `alex/ddtree-glm51-scaffold`

Touched files:

- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/spec_decode/ddtree.py`
- `tests/v1/spec_decode/test_ddtree.py`

What changed:

- `CommonAttentionMetadata` now carries DDTree tree-query start/lens metadata.
- MLA metadata now carries CPU query starts, sequence lengths, tree attention
  bias, and tree-window offsets.
- `MLAAttention.forward_impl()` routes DDTree tree verification through a Triton
  unified-attention path that applies the q-query tree mask to MLA cache rows.
- The MLA verifier supports the current K2.5 deployment shape:
  `FLASHINFER_MLA`, HND layout, `fp8_e4m3` KV cache, TP=4, and NVFP4 target
  weights.
- `gpu_model_runner` now supports scheduler steps where a request has a
  catch-up prefix plus a tree suffix. Tree positions are depth-based while
  prefix positions remain contiguous.
- DDTree attention bias is embedded into the full query window so non-tree
  catch-up rows keep normal causal visibility and tree rows get root/ancestor
  visibility.
- Accepted-KV compaction now handles the K2.5 MLA 3D cache layout and
  `UniformTypeKVCacheSpecs` wrapping full-attention specs.
- DFlash drafter token-index construction now understands tree suffix offsets,
  so the next proposal consumes the actual accepted tree path rather than a
  false contiguous prefix.
- A CPU reference test was added for MLA tree-attention visibility.

Unsupported in this patch:

- Decode context parallelism for MLA DDTree.
- Packed `fp8_ds_mla` KV cache.
- NVFP4 KV cache. The tested K2.5 deployment uses NVFP4 weights with
  `fp8_e4m3` KV cache, which is supported.
- Non-greedy sampling and logprobs.
- GLM-5.1 DSA/stateful-cache compaction.

## Test Environment

Target-only reference:

- Service: `c2-k25-v0191-bench-trtllm`
- Model: `nvidia/Kimi-K2.5-NVFP4:k25-v0191-bench-trtllm`

DFlash-only reference:

- Service: `c2-k25-v0191-bench-spec-trtllm`
- Model: `nvidia/Kimi-K2.5-NVFP4:k25-v0191-bench-spec-trtllm`

DDTree monkeypatch pod:

- Pod: `k25-ddtree-mla-patch-e2e`
- Model: `nvidia/Kimi-K2.5-NVFP4-ddtree-mla-patch`
- Target weights: existing in-cluster Kimi K2.5 NVFP4 weights
- Drafter: `z-lab/Kimi-K2.5-DFlash`
- TP: 4
- `max_num_seqs`: 8
- `max_num_batched_tokens`: 8192
- `num_speculative_tokens`: 8
- Debug metrics: `SPEC_DECODE_DEBUG_METRICS=1`
- Metrics file: `/tmp/spec_debug.jsonl`

The DDTree pod uses the previously built image as a base and monkeypatches the
current local source at runtime. This is intentional: the validation goal is to
prove correctness and performance before spending an hour or more on a fresh
multi-arch image build.

## Correctness Evidence

Local syntax validation:

```text
python3 -m py_compile \
  vllm/v1/attention/backend.py \
  vllm/model_executor/layers/attention/mla_attention.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/spec_decode/ddtree.py \
  tests/v1/spec_decode/test_ddtree.py
```

Result: passed.

In-cluster functional validation:

- The DDTree pod loaded Kimi K2.5 NVFP4 target weights and
  `z-lab/Kimi-K2.5-DFlash`.
- The target attention backend selected `FLASHINFER_MLA` with HND KV cache
  layout.
- KV cache dtype was `fp8_e4m3`.
- A direct arithmetic probe for `19 + 23` returned `42`.
- C=8 mixed requests completed without the TRTLLM ragged-prefill workspace
  crash that appeared before the mixed decode-path fix.
- DDTree debug metrics show accepted-token accounting and compacted-token
  accounting match exactly.

Accepted-KV compaction checks:

| Run | Budget | Accepted token sum | Compacted token sum | Result |
| --- | ---: | ---: | ---: | --- |
| C=8 smoke | 8 | 3,768 | 3,768 | Exact match |
| C=8 smoke | 16 | 4,120 | 4,120 | Exact match |
| C=8 smoke | 32 | 4,232 | 4,232 | Exact match |

The debug rows are emitted per tensor-parallel worker, so the absolute sums are
rank-repeated. The equality is the important invariant: every accepted DDTree
node that the sampler emits is also compacted into canonical KV.

## Performance Results

Workload:

- API: `/v1/completions`
- Requests: 32
- Client concurrency: 8
- `max_tokens`: 64
- Sampling: greedy, `temperature=0`
- Prompt shape: short arithmetic prompts:
  `Calculate A + B. Give a concise answer with the arithmetic.`

Fresh C=8 comparison:

| Mode | DDTree budget | Output tok/s | RPS | Mean latency s | P50 latency s | P95 latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target-only | n/a | 503.79 | 14.04 | 0.539 | 0.560 | 0.763 |
| DFlash-only | n/a | 765.50 | 17.36 | 0.428 | 0.394 | 0.684 |
| DDTree+DFlash | 8 | 114.40 | 3.05 | 2.341 | 1.876 | 4.367 |
| DDTree+DFlash | 16 | 122.46 | 3.11 | 2.250 | 1.746 | 4.219 |
| DDTree+DFlash | 32 | 178.94 | 4.61 | 1.544 | 1.509 | 2.522 |

Relative to the DFlash-only reference:

| DDTree budget | Output tok/s delta vs DFlash | RPS delta vs DFlash |
| ---: | ---: | ---: |
| 8 | -85.1% | -82.5% |
| 16 | -84.0% | -82.1% |
| 32 | -76.6% | -73.4% |

Budget 32 is the best of the three tested budgets on this smoke, but it is still
far below DFlash-only. This is not production-ready performance.

Stage metrics from the DDTree debug stream:

| Budget | Accepted mean | Accepted p50 | Scheduled tokens mean | Tree nodes mean | Target forward p50 ms | KV compact p50 ms | Draft total p50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3.04 | 2.5 | 9.6 | 7.2 | 155.39 | 7.92 | 5.23 |
| 16 | 3.69 | 3.0 | 16.78 | 14.22 | 160.96 | 8.08 | 6.05 |
| 32 | 3.82 | 4.0 | 30.99 | 28.42 | 150.1 | 7.92 | 6.10 |

The p50 target-forward time is roughly flat around 150-161 ms across these
budgets, and accepted length is only about 3-4 tokens. That combination is bad
for DDTree economics: the verifier pass is expensive, but the accepted path is
not long enough to amortize it.

## Why This Does Not Match DDTree Headline Gains

The current K2.5 path has the DDTree semantics, but not yet the DDTree economics.

Main causes:

- The MLA verifier currently uses a generic Triton unified-attention path for
  q-query tree masking. It is correct, but it is not an optimized FlashInfer MLA
  tree-verifier kernel.
- The verifier concatenates MLA `c_kv` and RoPE channels, runs attention, and
  slices the `c_kv` output. This is a practical correctness path, not the final
  minimum-work MLA implementation.
- Accepted-KV compaction costs about 8 ms p50 per debug step on this K2.5 run.
  That is not the dominant cost, but it is material.
- Acceptance is too low for the verifier work: budget 32 schedules about 31
  tokens per request step while accepting only about 3.8 tokens on average.
- Smaller static budgets did not fix the issue. Budgets 8 and 16 reduced the
  tree size but also reduced acceptance and remained slower than budget 32.
- The DFlash-only reference is already highly optimized for this K2.5 serving
  path. DDTree must beat that optimized baseline, not just autoregressive
  decoding.

The conclusion is not that DDTree is wrong; the conclusion is that the current
K2.5 MLA verifier is a correctness implementation, not yet the state-of-the-art
performance implementation.

## Build Decision

Do not kick off the full `ml-containers` image build from this state.

A build would prove packaging, but the monkeypatch already proved the important
runtime facts:

- MLA tree verification works.
- Accepted-KV compaction works.
- C=8 mixed scheduling no longer crashes.
- Performance is currently below DFlash-only by a large margin.

The next image build should wait until the monkeypatch path beats DFlash-only on
the same K2.5 serving-shaped workload.

## Production-Readiness Gaps

To make this production-ready for K2.5, the remaining work is performance, not
basic correctness.

Priority 1: native MLA tree verifier

- Replace the generic Triton unified-attention verifier with a specialized MLA
  tree-verification path.
- Avoid computing unused output channels.
- Preserve the root/ancestor/self q-query mask.
- Support varlen mixed rows without falling back into ragged prefill workspace
  growth after graph capture.
- Keep support for `fp8_e4m3` MLA KV cache and HND layout.

Priority 2: compaction optimization

- Reduce the current about-8 ms p50 accepted-KV compaction cost.
- Prefer a fused cache-copy path across layers/groups instead of repeated
  per-layer index copies.
- Keep exact accepted-node semantics; do not trade correctness for speed here.

Priority 3: adaptive tree policy

- Static budgets 8, 16, and 32 are all losing on this smoke.
- Add a policy that shrinks or skips DDTree when predicted acceptance is low or
  target batch pressure is high.
- Expand the tree only when DFlash confidence predicts that the accepted path
  will be long enough to amortize verifier cost.

Priority 4: workload validation

- Re-run the K2.5 no-delay Rebench/AIPerf-style trace once the C=8 smoke is
  positive.
- Keep no-debug runs for headline throughput.
- Keep sampled debug runs for stage timers and accepted/compacted accounting.

## Next Recommendation

The immediate next engineering target should be the native MLA tree verifier.
The compaction path is now correct and measurable, but verifier cost dominates
the current result. A better budget policy may reduce damage, but it will not
turn the current generic verifier into the state-of-the-art path on its own.

