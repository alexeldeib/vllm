# DDTree + DFlash Performance and Architecture Report

> Kimi K2.5 MLA update: this report is the older Qwen/full-attention DDTree
> readout. The current K2.5 MLA monkeypatch validation, including MLA tree
> verification, accepted-KV compaction, and the negative DFlash-vs-DDTree
> performance result, is in `DDTREE_K25_MLA_VALIDATION_REPORT.md`.

## Scope

This report covers the current vLLM DDTree integration as a DFlash extension:
what is implemented, what has been validated on GB200, the measured DFlash vs
DDTree+DFlash smoke deltas, what is now closer to upstream DDTree, and what
still blocks Kimi K2.5/K2.6 and GLM-5.1.

The detailed image-build history and earlier clean-image validation remain in
`DDTREE_VLLM_REPORT.md`. This file is the focused performance and architecture
readout for the DDTree path.

## Current State

- vLLM branch: `alex/ddtree-vllm-integration`
- Pushed review branch: `alexeldeib/vllm:alex/ddtree-vllm-integration`
- Current vLLM commit: `6acdb096fea8723d337292251068ee5928ce00a2`
- Current clean image:
  `ghcr.io/coreweave/ml-containers/vllm-tensorizer:alex-ddtree-vllm-image-7cf4eae-6acdb096fea8723d337292251068ee5928ce00a2`
- Current clean image manifest digest:
  `sha256:05a12a894d2a548ba7d9182eb2cab72feeca6a07cf1956e62950a8026060b21f`
- Platforms for that image: `linux/amd64`, `linux/arm64`
- Platform manifests:
  - `linux/amd64`: `sha256:aaf31d599ac88c9404d3ff91b20781c11affdd1b25ff12949adfc5543f1d9b17`
  - `linux/arm64`: `sha256:ce4dd39d65a0889b1e305587a3c708cf94bfad2c4830cb02936f2052451c3faa`
- `ml-containers` branch: `alex-ddtree-vllm-image`
- `ml-containers` commit: `7cf4eae7e6c1458928edbcf356fef24a4db66c6b`
- CI run: `https://github.com/coreweave/ml-containers/actions/runs/25264137128`
- Earlier monkeypatch validation pod: `ddtree-batch-vllm-test`
- Clean-image validation pod: `ddtree-vllm-7cf4eae-gb200`
- Cluster and namespace: `cw4637-dev-us-e-01a`, `ace-inference`
- Runtime architecture: `aarch64`
- Clean-image vLLM package: `0.1.dev16276+g6acdb096f.d20260502`

The current branch now implements the production-critical pieces for the
standard full-attention DDTree path:

- Speculative method `ddtree`.
- DFlash drafter integration for parallel draft logits.
- Batched prefix-closed DDTree construction from a full request batch.
- Batched 3D per-request tree attention bias for target verification.
- `TREE_ATTN` target verification with per-request tree masks.
- Greedy target walk that records accepted DDTree node indices.
- Accepted-path KV compaction from scratch tree slots into canonical contiguous
  paged-KV slots for standard full-attention KV cache.
- Scheduler accounting for compacted DDTree accepted tokens, so accepted tree
  nodes are not recomputed in the next step.
- Immediate next-tree proposal after DDTree verification, using the actual
  accepted tree-node path rather than assuming accepted nodes are a linear
  prefix of the depth-ordered tree.
- Mixed-batch handling where DDTree verification rows can share a scheduler
  step with ordinary prefill rows; DDTree rows use accepted-node indices and
  non-tree rows keep contiguous prompt hidden states for the drafter context.
- Batched top-k/logprob transfer for tree construction, reducing one CPU sync
  per request to one CPU transfer per DDTree batch.
- CPU-side dense tree-bias construction with one host-to-device transfer per
  batch, avoiding the earlier many-small-GPU-write path.
- Optional speculative-decode JSONL debug metrics and an offline no-delay
  DFlash vs DDTree benchmark harness.
- Triton qq-bias cleanup for the vector logical mask warning seen in the GB200
  smoke tests.

## Correctness Evidence

All current correctness checks use greedy decoding, Qwen3-8B target weights, and
`z-lab/Qwen3-8B-DFlash-b16` as the drafter.

| Environment | Check | Result |
| --- | --- | --- |
| Local source | `python3 -m py_compile` for modified DDTree, GPU runner, and benchmark harness files | Passed |
| Local source | `git diff --check` | Passed |
| Monkeypatch GB200 pod | Direct batched DDTree unit smoke | Passed |
| Monkeypatch GB200 pod | 3D qq-bias Triton kernel check | `max_diff 0.0`, passed |
| Monkeypatch GB200 pod | 4-request Qwen3-8B + DFlash/DDTree generate, `TREE_ATTN`, budget 32, immediate re-proposal enabled | Passed |
| Monkeypatch GB200 pod | Target-only greedy vs DDTree+DFlash greedy, selected stable prompts, token IDs | Exact match |
| Earlier monkeypatch image | 48-token target/TREE vs DFlash vs DDTree | Exact token ID match |
| Earlier monkeypatch image | 128-token DFlash vs DDTree, budgets 16, 32, 64 | Exact token ID match with DFlash |
| Current clean CI image | `py_compile` for installed modified files | Passed |
| Current clean CI image | Import/version check on `aarch64` GB200 image | `0.1.dev16276+g6acdb096f.d20260502`, passed |
| Current clean CI image | `DDTreeProposer.take_last_ddtree_debug_metrics()` present | Passed |
| Current clean CI image | Single stable prompt, target/TREE vs DFlash vs DDTree budget 32, 48 generated tokens | Exact token ID match |
| Current clean CI image | 4-prompt DFlash vs DDTree budget 32 run | Generated successfully; one low-margin prompt produced a different target-verified greedy continuation |
| Current clean CI image | Budget sensitivity on the low-margin median prompt | Budget 8 matched DFlash; budget 32 selected a different target-verified branch |
| Current clean CI image | C=4 short DFlash vs DDTree+DFlash timing smoke | Passed |
| Current clean CI image + same source during development | `benchmark_ddtree_dflash.py`, C=4 budget sweep, no debug sync | Passed |
| Current clean CI image + same source during development | `benchmark_ddtree_dflash.py`, C=8 and C=16 no-delay points | Passed |
| Current clean CI image + same source during development | Stage-timer run with acceptance metrics for DFlash and DDTree budgets 32/64 | Passed |

Important caveat: exact token equality against target-only or DFlash is not a
stable universal oracle under the eager BF16 `TREE_ATTN` + DFlash harness. On a
low-margin prompt asking for a median function, DFlash itself diverged from the
target-only baseline, and DDTree budget 32 selected `arguments` where DFlash
selected `parameters`. Budget 8 matched DFlash on the same prompt. This does not
show a verifier crash or malformed KV compaction; the debug counters show
accepted target-verified tree nodes followed by a fallback/root target decision.
It does mean the latest clean image should be described as passing integration,
shape, import, selected exact-match, and smoke-performance checks, with broader
distributional correctness still requiring a larger oracle than byte-for-byte
target-only equality on low-margin prompts.

The key accepted-KV compaction check is still that verified DDTree nodes can be
retained and reused without recomputing them. The branch now advances
`num_computed_tokens` through compacted accepted tree nodes and immediately
re-proposes from the actual accepted node path instead of assuming a linear
prefix.

## Measured Performance

The primary measurement below is from GB200 development runs using the same
source now built into the clean CI image. The target is Qwen3-8B, the
drafter is `z-lab/Qwen3-8B-DFlash-b16`, target attention is forced through
`TREE_ATTN` for every mode, `enforce_eager=True`, `max_tokens=128`,
`max_num_batched_tokens=8192`, and async scheduling is disabled for both DFlash
and DDTree. The workload is an offline no-delay prompt replay with varied
synthetic prompts, not the full K2.5 Rebench trace.

The no-debug throughput runs do not collect acceptance metrics and do not insert
per-stage CUDA synchronizations:

| Concurrency | Mode | Budget | Output tokens | Output tokens/s | Speedup vs target/TREE | Delta vs DFlash |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Target only, `TREE_ATTN` | n/a | 1,024 | 158.84 | 1.00x | n/a |
| 4 | DFlash, `TREE_ATTN` target | n/a | 1,024 | 281.14 | 1.77x | n/a |
| 4 | DDTree+DFlash | 16 | 1,024 | 361.95 | 2.28x | +28.7% |
| 4 | DDTree+DFlash | 32 | 1,024 | 380.84 | 2.40x | +35.5% |
| 4 | DDTree+DFlash | 64 | 1,024 | 401.31 | 2.53x | +42.7% |
| 4 | DDTree+DFlash | 128 | 1,024 | 377.80 | 2.38x | +34.4% |
| 8 | Target only, `TREE_ATTN` | n/a | 2,048 | 308.04 | 1.00x | n/a |
| 8 | DFlash, `TREE_ATTN` target | n/a | 2,048 | 525.25 | 1.71x | n/a |
| 8 | DDTree+DFlash | 32 | 2,048 | 590.20 | 1.92x | +12.4% |
| 8 | DDTree+DFlash | 64 | 2,048 | 489.12 | 1.59x | -6.9% |
| 16 | Target only, `TREE_ATTN` | n/a | 4,096 | 590.57 | 1.00x | n/a |
| 16 | DFlash, `TREE_ATTN` target | n/a | 4,096 | 1,020.30 | 1.73x | n/a |
| 16 | DDTree+DFlash | 32 | 4,096 | 1,161.69 | 1.97x | +13.9% |

This is the corrected answer to the earlier "+5.8%" result. The +5.8% number
came from a small 64-token warmed smoke with no budget sweep and an older
GPU-side dense-bias builder. It was a useful integration check, but not a useful
headline for the algorithm. With the production batching path, accepted-KV
compaction, async disabled for both modes, and the CPU-built tree-bias
optimization, DDTree now shows +35% to +43% over DFlash at C=4 and remains
positive at C=8/C=16 for budget 32.

The C=8 budget-64 loss is expected from DDTree's budget tradeoff: a larger tree
raises accepted-prefix length, but verifier, drafter, tree-build, and metadata
cost can dominate under batching. The current best budget is workload-dependent:
budget 64 wins in this C=4 offline sweep, while budget 32 is safer at C=8 and
C=16.

The final clean CI image was also smoke-tested directly on GB200 with a shorter
C=4, one-batch, 64-token no-debug run:

| Mode | Budget | Output tokens | Output tokens/s | Speedup vs target/TREE | Delta vs DFlash |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target only, `TREE_ATTN` | n/a | 256 | 130.29 | 1.00x | n/a |
| DFlash, `TREE_ATTN` target | n/a | 256 | 139.16 | 1.07x | n/a |
| DDTree+DFlash | 32 | 256 | 248.42 | 1.91x | +78.5% |
| DDTree+DFlash | 64 | 256 | 268.32 | 2.06x | +92.8% |

This short clean-image run is an integration smoke, not the headline result:
DFlash was unusually low on that small sample, so the more stable C=4/C=8/C=16
development sweep remains the better performance readout.

The stage-timer run below uses CUDA synchronizations, so absolute throughput is
slightly different from no-debug throughput. It is useful for decomposition:

| Mode | Budget | Output tokens/s | Mean acceptance length | Target forward ms | Draft propose ms | Tree CPU build ms | Tree bias ms | KV compact ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash, `TREE_ATTN` target | n/a | 275.36 | 2.73 | 24.53 | 4.72 | n/a | n/a | 0.01 |
| DDTree+DFlash | 32 | 375.08 | 4.06 | 24.40 | 5.89 | 0.78 | 0.48 | 2.45 |
| DDTree+DFlash | 64 | 402.07 | 4.29 | 24.12 | 6.66 | 1.52 | 0.92 | 2.47 |

The previous timer run, before CPU-built tree bias, showed `ddtree_attention_bias`
around 3.2 ms for budget 32 and 6.9 ms for budget 64. After the optimization,
that drops to 0.48 ms and 0.92 ms respectively. The remaining optimization
stack is now target/drafter scheduling, accepted-KV compaction, CPU tree build
at larger budgets, and async scheduling/CUDAGraph compatibility.

The public DDTree site reports speedups relative to autoregressive decoding, not
DDTree-vs-DFlash deltas. One visible example is HumanEval on Qwen3-30B-MoE at
T=0.0: 8.22x for DDTree and 6.09x for DFlash relative to autoregressive
decoding, which is a +35.0% DDTree-over-DFlash delta after normalization. The
new C=4 budget-32 result is in that range, and budget 64 is higher on this
synthetic profile. Higher-concurrency serving is still lower than that because
this vLLM path pays dynamic metadata, KV compaction, and tree-build overheads
inside a batched serving loop.

Earlier clean-image single-request smoke numbers, before batched accepted-KV
compaction, were:

| Mode | Budget | Tokens | Tokens/s | Delta vs DFlash | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| Target only, `TREE_ATTN` | n/a | 48 | 13.64 | n/a | Baseline |
| DFlash | n/a | 48 | 6.04 | n/a | Exact match with target/TREE |
| DDTree+DFlash | 32 | 48 | 8.59 | +42.3% | Exact match with target/TREE and DFlash |
| DFlash | n/a | 128 | 17.08 | n/a | Baseline for longer smoke |
| DDTree+DFlash | 64 | 128 | 20.90 | +22.4% | Exact match with DFlash |

The new batched compaction result is directionally consistent with the DDTree
algorithm: once accepted target KVs are retained, the verifier pass does useful
work beyond sampling and avoids recomputing the accepted path.

## Workload Profile Gap

The K2.5 `perfy` no-delay Rebench profile is materially different from the
current DDTree smoke tests.

From `/Users/aeldeib/code/claude/perfy/scenarios/k2_5_chat`:

- Dataset: `CoreWeave/rebench-playback` Mooncake trace
- Replay shape: full-session sequential replay
- Sessions: 48 conversations
- Turns: 2,025 requests
- Delay handling: `REBENCH_ZERO_DELAYS=1` rewrites trace delays to zero
- Concurrency sweep used for the accepted no-delay run: `4 8 16 32 48`
- AIPerf settings: streaming, server token counts, record-level export,
  `--custom-dataset-type mooncake-trace`, `--dataset-sampling-strategy sequential`
- Runtime in that K2.5 report: `FLASHINFER_MLA`, `HND` KV layout, TRT-LLM
  ragged MLA prefill, prefix caching, and `kv_cache_dtype=fp8_e4m3`

That workload matters because it keeps the full conversation depth. The older
request-count-bounded Rebench rows sampled shallow prefixes at high concurrency;
the no-delay full replay reaches late turns with large cached context. That is
the right workload shape for K2.5 serving economics.

What changed with this patch:

- The old single-active-request DDTree limitation is removed for standard
  full-attention `TREE_ATTN` verification.
- A Qwen3-style batched DDTree no-delay comparison now runs at C=4, C=8, and
  C=16 in the offline harness, with positive DDTree-over-DFlash deltas for the
  best tested budget at each concurrency.
- The actual K2.5 no-delay sweep is still blocked because K2.5 uses MLA/HND/fp8
  KV cache, not the standard full-attention `TREE_ATTN` cache path implemented
  here.

The closest honest workload ladder is now:

1. Run a server-backed Qwen3 full-trace replay shape with `max_num_seqs` swept
   through the AIPerf-style range, comparing DFlash vs DDTree+DFlash with the
   same prompt trace.
2. Keep stage timers and acceptance counters enabled for selected samples to
   decompose draft forward, tree build, verifier forward, sampler walk, KV
   compaction, and next-draft prep.
3. Validate the public Kimi K2.5 DFlash drafter with DFlash alone.
4. Implement MLA tree verification and MLA accepted-KV compaction.
5. Then rerun the real K2.5 AIPerf no-delay sweep at `C=4,8,16,32,48`.

## Upstream Parity

We now match more of the upstream/local DDTree performance architecture than the
earlier correctness-first branch.

| Area | Upstream/local prototype | Current vLLM branch | Status |
| --- | --- | --- | --- |
| DFlash proposal logits | Uses DFlash logits | Uses DFlash logits | Matched for Qwen3 DFlash path |
| Tree construction | Heap-based prefix-closed tree after top-k | Same algorithm; top-k/logprob transfer is batched across requests | Mostly matched, still CPU heap |
| Tree visibility | Root/ancestor/self mask | Batched 3D per-request dense qq-bias, now built on CPU and copied once per batch | Matched for full attention |
| Target verification | One target pass over tree | One target pass over batched trees through `TREE_ATTN` | Matched for full attention |
| Greedy tree walk | Greedy walk over target posterior | Greedy walk, now records accepted node indices | Matched for greedy |
| Accepted KV reuse | Retains accepted verified nodes | Compacts accepted nodes from scratch paged-KV slots to canonical slots | Matched for standard full-attention, no context parallelism |
| Next proposal | Reuses the accepted path immediately | Re-proposes immediately from the actual accepted tree-node path | Matched for standard full-attention greedy path |
| Serving batch | Prototype is offline/single sequence | vLLM batch support for standard full attention | Improved beyond prototype envelope |
| MLA / fp8 KV | Not the prototype focus | Not implemented | Missing for Kimi/GLM |
| Stochastic sampling | Not covered by current branch | Not implemented | Missing |
| Logprobs | Not covered by current branch | Not implemented | Missing |

The remaining CPU sync is now less severe: tree construction still moves top-k
data to CPU for heap expansion, but it does so once per batch rather than once
per request. Dense tree-bias construction no longer launches many small GPU
writes. A fully GPU-resident tree builder and sampler would still help,
especially for larger budgets and higher concurrency, but the larger upstream
parity gap, accepted-KV retention, is now closed for the standard full-attention
path.

## Current Architecture Limits

The current implementation is production-shaped for Qwen3-style full-attention
greedy DDTree on one context-parallel rank. It is not yet a universal DDTree
implementation across all vLLM model families.

Current limitations:

- Greedy decoding only.
- Logprobs are not implemented.
- Async scheduling is disabled for DDTree.
- CUDAGraph/compile is still effectively disabled for dynamic DDTree execution
  in the validated path.
- Target model must use `attention_backend="TREE_ATTN"`.
- Accepted-KV compaction supports standard full-attention 5D KV cache tensors
  shaped like `(2, num_blocks, block_size, num_kv_heads, head_size)`.
- Accepted-KV compaction is disabled when context parallelism is active because
  compaction may need cross-rank KV movement.
- `TREE_ATTN` supports `auto`, `float16`, and `bfloat16` KV cache dtypes, not
  fp8 KV cache.
- Dynamic tree attention bias is dense and rebuilt per step as a host buffer
  with one host-to-device copy per batch.
- M-RoPE, XD-RoPE, multimodal inputs, tensor parallelism, expert parallelism,
  MoE routing, and hybrid stateful attention under DDTree are not validated.
- Hybrid stateful attention models are not supported. The Qwen3.5 hybrid
  GDN/linear-attention path already failed in earlier DDTree validation.

MoE itself is not necessarily the hard blocker. The harder blockers are the
attention/cache architecture, distributed cache movement, and whether a
compatible DFlash drafter exists and is validated for the target model.

## Kimi K2.5 and Kimi K2.6

Kimi K2.5 and Kimi K2.6 are still outside the current DDTree support envelope.

Relevant current facts:

- Kimi K2.5 is a native multimodal MoE model with 1T total parameters, 32B
  activated parameters, 61 layers, 384 experts, 8 selected experts per token,
  256K context, MoonViT, and MLA attention.
- Kimi K2.6 uses the same architecture as Kimi K2.5, so the inference-stack
  blockers should be the same.
- vLLM has Kimi K2.5 model support and recipes for GB200/aarch64 deployment.
- There is a public Kimi K2.5 DFlash drafter:
  `z-lab/Kimi-K2.5-DFlash`.
  Its config advertises `architectures=["DFlashDraftModel"]`, target model
  `moonshotai/Kimi-K2.5`, block size 8, 6 draft layers, hidden size 7168, 61
  target layers, `mask_token_id=163838`, and target hidden layer IDs
  `[1, 12, 24, 35, 47, 58]`.
- The vLLM draft model registry maps `DFlashDraftModel` to the existing
  DFlash implementation. The Kimi drafter config uses `model_type="qwen3"` and
  Qwen3-style DFlash custom code, so it is plausibly compatible with the
  existing DFlash path. This still needs real load/run validation against
  `moonshotai/Kimi-K2.5`.

What Kimi support would take:

1. Validate and wire the public Kimi-compatible DFlash drafter.

   DDTree uses DFlash logits as its proposal distribution. The public
   `z-lab/Kimi-K2.5-DFlash` checkpoint likely removes the need to train or
   obtain a Kimi K2.5 drafter. The remaining work is to verify that vLLM can
   load it through the `DFlashDraftModel` path, target hidden states are
   gathered from the intended Kimi layers, tensor-parallel partitioning works,
   and DFlash alone is correct and performant before layering DDTree on top.

2. Add tree verification to MLA attention.

   Kimi's production path uses MLA, not standard full attention. The current
   DDTree verifier requires `TREE_ATTN`, whose KV cache shape and supported KV
   dtypes do not match Kimi's MLA path. We need the root/ancestor/self
   visibility semantics in an MLA backend such as FlashInfer MLA, FlashMLA, or
   the selected production Kimi backend.

3. Support Kimi KV cache layout and dtype.

   The K2.5 serving report used HND KV layout and fp8 KV cache. DDTree needs a
   way to write scratch verified tree nodes and then retain or compact accepted
   nodes in the MLA cache format.

4. Support distributed serving.

   K2.5 GB200 recipes use multi-GPU tensor parallelism and may use parallelism
   combinations where accepted-KV compaction needs rank-aware movement. DDTree
   metadata, verifier logits, accepted-token sampling, and drafter hidden states
   all need to be correct under TP and MoE routing.

5. Start text-only.

   Kimi is multimodal, but the shortest path is text-only Kimi DDTree first.
   Image/video handling can follow once text-only MLA, TP, and batching work.

Practical estimate: Kimi support is still a medium-to-large project, not a small
model registration change. The public Kimi K2.5 DFlash drafter removes one major
dependency for K2.5. The hard work is now MLA tree verification, MLA KV
retention/compaction, TP, and validation under the Rebench-shaped workload. Kimi
K2.6 has the same architecture, but the K2.5 drafter should not be assumed to
give the same acceptance on K2.6 weights; it can be tried as a bootstrap, but a
K2.6-tuned drafter or acceptance/performance validation is still required.

## GLM-5.1

GLM-5.1 is also outside the current support envelope and is probably at least as
hard as Kimi, possibly harder.

Relevant current facts:

- GLM-5.1 uses the `glm_moe_dsa` architecture.
- Public model coverage describes it as MoE plus DeepSeek-style MLA and DSA
  (Dynamic Sparse Attention).
- The model architecture is `GlmMoeDsaForCausalLM`.
- It has 78 layers, hidden size 6144, 256 routed experts, 8 active experts per
  token, KV compression, and about a 200K context window.

What GLM-5.1 support would take:

1. Confirm the exact target model config and serving backend we plan to use.
2. Build or obtain a GLM-compatible DFlash drafter.
3. Add DDTree verification semantics to the GLM MLA + DSA attention path.
4. Define scratch, accepted-node retention, and compaction for both MLA KV state
   and DSA/indexer state.
5. Validate MoE routing, TP/EP behavior, and distributed sampler correctness.
6. Run DFlash-only correctness/performance before adding DDTree.

The DSA part is the warning sign. DDTree is straightforward when a token's
future state is just full-attention KV. With DSA or other stateful/hybrid
attention, accepted tree nodes may carry additional per-layer state that must be
written, selected, compacted, and replayed exactly. That is the same class of
problem that made the Qwen3.5 hybrid GDN/linear-attention target fall outside
the current support envelope.

## Recommendation

Short term:

1. Keep this vLLM branch focused on Qwen3/full-attention DDTree and land the
   batched accepted-KV implementation cleanly.
2. Keep using the current multi-arch `vllm-tensorizer` image for clean-image
   review and validation.
3. Run a Qwen3 no-delay Rebench-shaped server comparison for DFlash vs
   DDTree+DFlash across the full C sweep.
4. Use the new stage timers and acceptance counters to choose budget defaults per
   workload instead of assuming one static budget is best everywhere.

Medium term:

1. Preallocate/reuse DDTree metadata buffers and benchmark non-eager execution
   once the dynamic shapes are bucketed.
2. Move tree expansion and/or tree walk closer to GPU-resident execution if
   timers show CPU sync is material after compaction.
3. Validate non-eager execution and decide whether DDTree can safely use
   compile/CUDAGraph bucketing for common tree budgets.
4. Add stochastic sampling and logprob support or enforce DDTree eligibility
   earlier per request.

Stretch models:

1. Kimi K2.5 is the more natural first stretch target because a public DFlash
   drafter exists. The architecture is MLA/MoE, but not DSA.
2. GLM-5.1 should follow after the MLA path is solved, because DSA adds another
   state-management surface.

## References

- DDTree public write-up: https://liranringel.github.io/ddtree/
- Local DDTree prototype: `/Users/aeldeib/code/claude/ddd/ddtree`
- K2.5 Rebench scenario: `/Users/aeldeib/code/claude/perfy/scenarios/k2_5_chat`
- Kimi K2.5 model card: https://huggingface.co/moonshotai/Kimi-K2.5
- Kimi K2.5 DFlash drafter: https://huggingface.co/z-lab/Kimi-K2.5-DFlash
- Kimi K2.6 model card: https://huggingface.co/moonshotai/Kimi-K2.6
- vLLM Kimi K2.5 recipe: https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html
- GLM-5.1 model card: https://huggingface.co/zai-org/GLM-5.1
- NVIDIA GLM-5/5.1 model coverage: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/llm/thudm/glm5-moe-dsa.html
