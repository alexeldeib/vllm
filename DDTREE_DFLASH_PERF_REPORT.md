# DDTree + DFlash Performance and Architecture Report

## Scope

This report focuses on the current vLLM DDTree integration as a DFlash
extension: correctness evidence, the measured DFlash vs DDTree+DFlash delta, why
the current workload profile is not yet representative of K2.5 serving, what is
missing relative to the upstream DDTree prototype, and what blocks Kimi K2.5,
Kimi K2.6, and GLM-5.1.

The detailed build log and clean-image validation remain in
`DDTREE_VLLM_REPORT.md`. This file is the shorter performance and architecture
readout.

## Current State

- vLLM branch: `alex/ddtree-vllm-integration`
- Pushed review branch: `alexeldeib/vllm:alex/ddtree-vllm-integration`
- Container branch: `coreweave/ml-containers:alex-ddtree-vllm-image`
- Clean image:
  `ghcr.io/coreweave/ml-containers/vllm-tensorizer:alex-ddtree-vllm-image-be7262e-74f6f52790887fb0729dde13928a12cb3d0a2dad`
- Clean image manifest digest:
  `sha256:425f069eefb838918cf77e09e16d3960e165fb5784ad183da197c5364aecd1be`
- Platforms: `linux/amd64`, `linux/arm64`
- Clean GB200 validation pod: `ddtree-vllm-ci`
- Cluster and namespace: `cw4637-dev-us-e-01a`, `ace-inference`
- Runtime architecture: `aarch64`
- vLLM package from image: `0.1.dev16269+g74f6f5279.d20260502`

The implementation is correctness-first:

- It adds speculative method `ddtree`.
- It uses the DFlash drafter to generate parallel draft logits.
- It builds a prefix-closed DDTree from those logits.
- It verifies the dynamic tree in one target forward pass through `TREE_ATTN`.
- It greedily walks the verified tree and emits the accepted path plus fallback.

## Correctness Evidence

All current correctness checks use greedy decoding, Qwen3-8B target weights, and
`z-lab/Qwen3-8B-DFlash-b16` as the drafter.

| Environment | Check | Result |
| --- | --- | --- |
| Monkeypatch image | Import and unit smoke | Passed |
| Monkeypatch image | 48-token target/TREE vs DFlash vs DDTree | Exact token ID match |
| Monkeypatch image | 128-token DFlash vs DDTree, budgets 16, 32, 64 | Exact token ID match with DFlash |
| Clean CI image | Import `vllm.v1.spec_decode.ddtree.DDTreeProposer` | Passed |
| Clean CI image | 48-token target/TREE vs DFlash vs DDTree, budget 32 | Exact token ID match |
| Clean CI image | 128-token DFlash vs DDTree, budget 64 | Exact token ID match with DFlash |

The 128-token smoke had one important caveat: DFlash and DDTree diverged
together from the non-spec target at the same token. Since DDTree exactly matched
DFlash in that run, this looks like an existing DFlash/spec baseline behavior in
the eager single-request harness rather than a DDTree-specific correctness
issue.

## Measured Performance

These are smoke-test numbers, not serving economics. They use one prompt,
`max_num_seqs=1`, and eager execution on GB200. They are useful for integration
directionality, but they should not be used as final workload claims.

Clean image on `ddtree-vllm-ci`:

| Mode | Budget | Tokens | Tokens/s | Delta vs DFlash | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| Target only, `TREE_ATTN` | n/a | 48 | 13.64 | n/a | Baseline |
| DFlash | n/a | 48 | 6.04 | n/a | Exact match with target/TREE |
| DDTree+DFlash | 32 | 48 | 8.59 | +42.3% | Exact match with target/TREE and DFlash |
| DFlash | n/a | 128 | 17.08 | n/a | Baseline for longer smoke |
| DDTree+DFlash | 64 | 128 | 20.90 | +22.4% | Exact match with DFlash |

Earlier monkeypatch validation on the v0.20.0-derived image showed the same
direction:

| Mode | Budget | Tokens | Tokens/s | Delta vs DFlash |
| --- | ---: | ---: | ---: | ---: |
| DFlash | n/a | 48 | 7.30 | n/a |
| DDTree+DFlash | 32 | 48 | 8.93 | +22.4% |
| DFlash | n/a | 128 | 18.71 | n/a |
| DDTree+DFlash | 64 | 128 | 20.84 | +11.4% |

The result is directionally good: DDTree is already faster than DFlash in the
single-request harness. It is not yet the DDTree paper/site result shape because
we have not implemented the performance-critical KV reuse path, and we are not
benchmarking a serving workload yet.

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

The current DDTree implementation cannot run that serving sweep yet. In
`gpu_model_runner.py`, DDTree explicitly raises if more than one request is
scheduled in a verification step:

```text
DDTree currently supports one scheduled request per step.
```

So a C=4,8,16,32,48 AIPerf comparison would currently be invalid or would fail.
The closest honest apples-to-apples profile today is:

1. Replay the same Rebench no-delay trace shape at C=1, or run an offline
   sequential full-trace harness over the same 2,025 turns.
2. Compare DFlash vs DDTree+DFlash with only the speculative method and tree
   budget changed.
3. Record token equality for a fixed greedy configuration.
4. Collect per-step timers for drafter forward, tree build, verifier forward,
   sampler walk, accepted length, and catch-up/recompute.
5. After batched DDTree lands, repeat the real AIPerf no-delay sweep at
   C=4,8,16,32,48.

## Upstream Parity

We matched the algorithmic skeleton from the upstream/local DDTree prototype:

- DFlash block logits
- Heap-based prefix-closed tree construction
- Root/ancestor/self tree visibility
- One target verification pass
- Greedy target walk

We have not matched the performance architecture of the prototype.

| Area | Upstream/local prototype | Current vLLM branch | Impact |
| --- | --- | --- | --- |
| Tree construction | CPU heap after top-k logits copied to CPU | Same broad approach | CPU sync exists, but this is not the main parity gap |
| Greedy tree walk | Converts target posterior to Python list | Same broad approach | CPU sync exists, but likely secondary |
| Buffer reuse | Preallocates verify input, positions, and tree visibility buffers | Rebuilds dynamic bias and metadata per step | Avoidable overhead |
| Accepted KV reuse | Compacts target `past_key_values` to keep accepted verified nodes | Verifies into scratch slots, then recomputes accepted tokens canonically | Major missing DDTree speedup |
| Cache compaction implementation | Has Python compaction plus optional inline C++ tail compaction | Not implemented for vLLM paged KV cache | Major parity gap |
| Serving batch support | Prototype is offline/single sequence | vLLM path is single active request | Blocks real serving benchmarks |

The CPU sync is real and worth removing, but it is not the reason we are below
the expected upstream DDTree gains. The upstream prototype also synchronizes for
top-k tree construction and tree following. The larger gap is that upstream
retains the target KVs for the accepted path; our vLLM path throws away that
advantage by verifying in scratch slots and then doing canonical catch-up.

## Stacked Optimization Plan

| Step | What changes | Why it matters | Expected value |
| --- | --- | --- | --- |
| Current branch | DDTree+DFlash with scratch verification | Already proves correctness and direction | +22.4% vs DFlash on clean 128-token smoke; +42.3% on clean 48-token smoke |
| Apples-to-apples C=1 Rebench | Use full no-delay trace shape without serving concurrency | Replaces toy prompt with realistic prompt growth and output caps | Determines whether gains survive real turn distribution |
| Stage timers | Add timers/counters for draft, tree build, verify, sample, accepted length, catch-up | Shows which overhead dominates in vLLM | Required before optimizing blindly |
| TREE_ATTN baseline control | Compare DFlash under the same target backend constraints when possible | Separates DDTree benefit from backend tax | Makes the comparison cleaner |
| Preallocate tree buffers | Reuse attention bias/input/position buffers per runner | Removes obvious allocation overhead | Modest, low risk |
| GPU/async tree build and sampler | Keep top-k expansion and target walk on GPU, or overlap CPU work | Removes CPU sync stalls | Useful, but not sufficient for upstream parity |
| Accepted KV reuse | Compact/remap accepted verified target KVs into canonical cache slots | Avoids recomputing tokens DDTree already verified | High impact; required for real DDTree speedup |
| Batched DDTree | Carry per-request tree metadata and masks for multiple active requests | Enables serving workloads and AIPerf concurrency sweeps | Required for production relevance |
| MLA/DSA tree verification | Implement tree visibility in MLA/DSA attention backends and cache layouts | Enables Kimi and GLM families | Required for stretch models |

## Current Architecture Limits

The current branch is intentionally narrow. It supports a full-attention,
greedy, single-request path and has not been generalized to the large
MLA/MoE/hybrid models we care about next.

Current limitations:

- Greedy decoding only.
- Logprobs are not implemented.
- One active request per DDTree verification step.
- Async scheduling is disabled for DDTree.
- Target model must use `attention_backend="TREE_ATTN"`.
- Drafter path is DFlash-based and currently tied to the Qwen3 DFlash draft
  model implementation.
- `TREE_ATTN` supports standard full-attention KV cache shape, not MLA cache
  layout.
- `TREE_ATTN` supports `auto`, `float16`, and `bfloat16` KV cache dtypes, not
  fp8 KV cache.
- Dynamic tree attention bias is dense and rebuilt per step.
- Verified target KVs are not retained for the accepted path.
- M-RoPE, XD-RoPE, multimodal inputs, tensor parallelism, expert parallelism,
  and MoE routing under DDTree are not validated.
- Hybrid stateful attention models are not supported. The Qwen3.5 hybrid
  GDN/linear-attention path already failed in multi-node DDTree validation.

MoE itself is not necessarily the hard blocker. The harder blockers are the
attention/cache architecture, serving batch shape, and whether a compatible
DFlash drafter exists.

## Kimi K2.5 and Kimi K2.6

Kimi K2.5 and Kimi K2.6 are outside the current DDTree support envelope.

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
- Our local vLLM speculative config already lists `kimi_k2` and `kimi_k25` in
  the aux-hidden-state supported set for DFlash/DDTree-style methods. That means
  hidden-state plumbing is probably not the first blocker.
- The vLLM draft model registry maps `DFlashDraftModel` to the existing
  `DFlashQwen3ForCausalLM` implementation. The Kimi drafter config uses
  `model_type="qwen3"` and Qwen3-style DFlash custom code, so it is plausibly
  compatible with the existing DFlash draft-model path. This still needs a real
  load/run validation against `moonshotai/Kimi-K2.5`.

What Kimi support would take:

1. Validate and wire the public Kimi-compatible DFlash drafter.

   DDTree uses DFlash logits as its proposal distribution. The public
   `z-lab/Kimi-K2.5-DFlash` checkpoint likely removes the need to train or
   obtain a Kimi K2.5 drafter. The remaining work is to verify that vLLM can
   load it through the `DFlashDraftModel` registry path, that target hidden
   states are gathered from the intended Kimi layers, that tensor-parallel
   partitioning works, and that DFlash alone is correct and performant before
   layering DDTree on top.

2. Add tree verification to MLA attention.

   Kimi's production path uses MLA, not standard full attention. The current
   DDTree verifier requires `TREE_ATTN`, whose KV cache shape and supported KV
   dtypes do not match Kimi's MLA path. We need the root/ancestor/self visibility
   semantics in an MLA backend such as FlashInfer MLA, FlashMLA, or another
   selected Kimi backend.

3. Support Kimi KV cache layout and dtype.

   The K2.5 serving report used HND KV layout and fp8 KV cache. DDTree needs a
   way to write scratch verified tree nodes and then retain or compact accepted
   nodes in the MLA cache format.

4. Support tensor/expert parallel serving.

   K2.5 GB200 recipes use multi-GPU tensor parallelism. DDTree metadata,
   verifier logits, accepted-token sampling, and drafter hidden states all need
   to be correct under TP and MoE routing.

5. Add batched DDTree.

   K2.5 serving economics are measured with concurrent chat requests. A
   single-active-request implementation cannot validate or ship that workload.

6. Start text-only.

   Kimi is multimodal, but the shortest path is text-only Kimi DDTree first.
   Image/video handling can follow once text-only MLA, TP, and batching work.

Practical estimate: Kimi support is still a medium-to-large project, not a small
model registration change. The public Kimi K2.5 DFlash drafter removes one major
dependency for K2.5. The hard work is now MLA tree verification, KV retention,
TP, batching, and validation under the Rebench-shaped workload. Kimi K2.6 has
the same architecture, but the K2.5 drafter should not be assumed to give the
same acceptance on K2.6 weights; it can be tried as a bootstrap, but a K2.6-tuned
drafter or acceptance/performance validation is still required.

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
4. Define cache scratch, accepted-node retention, and compaction for both MLA KV
   state and DSA/indexer state.
5. Validate MoE routing, TP/EP behavior, and distributed sampler correctness.
6. Add batched DDTree before any serving-level Rebench/AIPerf claim.

The DSA part is the warning sign. DDTree is straightforward when a token's
future state is just full-attention KV. With DSA or other stateful/hybrid
attention, accepted tree nodes may carry additional per-layer state that must be
written, selected, compacted, and replayed exactly. That is the same class of
problem that made the Qwen3.5 hybrid GDN/linear-attention target fall outside
the current support envelope.

## Recommendation

Short term:

1. Keep the current branch scoped to Qwen3 full-attention correctness.
2. Run a full-depth no-delay Rebench-shaped C=1/offline comparison for DFlash vs
   DDTree+DFlash.
3. Add stage timers and acceptance counters before doing more performance work.
4. Implement accepted KV reuse/compaction before spending much time on GPU-only
   tree build optimizations.

Medium term:

1. Add batched DDTree for standard full attention.
2. Rerun the real AIPerf no-delay sweep at C=4,8,16,32,48.
3. Compare stacked deltas: DFlash baseline, DDTree scratch, DDTree with buffer
   reuse, DDTree with GPU/async tree build, DDTree with accepted KV reuse, and
   DDTree batched.

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
