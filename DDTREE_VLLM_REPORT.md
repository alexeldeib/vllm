# DDTree vLLM Integration Report

## Summary

This branch adds a DDTree speculative decoding path on top of vLLM's existing
DFlash support. DDTree uses one DFlash parallel draft pass to build a
prefix-closed draft tree, verifies the dynamic tree in one target-model forward
pass with `TREE_ATTN`, greedily walks the verified tree to emit the accepted
target path plus the first fallback token, and now compacts accepted target KVs
from scratch tree slots into canonical paged-KV slots for the standard
full-attention path. The current branch also immediately re-proposes the next
DDTree from the actual accepted tree-node path, avoiding the earlier
correctness-first catch-up step. Mixed scheduler steps with DDTree verification
rows and ordinary prefill rows are handled by preserving the accepted DDTree path
for tree rows and contiguous prompt hidden states for non-tree rows.

The implementation was first monkey-patched into the existing
`docker.cloudsmith.io/coreweave/infr/vllm:v2.10.0` image and validated on a GB200
pod in `cw4637-dev-us-e-01a`, namespace `ace-inference`. It was then built into
a multi-arch `vllm-tensorizer` image through `ml-containers` CI and revalidated
from a clean image on GB200. The current clean image includes the batched DDTree
verification, immediate re-proposal, mixed-batch drafter prep, and accepted-KV
compaction fixes, plus the latest debug/timer and benchmark harness updates.

## Algorithm

The DDTree algorithm follows the reference implementation in
https://liranringel.github.io/ddtree/ and the local prototype under
`/Users/aeldeib/code/claude/ddd/ddtree`:

- Run the DFlash drafter once to get per-position draft logits for the next
  block.
- Build a prefix-closed dynamic tree with a fixed node budget by expanding the
  highest cumulative draft-logprob nodes from a heap.
- Verify root plus all draft-tree nodes in a single target-model forward pass.
  Tree nodes use unique scratch KV slots, while model positions are based on
  tree depth so siblings share the correct RoPE position.
- Apply a tree attention bias where each node can see the root, itself, and its
  ancestors only.
- Greedily walk from the root through matching child tokens. The first target
  token that leaves the tree is emitted as the fallback token.

The public DDTree write-up reports that DDTree is distribution-preserving because
the target model still chooses the accepted path and fallback token. It also
shows that speedup improves with tree budget until verifier cost dominates, so
this branch defaults the DDTree node budget to `4 * num_speculative_tokens`
instead of the DFlash horizon itself.

## vLLM Changes

- Added speculative method `ddtree`, with `ddtree_tree_budget` and an effective
  default of `4 * num_speculative_tokens`.
- Added `vllm/v1/spec_decode/ddtree.py` with:
  - dynamic DDTree heap builder,
  - node-depth metadata,
  - batched dynamic tree attention bias construction,
  - greedy DDTree verifier walk,
  - accepted node-index reporting for KV compaction,
  - `DDTreeProposer`, implemented as a DFlash proposer that converts draft
    logits into tree nodes.
- Extended scheduler and request state to carry method-specific speculative
  metadata from drafter output to verifier scheduling.
- Extended `TREE_ATTN` metadata to accept a dynamic tree attention bias for the
  current verification step.
- Updated the GPU model runner to:
  - require target `attention_backend="TREE_ATTN"` for DDTree,
  - keep the DFlash drafter on `FLASH_ATTN`,
  - schedule batched DDTree scratch slots separately from model positions,
  - compute logits for root plus every tree node,
  - sample with the DDTree greedy walk,
  - compact accepted target KVs into canonical slots,
  - prepare the next DFlash/DDTree proposal from non-contiguous accepted tree
    nodes,
  - keep ordinary prefill rows contiguous when they share a batch with DDTree
    verification rows,
  - disable async scheduling for DDTree.
- Added focused unit coverage in `tests/v1/spec_decode/test_ddtree.py`.

## Correctness Results

Monkeypatch environment:

- Cluster: `cw4637-dev-us-e-01a`
- Namespace: `ace-inference`
- Test pod: `ddtree-batch-vllm-test`
- GPU class: GB200
- Base image: `docker.cloudsmith.io/coreweave/infr/vllm:v2.10.0`
- Target model: `Qwen/Qwen3-8B`
- Draft model: `z-lab/Qwen3-8B-DFlash-b16`
- Sampling: `temperature=0.0`
- Target backend for DDTree and target/TREE baseline: `TREE_ATTN`
- Drafter backend: `FLASH_ATTN`

Passing checks:

| Check | Result |
| --- | --- |
| Local syntax check | `py_compile` passed for touched Python files |
| Local whitespace check | `git diff --check` passed |
| Current batched DDTree direct unit smoke | Passed on GB200 monkeypatch pod |
| Current 3D qq-bias Triton kernel check | `max_diff 0.0`, passed |
| Current 4-request batched DDTree generate with accepted-KV compaction and immediate re-proposal | Passed |
| Current target-only greedy vs DDTree+DFlash greedy on selected stable prompts | Exact token-id match |
| Pod import/unit smoke | DDTree builder, depth ordering, parent links, attention bias passed |
| 48-token target/TREE vs DFlash vs DDTree | Exact token-id match |
| 48-token DDTree with no explicit `ddtree_tree_budget` | Exact token-id match with target/TREE |
| 128-token DFlash vs DDTree, budget 16 | Exact token-id match |
| 128-token DFlash vs DDTree, budget 32 | Exact token-id match |
| 128-token DFlash vs DDTree, budget 64 | Exact token-id match |

Clean CI image validation:

- `ml-containers` branch: `alex-ddtree-vllm-image`
- `ml-containers` commit: `7cf4eae7e6c1458928edbcf356fef24a4db66c6b`
- vLLM commit: `6acdb096fea8723d337292251068ee5928ce00a2`
- CI run: `https://github.com/coreweave/ml-containers/actions/runs/25264137128`
- Image:
  `ghcr.io/coreweave/ml-containers/vllm-tensorizer:alex-ddtree-vllm-image-7cf4eae-6acdb096fea8723d337292251068ee5928ce00a2`
- Manifest index digest:
  `sha256:05a12a894d2a548ba7d9182eb2cab72feeca6a07cf1956e62950a8026060b21f`
- Platforms: `linux/amd64`, `linux/arm64`
- Platform manifests:
  - `linux/amd64`: `sha256:aaf31d599ac88c9404d3ff91b20781c11affdd1b25ff12949adfc5543f1d9b17`
  - `linux/arm64`: `sha256:ce4dd39d65a0889b1e305587a3c708cf94bfad2c4830cb02936f2052451c3faa`
- Clean-image test pod: `ddtree-vllm-7cf4eae-gb200`
- Runtime architecture: `aarch64`
- vLLM package version:
  `0.1.dev16276+g6acdb096f.d20260502`

| Clean image check | Result |
| --- | --- |
| Installed source syntax | `py_compile` passed for modified installed files |
| Import/version check | vLLM package and DDTree source loaded from the clean image on `aarch64` |
| DDTree debug metrics hook | `DDTreeProposer.take_last_ddtree_debug_metrics()` present |
| Single stable prompt, target/TREE vs DFlash vs DDTree budget 32 | Exact token-id match over 48 generated tokens |
| 4-prompt DFlash vs DDTree budget 32 generate | Completed; one low-margin prompt selected a different target-verified continuation |
| Budget sensitivity on low-margin median prompt | Budget 8 matched DFlash; budget 32 chose a different verified branch |
| C=4 short no-debug timing smoke | Passed; DDTree budget 32/64 was faster than DFlash on the short run |

Earlier clean-image checks, from the pre-compaction CI image, also passed
48-token target/TREE vs DFlash vs DDTree and 128-token DFlash vs DDTree
correctness. The current image supersedes those results.

Observed baseline caveats:

- In the 128-token smoke, both DFlash and DDTree diverged from the non-spec
  target/TREE baseline at token index 112, with the same token selected by
  DFlash and DDTree. This appears to be an existing DFlash/speculative baseline
  behavior under the eager single-request harness, not a DDTree-specific
  divergence.
- In the latest clean image, a median-function prompt exposed low-margin greedy
  sensitivity: target-only, DFlash, DDTree budget 8, and DDTree budget 32 do not
  all pick byte-identical continuations. DFlash itself can differ from
  target-only, and larger DDTree budgets can change which target-verified branch
  is reached before fallback. Treat exact token identity on small low-margin
  prompt sets as a useful smoke signal, not a complete correctness proof.

Unsupported or not yet validated:

- `Qwen/Qwen3.5-4B` hybrid GDN/linear-attention target path failed during
  multi-node DDTree verification and is not included in the supported envelope.
- Budget 8 was not accepted into the validated envelope; on the 128-token smoke
  it diverged from the DFlash baseline at token index 71. The branch therefore
  uses budget 32 by default when `num_speculative_tokens=8`.

## Performance Results

Current focused results are in `DDTREE_DFLASH_PERF_REPORT.md`. The corrected
headline is no longer the earlier +5.8% warmed smoke. That smoke used a small
64-token prompt shape, no tree-budget sweep, and the older GPU-side dense-bias
builder. It was useful as a clean-image integration check, but not as an
algorithmic performance result.

The current GB200 no-delay offline sweep uses Qwen3-8B, `z-lab/Qwen3-8B-DFlash-b16`,
`TREE_ATTN` for every target verifier, `enforce_eager=True`,
`max_num_batched_tokens=8192`, `max_tokens=128`, and async scheduling disabled
for both DFlash and DDTree. The latest branch source was monkeypatched into the
validation pod for this sweep; the same source is now built into the clean image
listed above.

| Concurrency | Mode | Budget | Output tokens/s | Delta vs DFlash |
| ---: | --- | ---: | ---: | ---: |
| 4 | DFlash, `TREE_ATTN` target | n/a | 281.14 | n/a |
| 4 | DDTree+DFlash | 32 | 380.84 | +35.5% |
| 4 | DDTree+DFlash | 64 | 401.31 | +42.7% |
| 8 | DFlash, `TREE_ATTN` target | n/a | 525.25 | n/a |
| 8 | DDTree+DFlash | 32 | 590.20 | +12.4% |
| 16 | DFlash, `TREE_ATTN` target | n/a | 1,020.30 | n/a |
| 16 | DDTree+DFlash | 32 | 1,161.69 | +13.9% |

The final clean image also passed a shorter C=4, one-batch, 64-token no-debug
smoke:

| Mode | Budget | Output tokens/s | Delta vs DFlash |
| --- | ---: | ---: | ---: |
| DFlash, `TREE_ATTN` target | n/a | 139.16 | n/a |
| DDTree+DFlash | 32 | 248.42 | +78.5% |
| DDTree+DFlash | 64 | 268.32 | +92.8% |

That short run validates the image path, but it is not the headline number
because DFlash was unusually low on the small sample.

The timer run at C=4 shows why the result moved: after changing dense tree-bias
construction from many small GPU writes to CPU construction plus one device
copy, `ddtree_attention_bias` is about 0.48 ms at budget 32 and 0.92 ms at
budget 64. Mean acceptance length rises from 2.73 for DFlash to 4.06 for
DDTree budget 32 and 4.29 for DDTree budget 64. The remaining costs to optimize
are target/drafter scheduling, accepted-KV compaction, CPU tree build at larger
budgets, and non-eager execution.

The public DDTree site reports speedups relative to autoregressive decoding, not
DDTree-vs-DFlash deltas. Its visible HumanEval Qwen3-30B-MoE T=0.0 example is
8.22x for DDTree and 6.09x for DFlash relative to autoregressive decoding,
which normalizes to a +35.0% DDTree-over-DFlash delta. The current C=4 budget-32
vLLM result is now in that range; higher-concurrency serving remains more
overhead-sensitive and needs the Rebench-shaped server sweep before making SLO
claims.

## Current Limitations

- Greedy decoding only. Non-greedy sampling and logprobs raise
  `NotImplementedError`.
- Batched DDTree is implemented for standard full-attention `TREE_ATTN`, but not
  yet validated under the full Rebench/AIPerf serving sweep.
- Target model must use `attention_backend="TREE_ATTN"`.
- Async scheduling is disabled for DDTree.
- Accepted-KV compaction supports standard full-attention 5D KV cache tensors and
  is disabled when context parallelism is active.
- fp8 KV cache, MLA cache layouts, and DSA/stateful attention cache layouts are
  not supported by the current DDTree verifier.
- M-RoPE/XD-RoPE and multimodal targets are not validated.
- Hybrid linear-attention/GDN target models are not validated.
- Dynamic tree attention bias is rebuilt per verification step as a dense host
  buffer and copied once to the GPU; this is simple and correct but not the
  final optimized metadata path.

## Follow-Up Work

- Run a server-backed Rebench-shaped Qwen3 full-attention comparison for DFlash
  vs DDTree+DFlash across the AIPerf-style concurrency sweep.
- Keep stage timers and acceptance counters enabled for sampled runs while using
  no-debug runs for headline throughput.
- Add stochastic sampling and logprob support.
- Add MLA/fp8 KV tree verification and accepted-KV compaction for Kimi K2.5/K2.6.
- Add DSA/stateful-cache support before attempting GLM-5.1 DDTree.
