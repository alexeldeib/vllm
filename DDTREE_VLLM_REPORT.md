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
compaction fixes.

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
| Current target-only greedy vs DDTree+DFlash greedy | Exact token-id match |
| Pod import/unit smoke | DDTree builder, depth ordering, parent links, attention bias passed |
| 48-token target/TREE vs DFlash vs DDTree | Exact token-id match |
| 48-token DDTree with no explicit `ddtree_tree_budget` | Exact token-id match with target/TREE |
| 128-token DFlash vs DDTree, budget 16 | Exact token-id match |
| 128-token DFlash vs DDTree, budget 32 | Exact token-id match |
| 128-token DFlash vs DDTree, budget 64 | Exact token-id match |

Clean CI image validation:

- `ml-containers` branch: `alex-ddtree-vllm-image`
- `ml-containers` commit: `fdc3617b2595279b38e4f31818ab16c7e42db55a`
- vLLM commit: `8f0c77a7c20db5843a2fadddf47a417b8777ee3c`
- CI run: `https://github.com/coreweave/ml-containers/actions/runs/25262221047`
- Image:
  `ghcr.io/coreweave/ml-containers/vllm-tensorizer:alex-ddtree-vllm-image-fdc3617-8f0c77a7c20db5843a2fadddf47a417b8777ee3c`
- Manifest index digest:
  `sha256:69c787bc1324333f2f50ddd3cf89eb2c33369c00808e0db463c5ffce200bd551`
- Platforms: `linux/amd64`, `linux/arm64`
- Clean-image test pod: `ddtree-clean-vllm-test`
- Runtime architecture: `aarch64`
- vLLM package version:
  `0.1.dev16274+g8f0c77a7c.d20260502`

| Clean image check | Result |
| --- | --- |
| Installed source syntax | `py_compile` passed for modified installed files |
| Direct batched DDTree unit smoke | Passed |
| 3D qq-bias Triton kernel check | `max_diff 0.0`, passed |
| Target-only greedy vs DDTree+DFlash greedy | Exact token-id match |
| 4-request batched DDTree generate with accepted-KV compaction and immediate re-proposal | Passed |
| Same-shape warmed DFlash vs DDTree+DFlash timing smoke | Passed |

Earlier clean-image checks, from the pre-compaction CI image, also passed
48-token target/TREE vs DFlash vs DDTree and 128-token DFlash vs DDTree
correctness. The current image supersedes those results.

Observed baseline caveat:

- In the 128-token smoke, both DFlash and DDTree diverged from the non-spec
  target/TREE baseline at token index 112, with the same token selected by
  DFlash and DDTree. This appears to be an existing DFlash/speculative baseline
  behavior under the eager single-request harness, not a DDTree-specific
  divergence.

Unsupported or not yet validated:

- `Qwen/Qwen3.5-4B` hybrid GDN/linear-attention target path failed during
  multi-node DDTree verification and is not included in the supported envelope.
- Budget 8 was not accepted into the validated envelope; on the 128-token smoke
  it diverged from the DFlash baseline at token index 71. The branch therefore
  uses budget 32 by default when `num_speculative_tokens=8`.

## Performance Results

Current batched accepted-KV-compaction smoke on the GB200 clean CI image,
Qwen3-8B target, `z-lab/Qwen3-8B-DFlash-b16` drafter, `max_num_seqs=4`, four
prompts per batch, `max_tokens=64`, and target verifier forced through
`TREE_ATTN` for both modes. The script warms each mode with the same batch shape
before measuring three distinct prompt groups.

| Mode | Budget | Median output tokens | Median elapsed generation time | Median output tokens/s | Delta vs DFlash |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFlash, `TREE_ATTN` target | n/a | 256 | 1.0605s | 241.40 | n/a |
| DDTree+DFlash, `TREE_ATTN` target | 32 | 256 | 1.0022s | 255.44 | +5.8% |

This is a smoke result, not a serving benchmark, but it confirms that batched
verification plus accepted-KV compaction works end-to-end from the built image
and is modestly faster than DFlash on this warmed four-request profile. DFlash
still has async scheduling enabled while DDTree disables it, so this is not
stacking every possible DDTree optimization.

Measured samples:

| Mode | Samples |
| --- | --- |
| DFlash, `TREE_ATTN` target | `1.3977s/256tok/183.16tps`, `1.0562s/256tok/242.39tps`, `1.0605s/256tok/241.40tps` |
| DDTree+DFlash, `TREE_ATTN` target | `1.0022s/256tok/255.44tps`, `0.9659s/256tok/265.05tps`, `1.0049s/256tok/254.76tps` |

An earlier one-pass monkeypatch smoke reported `155.83` output tokens/s for
DFlash and `236.50` output tokens/s for DDTree+DFlash (+51.8%). That run used
only a tiny warmup prompt and is too warmup-sensitive to use as the headline
comparison.

The following historical numbers are single-request smoke measurements from a
single GB200 pod with `enforce_eager=True`, `max_num_seqs=1`, and one prompt.
They are useful for relative integration validation, not as final serving
benchmarks.

Prompt: `List three properties of a binary tree in one sentence.`

48 generated tokens:

| Mode | Tokens/s | Notes |
| --- | ---: | --- |
| Target only, default backend | 25.15 | Non-spec baseline |
| Target only, `TREE_ATTN` | 24.18 | Same token IDs as default target |
| DFlash, `num_speculative_tokens=8` | 7.30 | Known-good DFlash setup |
| DDTree, budget 32 | 8.93 | Exact token-id match with target/TREE and DFlash |
| DDTree, default budget | 9.04 | No explicit `ddtree_tree_budget`; defaulted to 32 |

DDTree budget 32 improved the 48-token smoke by about 22% over DFlash. The
no-explicit-budget run improved by about 24% over DFlash.

128 generated tokens:

| Mode | Budget | Tokens/s | Delta vs DFlash | Correctness note |
| --- | ---: | ---: | ---: | --- |
| Target only, `TREE_ATTN` | n/a | 34.11 | n/a | Baseline |
| DFlash | n/a | 18.71 | n/a | Diverged from target at index 112 |
| DDTree | 16 | 20.08 | +7.3% | Exact match with DFlash |
| DDTree | 32 | 20.20 | +8.0% | Exact match with DFlash |
| DDTree | 64 | 20.84 | +11.4% | Exact match with DFlash |

Earlier clean CI image results from
`ghcr.io/coreweave/ml-containers/vllm-tensorizer:alex-ddtree-vllm-image-be7262e-74f6f52790887fb0729dde13928a12cb3d0a2dad`
on `ddtree-vllm-ci`:

| Mode | Budget | Tokens | Tokens/s | Delta vs DFlash | Correctness note |
| --- | ---: | ---: | ---: | ---: | --- |
| Target only, `TREE_ATTN` | n/a | 48 | 13.64 | n/a | Baseline |
| DFlash | n/a | 48 | 6.04 | n/a | Exact match with target/TREE |
| DDTree | 32 | 48 | 8.59 | +42.3% | Exact match with target/TREE and DFlash |
| DFlash | n/a | 128 | 17.08 | n/a | Baseline for longer speculative smoke |
| DDTree | 64 | 128 | 20.90 | +22.4% | Exact match with DFlash |

The public DDTree results at https://liranringel.github.io/ddtree/ report much
larger expected end-to-end speedups in optimized benchmarking, with DDTree
typically improving over DFlash by increasing accepted-prefix lengths. The site
shows examples such as HumanEval on Qwen3-30B-MoE improving from 6.09x with
DFlash to 8.22x with DDTree relative to autoregressive decoding, and multiple
Qwen3 benchmark/model combinations in the roughly 3x to 8x range relative to
autoregressive decoding.

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
- Dynamic tree attention bias is rebuilt per verification step on GPU; this is
  simple and correct but not the final optimized metadata path.

## Follow-Up Work

- Run a Rebench-shaped Qwen3 full-attention comparison for DFlash vs
  DDTree+DFlash at C=1 and then a small concurrency sweep.
- Add stage timers and acceptance counters for draft forward, tree build,
  verifier forward, sampler walk, KV compaction, and next-draft prep.
- Add stochastic sampling and logprob support.
- Add MLA/fp8 KV tree verification and accepted-KV compaction for Kimi K2.5/K2.6.
- Add DSA/stateful-cache support before attempting GLM-5.1 DDTree.
