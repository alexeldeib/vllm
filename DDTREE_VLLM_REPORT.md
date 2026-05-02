# DDTree vLLM Integration Report

## Summary

This branch adds a correctness-first DDTree speculative decoding path on top of
vLLM's existing DFlash support. DDTree uses one DFlash parallel draft pass to
build a prefix-closed draft tree, verifies the dynamic tree in one target-model
forward pass with `TREE_ATTN`, and greedily walks the verified tree to emit the
accepted target path plus the first fallback token.

The implementation was monkey-patched into the existing
`docker.cloudsmith.io/coreweave/infr/vllm:v2.10.0` image and validated on a GB200
pod in `cw4637-dev-us-e-01a`, namespace `ace-inference`.

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
  - dynamic tree attention bias construction,
  - greedy DDTree verifier walk,
  - `DDTreeProposer`, implemented as a DFlash proposer that converts draft
    logits into tree nodes.
- Extended scheduler and request state to carry method-specific speculative
  metadata from drafter output to verifier scheduling.
- Extended `TREE_ATTN` metadata to accept a dynamic tree attention bias for the
  current verification step.
- Updated the GPU model runner to:
  - require target `attention_backend="TREE_ATTN"` for DDTree,
  - keep the DFlash drafter on `FLASH_ATTN`,
  - schedule DDTree scratch slots separately from model positions,
  - compute logits for root plus every tree node,
  - sample with the DDTree greedy walk,
  - disable async scheduling for DDTree.
- Added focused unit coverage in `tests/v1/spec_decode/test_ddtree.py`.

## Correctness Results

Environment:

- Cluster: `cw4637-dev-us-e-01a`
- Namespace: `ace-inference`
- Test pod: `ddtree-vllm-dev`
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
| Pod import/unit smoke | DDTree builder, depth ordering, parent links, attention bias passed |
| 48-token target/TREE vs DFlash vs DDTree | Exact token-id match |
| 48-token DDTree with no explicit `ddtree_tree_budget` | Exact token-id match with target/TREE |
| 128-token DFlash vs DDTree, budget 16 | Exact token-id match |
| 128-token DFlash vs DDTree, budget 32 | Exact token-id match |
| 128-token DFlash vs DDTree, budget 64 | Exact token-id match |

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

These numbers are smoke-test measurements from a single GB200 pod with
`enforce_eager=True`, `max_num_seqs=1`, and one prompt. They are useful for
relative integration validation, not as final serving benchmarks.

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
- One active request per DDTree verification step. The runner explicitly rejects
  larger active batches for now.
- Target model must use `attention_backend="TREE_ATTN"`.
- Async scheduling is disabled for DDTree.
- The correctness-first path verifies into scratch future slots and then lets
  the next engine step recompute accepted tokens canonically. This avoids unsafe
  KV-cache compaction but leaves performance on the table.
- M-RoPE/XD-RoPE and multimodal targets are not validated.
- Hybrid linear-attention/GDN target models are not validated.
- Dynamic tree attention bias is rebuilt per verification step on GPU; this is
  simple and correct but not the final optimized metadata path.

## Follow-Up Work

- Add KV-cache compaction or accepted-path slot remapping so DDTree can retain
  verified target KVs instead of performing a canonical catch-up step.
- Extend DDTree metadata handling to batched active requests.
- Add stochastic sampling and logprob support.
- Add CI e2e coverage once the container image branch is built.
- Benchmark under serving-like settings without eager mode and with realistic
  prompt/output distributions.
