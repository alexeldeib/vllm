# DDTree Kimi K2.5 and GLM 5.1 Plan

## Branch Stack

1. `alex/ddtree-vllm-integration`
   - Initial correctness branch.
   - Adds batched DDTree on top of DFlash for non-MLA targets using
     `TREE_ATTN`.
   - Includes the DFlash vs DDTree+DFlash validation reports.

2. `alex/ddtree-kimi-k25`
   - Kimi K2.5 readiness branch.
   - Keeps DDTree guarded for MLA targets until target verification is exact.
   - Fixes DFlash/DDTree aux-hidden-layer config handling for public Kimi
     DFlash heads.

3. `alex/ddtree-glm51-scaffold`
   - GLM 5.1 scaffolding branch.
   - Intended to add aux-hidden-state support for GLM MoE targets and keep it
     behind the same verifier capability checks until GLM can be tested.

## Current K2.5 Cluster Baseline

Observed running pods are in context `cw4637-dev-us-e-01a`, namespace
`ace-inference`. I only inspected them and did not modify them.

The preferred K2.5 baseline is NVFP4:

- Pod: `c2-k25-v0191-bench-trtllm-79d97865c4-kghc8`
- Image: `docker.cloudsmith.io/coreweave/infr-dev/vllm:25d923d8fad8cd13885711a98cfc85979f1a14ba`
- Model: `s3://infr/raw/nvidia/Kimi-K2.5-NVFP4/e4e908c073784de20ad3af0be653421f1088922d`
- Target args:
  - `--attention-config={"use_trtllm_ragged_deepseek_prefill": true}`
  - `--block-size=32`
  - `--compilation-config={"mode": 3}`
  - `--gpu-memory-utilization=0.95`
  - `--max-model-len=262144`
  - `--max-num-batched-tokens=8192`
  - `--max-num-seqs=50`
  - `--tensor-parallel-size=4`
  - `--reasoning-parser=kimi_k2`
  - `--tool-call-parser=kimi_k2`
  - `--trust-remote-code`

The in-cluster speculative K2.5 reference uses Eagle3:

- Pod: `c2-k25-v0191-bench-spec-trtllm-6b79f9468-6vf4f`
- Target model: same NVFP4 K2.5 path as the baseline.
- Draft model: `s3://infr/raw/nvidia/Kimi-K2.5-Thinking-Eagle3/0b0c6ac039089ad2c2418c91c039553381a302d9`
- Spec args:
  - `--speculative-config={"model": "/tmp/nvidia/Kimi-K2.5-Thinking-Eagle3", "method": "eagle3", "num_speculative_tokens": 4, "draft_tensor_parallel_size": 4}`

The public DFlash head to target next is
`https://huggingface.co/z-lab/Kimi-K2.5-DFlash`. Its config declares a
`DFlashDraftModel` head with Kimi K2.5 target layer ids in `dflash_config`.

## What Is Ready

The Kimi branch now handles the public DFlash head's aux-layer config in the
same way at graph-hash time and target hidden-state setup time:

- `eagle_aux_hidden_state_layer_ids`
- `dflash_config.target_layer_ids`
- `dflash_config.layer_ids` compatibility alias

This matters because changing the target hidden layers changes the target
forward graph shape: vLLM must rebuild the graph rather than reusing a graph
captured for a different DFlash head layout.

Kimi K2.5 model-side aux hidden states are already available through
`KimiK25ForConditionalGeneration`, which delegates to the DeepSeek-V2 language
model. The target can expose the hidden states needed by DFlash/DDTree.

## Why K2.5 DDTree Is Still Guarded

The serving K2.5 pods use MLA attention and FP8 KV cache with TRT-LLM ragged
DeepSeek prefill. The current DDTree verifier is implemented for standard
`Attention` layers through `TREE_ATTN`, not for `MLAAttention`.

DDTree verification is not equivalent to normal linear speculative decoding.
For each target query, the verifier must attend to:

- the prefix context,
- the DDTree root,
- that node's ancestors,
- the node itself,
- no siblings or non-ancestor tree nodes.

Current MLA decode kernels support normal single-token decode and linear
multi-token speculative decode, where token `j` attends to the prefix plus the
previous linear draft tokens. They do not accept DDTree's arbitrary per-request
root/ancestor/self mask. Flattening DDTree nodes into a linear sequence would
allow sibling attention and would be incorrect.

The guard in `GPUModelRunner` is therefore intentional: DDTree remains enabled
only when the target backend is `TREE_ATTN`. Kimi/DeepSeek/GLM MLA targets now
fail with an explicit message instead of silently running an invalid verifier.

## Production Work Remaining for K2.5 Gains

To get paper-like DDTree gains on Kimi K2.5, the missing production work is:

1. Native MLA tree verification.
   - Add a DDTree-capable MLA target attention path for the GB200 production
     backend mix: FlashInfer MLA decode, FlashMLA where selected, and TRT-LLM
     ragged prefill interaction.
   - The path must support a per-request tree mask or equivalent metadata that
     hides siblings/non-ancestors while preserving prefix attention.
   - It must support `kv-cache-dtype=fp8`, TP4/TP8, and Kimi's MLA dimensions.

2. MLA accepted-KV compaction.
   - Current DDTree accepted-KV compaction only handles standard
     `FullAttentionSpec` KV caches shaped like separate K/V tensors.
   - Kimi uses `MLAAttentionSpec`, where each token stores compressed MLA KV
     state. Accepted DDTree nodes need to be copied from scratch slots into the
     committed path slots for every MLA layer.

3. CUDA graph and batching validation.
   - Batched DDTree exists for `TREE_ATTN`.
   - The MLA path must preserve the production K2.5 batch shape:
     `max-num-seqs=50`, `max-num-batched-tokens=8192`, `block-size=32`, TP4,
     and no-delay multi-client AIPerf style traffic.

4. K2.5 benchmark sweep.
   - Start with a copied single-node pod from the observed NVFP4 baseline.
   - Validate DFlash only.
   - Enable DDTree once MLA tree verification is exact.
   - Compare DFlash vs DDTree+DFlash on the same prompts, request concurrency,
     output lengths, and sampling settings.

## GLM 5.1 Stretch Goal

The observed GLM 5.1 deployment is scaled to zero in namespace `inference-vh`,
so it cannot be validated yet. Its config uses:

- Model: `s3://infr/raw/zai-org/GLM-5.1-FP8/baa97427535b5456e7e6a32a58045512698e2357`
- Image: `docker.cloudsmith.io/coreweave/infr/vllm:v2.7.0`
- `--tensor-parallel-size=8`
- `--block-size=64`
- `--kv-cache-dtype=fp8`
- `--max-model-len=131072`
- `--max-num-batched-tokens=8192`
- `--max-num-seqs=10`

GLM 5.1 should be added after Kimi by first enabling aux hidden-state capture
for the GLM MoE target class and keeping DDTree behind the same verifier guard.
If GLM 5.1 remains standard attention, `TREE_ATTN` may be enough for DDTree. If
the selected GLM target path is MLA or another specialized attention path, it
needs the same tree-verification support as Kimi before DDTree can be enabled.
