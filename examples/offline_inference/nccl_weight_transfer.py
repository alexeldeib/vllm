# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates NCCL weight transfer between vLLM instances.

This example shows how to start a receiver that loads model weights
from a running donor instance via NCCL P2P instead of from disk.

Prerequisites:
  - A donor vLLM instance running with the weight server enabled:
    VLLM_WEIGHT_SERVER_ENABLED=1 vllm serve <model> --tensor-parallel-size 8

Usage:
  python nccl_weight_transfer.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --load-format nccl \
      --model-loader-extra-config '{"nccl_addr": "10.0.0.1"}' \
      --tensor-parallel-size 8 \
      --prompt "Hello, my name is" \
      --max-tokens 50

If the donor is unreachable, the receiver falls back to disk loading
automatically.
"""

import dataclasses

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    parser.set_defaults(load_format="nccl")

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Prompt for inference validation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine_args = EngineArgs.from_cli_args(args)

    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate([args.prompt], sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated!r}")


if __name__ == "__main__":
    main()
