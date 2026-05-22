# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import threading
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput

logger = init_logger(__name__)


def _k26_step_debug_enabled() -> bool:
    return os.getenv("VLLM_K26_TEP8_STEP_DEBUG", "0") == "1"


def _k26_async_output_event_log_interval_s() -> float:
    try:
        return float(os.getenv("VLLM_K26_ASYNC_OUTPUT_EVENT_LOG_INTERVAL_S", "15"))
    except ValueError:
        return 15.0


def _k26_async_output_event_timeout_s() -> float:
    try:
        return float(os.getenv("VLLM_K26_ASYNC_OUTPUT_EVENT_TIMEOUT_S", "0"))
    except ValueError:
        return 0.0


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = torch.cuda.Event()
        if _k26_step_debug_enabled():
            logger.warning(
                "K26 TEP8 step debug AsyncOutput init begin: "
                "reqs=%s sampled_shape=%s num_sampled_shape=%s",
                len(model_runner_output.req_ids),
                tuple(sampler_output.sampled_token_ids.shape),
                tuple(num_sampled_tokens.shape),
            )

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            self.copy_event.record(copy_stream)
        if _k26_step_debug_enabled():
            logger.warning("K26 TEP8 step debug AsyncOutput init complete")

    def get_output(self) -> ModelRunnerOutput:
        start_time = time.monotonic()
        if _k26_step_debug_enabled():
            logger.warning(
                "K26 TEP8 step debug AsyncOutput get_output synchronize begin: "
                "reqs=%s thread=%s current_device=%s",
                len(self.model_runner_output.req_ids),
                threading.current_thread().name,
                torch.cuda.current_device() if torch.cuda.is_available() else None,
            )
            log_interval_s = max(_k26_async_output_event_log_interval_s(), 1.0)
            timeout_s = _k26_async_output_event_timeout_s()
            next_log_s = log_interval_s
            while not self.copy_event.query():
                elapsed_s = time.monotonic() - start_time
                if elapsed_s >= next_log_s:
                    logger.warning(
                        "K26 TEP8 step debug AsyncOutput copy event waiting: "
                        "elapsed_s=%.3f reqs=%s thread=%s current_device=%s",
                        elapsed_s,
                        len(self.model_runner_output.req_ids),
                        threading.current_thread().name,
                        torch.cuda.current_device() if torch.cuda.is_available() else None,
                    )
                    next_log_s += log_interval_s
                if timeout_s > 0 and elapsed_s >= timeout_s:
                    raise TimeoutError(
                        "K26 TEP8 AsyncOutput copy event did not complete within "
                        f"{timeout_s:.3f}s for {len(self.model_runner_output.req_ids)} "
                        "request(s)"
                    )
                time.sleep(0.1)
        else:
            self.copy_event.synchronize()
        if _k26_step_debug_enabled():
            logger.warning(
                "K26 TEP8 step debug AsyncOutput get_output synchronize complete: "
                "elapsed_s=%.6f",
                time.monotonic() - start_time,
            )

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        if _k26_step_debug_enabled():
            logger.warning("K26 TEP8 step debug AsyncOutput get_output complete")
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        self.copy_event = torch.cuda.Event()

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        self.copy_event.synchronize()
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
