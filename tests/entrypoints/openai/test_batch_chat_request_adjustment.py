# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser


class _FakeRender:
    use_harmony = False
    chat_template = None
    chat_template_content_format = "auto"
    default_chat_template_kwargs = None
    tool_parser = None
    reasoning_parser = None
    trust_request_chat_template = False

    def validate_chat_template(self, **kwargs):
        return None

    async def preprocess_chat(self, request, messages, **kwargs):
        request.skip_special_tokens = False
        return list(messages), [{"prompt_token_ids": [len(messages)]}]


async def _check_model(request):
    return None


def test_batch_render_reuses_requests_adjusted_by_preprocess():
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
    )
    single_requests = [
        request.to_chat_completion_request(messages) for messages in request.messages
    ]
    assert all(req.skip_special_tokens for req in single_requests)

    service = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    service._check_model = _check_model
    service.engine_client = SimpleNamespace(
        errored=False,
        dead_error=RuntimeError("engine errored"),
    )
    service.openai_serving_render = _FakeRender()

    result = asyncio.run(service.render_batch_chat_request(request, single_requests))

    conversations, engine_prompts = result
    assert [req.skip_special_tokens for req in single_requests] == [False, False]
    assert len(conversations) == 2
    assert engine_prompts == [
        {"prompt_token_ids": [1]},
        {"prompt_token_ids": [1]},
    ]


class _StatefulBatchReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.is_end_calls = 0
        self.extract_calls = 0

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        self.is_end_calls += 1
        return self.is_end_calls == 1

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning(self, model_output, request):
        self.extract_calls += 1
        return f"extract-{self.extract_calls}", model_output

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        return None


async def _single_output(request_id: str, prompt_token_ids: list[int]):
    yield RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="answer",
                token_ids=[1],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
    )


def test_batch_chat_completion_uses_isolated_reasoning_parsers():
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
    )
    captured_reasoning_ended: list[bool | None] = []
    prompts = [
        {"prompt_token_ids": [11]},
        {"prompt_token_ids": [22]},
    ]

    async def render_batch_chat_request(request, single_requests):
        return request.messages, prompts

    def generate(engine_prompt, sampling_params, request_id, **kwargs):
        captured_reasoning_ended.append(kwargs["reasoning_ended"])
        return _single_output(request_id, engine_prompt["prompt_token_ids"])

    service = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    service.renderer = SimpleNamespace(tokenizer=object())
    service.reasoning_parser_cls = _StatefulBatchReasoningParser
    service.render_batch_chat_request = render_batch_chat_request
    service._effective_chat_template_kwargs = lambda request: {}
    service._base_request_id = lambda raw_request, request_id: "batch"
    service._maybe_get_adapters = lambda request, supports_default_mm_loras: None
    service.models = SimpleNamespace(model_name=lambda lora_request: "test-model")
    service._get_data_parallel_rank = lambda raw_request: None
    service.model_config = SimpleNamespace(max_model_len=64)
    service.default_sampling_params = {}
    service.override_max_tokens = None
    service._extract_prompt_len = lambda prompt: len(prompt["prompt_token_ids"])
    service._extract_prompt_components = lambda prompt: SimpleNamespace(
        token_ids=prompt["prompt_token_ids"]
    )
    service._log_inputs = lambda *args, **kwargs: None
    service.engine_client = SimpleNamespace(generate=generate)
    service.get_chat_request_role = lambda request: "assistant"
    service._raise_if_error = lambda finish_reason, request_id: None
    service.system_fingerprint = None

    response = asyncio.run(service.create_batch_chat_completion(request))

    assert captured_reasoning_ended == [True, True]
    assert [choice.message.reasoning for choice in response.choices] == [
        "extract-1",
        "extract-1",
    ]
