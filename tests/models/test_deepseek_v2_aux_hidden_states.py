# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

import vllm.model_executor.models.deepseek_v2 as deepseek_v2


class _IncrementLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions, residual, llama_4_scaling
        return hidden_states + 1, torch.zeros_like(hidden_states)


class _IdentityNorm(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return hidden_states + residual, None


def test_deepseek_v2_can_extract_final_hidden_state(monkeypatch) -> None:
    monkeypatch.setattr(
        deepseek_v2,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    model = deepseek_v2.DeepseekV2Model.__new__(deepseek_v2.DeepseekV2Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([_IncrementLayer(), _IncrementLayer()])
    model.norm = _IdentityNorm()
    model.aux_hidden_state_layers = (2,)

    final_hidden_states, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.arange(3),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(3, 4),
    )

    torch.testing.assert_close(final_hidden_states, torch.full((3, 4), 2.0))
    assert len(aux_hidden_states) == 1
    torch.testing.assert_close(aux_hidden_states[0], final_hidden_states)
