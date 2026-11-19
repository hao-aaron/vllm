# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency MiniMax-M3.1 DSpark draft model: fused Markov sampling, the
TP-split context projection and fused norms."""

from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkForCausalLM
from vllm.specialized import Model

from .drafter import FusedMarkovSampler, ShardedContextFc, build_drafter


class LowLatencyMiniMaxM3DSpark(Model, MiniMaxM3DSparkForCausalLM):
    _sampler: FusedMarkovSampler | None = None
    _context_fc: ShardedContextFc | None = None

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        if self._sampler is None:
            self._sampler = build_drafter(self, get_current_vllm_config())
            self._context_fc = ShardedContextFc(self, super().combine_hidden_states)

    def warm_up(self, num_speculative_tokens: int) -> None:
        """Collective; from the target's `warmup`."""
        if self._sampler is not None:
            self._sampler.warm_up(num_speculative_tokens)

    def sample_sequential(
        self, speculator: Any, num_reqs: int, head_hidden: torch.Tensor
    ) -> bool:
        """The Markov steps on fused kernels; False leaves a step to the stock loop."""
        return self._sampler is not None and self._sampler(
            speculator, num_reqs, head_hidden
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._context_fc is None:
            return super().combine_hidden_states(hidden_states)
        return self._context_fc(hidden_states)
