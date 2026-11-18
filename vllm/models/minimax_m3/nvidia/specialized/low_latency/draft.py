# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency MiniMax-M3.1 DSpark draft model."""

from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkForCausalLM
from vllm.specialized import Model

from .drafter import attach_drafter


class LowLatencyMiniMaxM3DSpark(Model, MiniMaxM3DSparkForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Fused Markov sampling, the TP-split context fc and fused norms.
        attach_drafter(self, vllm_config)
