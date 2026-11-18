# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency GLM-5.3-Flash models: the general classes plus a decode
executor built after load, run inside FULL decode graph captures."""

import torch

from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)
from vllm.specialized import Model

from .runtime import decode_config
from .runtime.decode import SpecializedDecode


class _LowLatency(Model):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._decode_config = decode_config(vllm_config)
        self._decode: SpecializedDecode | None = None

    def _language_model(self) -> Glm5NextForCausalLM:
        if isinstance(self, Glm5NextForConditionalGeneration):
            return self.language_model
        return self  # type: ignore[return-value]

    def warmup(self) -> None:
        # Collective: symmetric-memory rendezvous across the TP ranks.
        lm = self._language_model()
        self._decode = SpecializedDecode(lm.model, self._decode_config)
        self._decode.build_logits(lm.logits_processor, lm.lm_head)

    def init_kv_cache(self, kv_cache_config, block_tables) -> None:
        if self._decode is not None:
            self._decode.prepare_kv()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        decode = self._decode
        if decode is not None and intermediate_tensors is None:
            step = decode.plan(positions, intermediate_tensors)
            if step is not None:
                self.count("low_latency_capture")
                return decode.run(step, input_ids, inputs_embeds)
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        decode = self._decode
        if decode is not None:
            lm = self._language_model()
            logits = decode.compute_logits(
                lm.logits_processor, lm.lm_head, hidden_states
            )
            if logits is not None:
                return logits
        return super().compute_logits(hidden_states)


class LowLatencyGlm5Next(_LowLatency, Glm5NextForCausalLM):
    pass


class LowLatencyGlm5NextVL(_LowLatency, Glm5NextForConditionalGeneration):
    pass
