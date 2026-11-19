# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency GLM-5.3-Flash language model: the general class plus a decode
executor built from the processed weights, run inside FULL decode graphs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.models.glm5next.common.model import Glm5NextForCausalLM
from vllm.specialized import Model

from .runtime import decode_config
from .runtime.decode import SpecializedDecode


class LowLatencyGlm5Next(Model, Glm5NextForCausalLM):
    _decode: SpecializedDecode | None = None

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        # Built once: no deployment with weight reload routes here.
        if self._decode is None:
            # Collective: symmetric-memory rendezvous across the TP ranks.
            self._decode = SpecializedDecode(
                self.model, decode_config(self.vllm_config)
            )
            self._decode.build_logits(self.logits_processor, self.lm_head)

    def attach_draft(self, draft: nn.Module) -> None:
        from .draft import LowLatencyGlm5NextMTP

        if isinstance(draft, LowLatencyGlm5NextMTP) and self._decode is not None:
            draft.build_executor(decode_config(self.vllm_config), self._decode.kpool)

    def indexer_kpool_stage(self) -> Any:
        return None if self._decode is None else self._decode.kpool

    def initialize_kv_cache(self, kv_cache_config, block_tables) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
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
                return decode.run(step, input_ids, inputs_embeds)
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        decode = self._decode
        if decode is not None:
            logits = decode.compute_logits(
                self.logits_processor, self.lm_head, hidden_states
            )
            if logits is not None:
                return logits
        return super().compute_logits(hidden_states)
