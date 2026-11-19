# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency GLM-5.3-Flash MTP draft model."""

from typing import Any

import torch

from vllm.models.glm5next.common.mtp import Glm5NextMTP, Glm5NextMultiTokenPredictor
from vllm.specialized import Model

from .runtime import DecodeConfig
from .runtime.draft import SpecializedDraft


class _Predictor(Glm5NextMultiTokenPredictor):
    _decode: SpecializedDraft | None = None
    prefill_rows: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if self._decode is not None:
            step = self._decode.plan(positions)
            if step is not None:
                layer = self._mtp_layers[spec_step_idx % self.num_mtp_layers]
                return self._decode.run(
                    layer,
                    step,
                    input_ids,
                    inputs_embeds,
                    previous_hidden_states,
                    self.prefill_rows,
                )
        return super().forward(
            input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_idx
        )


class LowLatencyGlm5NextMTP(Model, Glm5NextMTP):
    predictor_cls = _Predictor
    _kpool: Any = None

    def build_executor(self, config: DecodeConfig, kpool: Any) -> None:
        """From the target's `attach_draft`, once the target embedding and head
        are shared. Collective across TP ranks."""
        self._kpool = kpool
        self.model._decode = SpecializedDraft(self.model, config)

    def indexer_kpool_stage(self) -> Any:
        return self._kpool

    def set_prefill_rows(self, rows: torch.Tensor) -> None:
        """The speculator's [max_num_reqs] last-token rows of a draft prefill."""
        self.model.prefill_rows = rows
