# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency GLM-5.3-Flash MTP draft model."""

import torch

from vllm.models.glm5next.common.mtp import Glm5NextMTP, Glm5NextMultiTokenPredictor
from vllm.specialized import Model

from .runtime import decode_config
from .runtime.draft import SpecializedDraft


class _Predictor(Glm5NextMultiTokenPredictor):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._decode_config = decode_config(vllm_config)
        self._decode: SpecializedDraft | None = None
        self.prefill_rows: torch.Tensor | None = None

    def set_prefill_rows(self, rows: torch.Tensor) -> None:
        """The speculator's [max_num_reqs] last-token rows of a draft prefill."""
        self.prefill_rows = rows

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Built on the first eager forward, after the weights are processed;
        # collective across TP ranks.
        if self._decode is None and not torch.cuda.is_current_stream_capturing():
            self._decode = SpecializedDraft(self, self._decode_config)
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
