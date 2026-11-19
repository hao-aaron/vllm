# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency GLM-5.3-Flash language model: the general class plus a decode
executor built from the processed weights, run inside FULL decode graphs."""

import weakref
from typing import Any

import torch
import torch.nn as nn

from vllm.models.glm5next.common.model import Glm5NextForCausalLM
from vllm.specialized import Model

from .runtime import decode_config
from .runtime.decode import SpecializedDecode


class LowLatencyGlm5Next(Model, Glm5NextForCausalLM):
    _decode: SpecializedDecode | None = None
    _draft: weakref.ref | None = None  # not a submodule

    def attach_draft(self, draft: nn.Module) -> None:
        from .draft import LowLatencyGlm5NextMTP

        if isinstance(draft, LowLatencyGlm5NextMTP):
            self._draft = weakref.ref(draft)

    def warmup(self) -> None:
        # Built here, so the directory can't list weight_reload. Collective, in
        # order: the decode executor, then the MTP draft's, which shares its kpool.
        if self._decode is not None:
            return
        cfg = decode_config(self.vllm_config)
        self._decode = SpecializedDecode(self.model, cfg)
        self._decode.build_logits(self.logits_processor, self.lm_head)
        draft = None if self._draft is None else self._draft()
        if draft is not None:
            draft.build_executor(cfg, self._decode.kpool)

    def fast_path_sizes(self) -> set[int]:
        return set() if self._decode is None else set(self._decode.fast_sizes)

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
