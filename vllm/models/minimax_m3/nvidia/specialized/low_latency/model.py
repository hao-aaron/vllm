# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency MiniMax-M3.1 language model: the general class plus a decode
executor built from the processed weights, run inside FULL decode graphs."""

import weakref

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseForCausalLM
from vllm.specialized import Model

from .config import MAX_REQUESTS, MAX_SPECULATIVE_TOKENS, decode_config
from .peer_logits import peer_logits, peer_logits_unsupported
from .runtime.decode import SpecializedDecode
from .runtime.warmup import warm_up

logger = init_logger(__name__)


class LowLatencyMiniMaxM3(Model, MiniMaxM3SparseForCausalLM):
    _decode: SpecializedDecode | None = None
    _peer_logits = None
    _draft: weakref.ref | None = None

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        # Built once: no deployment with weight reload routes here. Collective, on
        # every rank in the same order: the executor, then the peer logits.
        if self._decode is not None:
            return
        cfg = decode_config(get_current_vllm_config())
        self._decode = SpecializedDecode.for_target(self.model, cfg, self.lm_head)
        reason = peer_logits_unsupported(self.lm_head, self.logits_processor)
        if reason:
            logger.info("Target logits stay gathered: %s.", reason)
        else:
            self._peer_logits = peer_logits(
                self.lm_head,
                self.logits_processor,
                MAX_REQUESTS * (1 + MAX_SPECULATIVE_TOKENS),
                "target_logits",
            )

    def attach_draft(self, draft: nn.Module) -> None:
        self._draft = weakref.ref(draft)

    def warmup(self) -> None:
        # Collective: the executor's kernels, the peer logits, then the drafter's.
        vllm_config = get_current_vllm_config()
        if self._decode is not None:
            warm_up(self._decode, vllm_config)
        if self._peer_logits is not None:
            self._peer_logits.warm_up()
        draft = None if self._draft is None else self._draft()
        spec = vllm_config.speculative_config
        if draft is not None and spec is not None and spec.num_speculative_tokens:
            draft.warm_up(int(spec.num_speculative_tokens))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        decode = self._decode
        step = None if decode is None else decode.plan(positions, intermediate_tensors)
        if decode is None or step is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_input_ids(input_ids)
        # The executor runs the layers and the final norm (no pipeline parallelism).
        return decode.run(step, inputs_embeds, positions)

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The same on every TP rank: every rank samples every request.
        if self._peer_logits is None:
            return super().compute_sample_logits(hidden_states)
        return self._peer_logits(hidden_states)
