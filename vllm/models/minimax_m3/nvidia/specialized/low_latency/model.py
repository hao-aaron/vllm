# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-latency MiniMax-M3.1 language model: the general class plus a decode
executor built after load, run inside FULL decode graph captures."""

import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.nvidia.model import (
    MiniMaxM3Model,
    MiniMaxM3SparseForCausalLM,
)
from vllm.specialized import Model

from .config import MAX_REQUESTS, MAX_ROWS, MAX_SPECULATIVE_TOKENS, decode_config
from .peer_logits import peer_logits, peer_logits_unsupported
from .runtime.decode import SpecializedDecode
from .runtime.warmup import warm_up

logger = init_logger(__name__)


class _Backbone(MiniMaxM3Model):
    # Every executor step defers the MoE finalize, up to MAX_ROWS: the executor's
    # wide front GEMV reads the fused finalize's MXFP8 rows.
    fused_finalize_max_tokens = max(MiniMaxM3Model.fused_finalize_max_tokens, MAX_ROWS)


class LowLatencyMiniMaxM3(Model, MiniMaxM3SparseForCausalLM):
    backbone_cls = _Backbone

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._vllm_config = vllm_config
        self._decode_config = decode_config(vllm_config)
        self._decode: SpecializedDecode | None = None
        self._peer_logits = None

    def warmup(self) -> None:
        # Collective, on every rank in the same order: the executor's kernels, the
        # target's peer logits, then the DSpark drafter's.
        cfg = self._vllm_config
        self._decode = SpecializedDecode.for_target(
            self.model, self._decode_config, self.lm_head
        )
        warm_up(self._decode, cfg)
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
            self._peer_logits.warm_up()
        spec = cfg.speculative_config
        if spec is not None and spec.num_speculative_tokens:
            from .drafter import warm_up_drafters

            warm_up_drafters(int(spec.num_speculative_tokens))

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
        if step is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        self.count("low_latency_capture")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_input_ids(input_ids)
        # The executor runs the layers and the final norm (no pipeline parallelism).
        return decode.run(step, inputs_embeds, positions)

    def sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # The same on every TP rank: every rank samples every request.
        if self._peer_logits is None:
            return None
        return self._peer_logits(hidden_states)
