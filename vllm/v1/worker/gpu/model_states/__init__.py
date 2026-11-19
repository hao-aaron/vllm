# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState

logger = init_logger(__name__)


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    """The `Model` interface the runner talks to: core's automatic wrapper for
    the model's kind, dispatching to a `vllm.specialized.Model`'s own hooks."""
    from vllm.specialized.model import Model

    target = model if isinstance(model, Model) else None
    if target is None:
        # A general wrapper forwards to a specialized language model.
        language_model = getattr(model, "get_language_model", lambda: None)()
        if isinstance(language_model, Model):
            target = language_model
    cls = resolve_model_state_cls(vllm_config, model)

    # Reject enable_prompt_embeds for states that would silently ignore it.
    if vllm_config.model_config.enable_prompt_embeds and not cls.supports_prompt_embeds:
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")

    if target is None:
        return cls(vllm_config, model, encoder_cache, device)
    from vllm.v1.worker.gpu.model_states.specialized import dispatching

    state = dispatching(cls, type(target))(vllm_config, model, encoder_cache, device)
    state.bind(target, cls)
    return state


def resolve_model_state_cls(
    vllm_config: VllmConfig, model: nn.Module
) -> type[ModelState]:
    # Deprecated: a model that brings its own ModelState.
    if hasattr(model, "get_model_state_cls"):
        logger.warning_once(
            "%s.get_model_state_cls is deprecated; move its hooks onto the model "
            "(vllm.specialized.Model).",
            type(model).__name__,
        )
        return model.get_model_state_cls()
    return _general_model_state_cls(vllm_config, model)


def _general_model_state_cls(
    vllm_config: VllmConfig, model: nn.Module
) -> type[ModelState]:
    # Cross-attention encoder-decoder models (Whisper, CohereASR, NemotronParse, ...)
    if any(isinstance(m, CrossAttention) for m in model.modules()):
        from vllm.v1.worker.gpu.model_states.encoder_decoder import (
            EncoderDecoderModelState,
        )

        return EncoderDecoderModelState

    # Encoder-only attention is non-causal and needs no KV cache.
    if any(
        layer.attn_type == AttentionType.ENCODER_ONLY
        for layer in get_layers_from_vllm_config(vllm_config, Attention).values()
    ):
        from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

        return EncoderOnlyModelState

    if vllm_config.model_config.is_hybrid or vllm_config.model_config.is_attention_free:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState

    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState
