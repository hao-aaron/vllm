# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    cls = resolve_model_state_cls(vllm_config, model)

    # Reject enable_prompt_embeds for states that would silently ignore it.
    if vllm_config.model_config.enable_prompt_embeds and not cls.supports_prompt_embeds:
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")

    return cls(vllm_config, model, encoder_cache, device)


def resolve_model_state_cls(
    vllm_config: VllmConfig, model: nn.Module
) -> type[ModelState]:
    from vllm.specialized.model import Model

    language_model = getattr(model, "get_language_model", lambda: None)()
    if isinstance(model, Model) or isinstance(language_model, Model):
        from vllm.v1.worker.gpu.model_states.specialized import adapter_for

        return adapter_for(_general_model_state_cls(vllm_config, model))

    return _general_model_state_cls(vllm_config, model)


def _general_model_state_cls(
    vllm_config: VllmConfig, model: nn.Module
) -> type[ModelState]:
    # Let the model provide its own ModelState if it defines one.
    if hasattr(model, "get_model_state_cls"):
        return model.get_model_state_cls()

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
