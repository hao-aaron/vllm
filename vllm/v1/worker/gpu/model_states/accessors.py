# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What core reads from a model, through the `Model` accessors; plain modules
are read through their conventional attributes."""

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from vllm.specialized.model import Model


def get_language_model(model: nn.Module) -> nn.Module:
    fn = getattr(model, "get_language_model", None)
    return fn() if fn is not None else model


def get_lm_head(model: nn.Module) -> nn.Module | None:
    if isinstance(model, Model):
        return model.get_lm_head()
    return getattr(get_language_model(model), "lm_head", None) or getattr(
        model, "lm_head", None
    )


def get_embedding_layer(model: nn.Module) -> nn.Module | None:
    """The target's input embedding layer, for drafters that share it."""
    if isinstance(model, Model):
        return model.get_embedding_layer()
    lm = get_language_model(model)
    inner = getattr(lm, "model", lm)
    return getattr(inner, "embed_tokens", None) or getattr(inner, "embedding", None)


def get_topk_indices_buffer(model: nn.Module) -> Any:
    """The target's sparse-attention top-k buffer that MTP drafts share, if any."""
    fn = getattr(model, "get_topk_indices_buffer", None)
    if isinstance(model, Model) and fn is not None:
        return fn()
    inner = getattr(get_language_model(model), "model", None)
    return getattr(inner, "topk_indices_buffer", None)


def draft_hook(draft: nn.Module, name: str) -> Callable[..., Any] | None:
    """A draft hook (`vllm.specialized.model.DRAFT_HOOKS`) if the draft has it;
    plain draft models may still define it on their inner `.model`."""
    hook = getattr(draft, name, None)
    if hook is None and not isinstance(draft, Model):
        hook = getattr(getattr(draft, "model", None), name, None)
    return hook
