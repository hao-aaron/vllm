# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`Model`: the one model interface core talks to."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Per-step and lifecycle hooks, with `ModelState`'s names and signatures. Each
# default runs core's automatic wrapper for the model's kind (default, hybrid,
# encoder-decoder, encoder-only); an override replaces it and may call super().
HOOKS = (
    "initialize_kv_cache",
    "add_request",
    "remove_request",
    "apply_staged_writes",
    "capture_inner_cudagraphs",
    "get_additional_cg_support",
    "preprocess_state",
    "postprocess_state",
    "prepare_inputs_embeds",
    "dummy_inputs_embeds",
    "prepare_inputs",
    "prepare_dummy_inputs",
    "prepare_attn",
    "custom_sampler",
    "compute_sample_logits",
    "dummy_sampler_run",
)
# Called once by core: attach_draft then warmup before memory profiling (the
# draft's own warmup runs after the target's); fast_path_sizes after capture.
LIFECYCLE = ("attach_draft", "warmup", "fast_path_sizes")
# get_topk_indices_buffer is optional: core falls back to the backbone attribute.
ACCESSORS = (
    "get_embedding_layer",
    "get_lm_head",
    "get_language_model",
    "get_topk_indices_buffer",
)
# Methods core probes on draft models; a draft class defines the ones it has.
DRAFT_HOOKS = (
    "set_prefill_rows",
    "sample_sequential",
    "sample_markov_chain",
    "combine_hidden_states",
)
# What every model already implements and a specialized subclass may override.
MODULE_METHODS = (
    "forward",
    "compute_logits",
    "load_weights",
    "process_weights_after_loading",
)


@dataclass
class StepResult:
    """A fused step's outcome; see `Model.fused_step`."""

    sampled: torch.Tensor  # [num_reqs, k + 1] int32, left-aligned
    num_sampled: torch.Tensor  # [num_reqs] int32
    next_drafts: torch.Tensor | None = None  # [num_reqs, k] int32


class Model(nn.Module):
    """An `nn.Module` that owns its run-time state and hooks.

    A specialized model mixes this into its family's general class. Core wraps
    plain modules into the same interface automatically.
    """

    def _core_default(self, name: str, *args: Any, **kwargs: Any) -> Any:
        core = self.__dict__.get("_core")
        if core is None:
            raise RuntimeError(f"{type(self).__name__}.{name} before core bound it")
        state, base = core
        return getattr(base, name)(state, *args, **kwargs)

    def process_weights_after_loading(self) -> None:
        """Derived weights, after quantization processing, on load and on every
        reload. An override calls super() first."""
        fn = getattr(super(), "process_weights_after_loading", None)
        if fn is not None:
            fn()

    def warmup(self) -> None:
        """JIT compile and autotune, once, before memory profiling; not rerun on
        reload. Derived weights belong in `process_weights_after_loading()`."""

    def attach_draft(self, draft: nn.Module) -> None:
        """Called on the target once it and its `DRAFT_MODEL` are both loaded."""

    def fast_path_sizes(self) -> set[int]:
        """The captured graph sizes that take this model's fast path."""
        return set()

    def fused_step(self, batch: Any, sampling: Any) -> StepResult | None:
        return None

    # Accessors core reads instead of attributes. Defaults read the general
    # class's own accessor when it has one, else the conventional attributes.
    def get_language_model(self) -> nn.Module:
        fn = getattr(super(), "get_language_model", None)
        return fn() if fn is not None else self

    def get_lm_head(self) -> nn.Module | None:
        return getattr(self.get_language_model(), "lm_head", None)

    def get_embedding_layer(self) -> nn.Module | None:
        lm = self.get_language_model()
        inner = getattr(lm, "model", lm)
        return getattr(inner, "embed_tokens", None) or getattr(inner, "embedding", None)


def _default_hook(name: str):
    def hook(self: Model, *args: Any, **kwargs: Any) -> Any:
        return self._core_default(name, *args, **kwargs)

    hook.__name__ = hook.__qualname__ = name
    hook.__doc__ = f"`ModelState.{name}`; defaults to core's for this model's kind."
    return hook


for _name in HOOKS:
    setattr(Model, _name, _default_hook(_name))


def overridden(cls: type, name: str) -> bool:
    """Whether `cls` replaces `Model`'s default for hook `name`."""
    return getattr(cls, name, None) is not getattr(Model, name)
