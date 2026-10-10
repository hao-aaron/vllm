# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refresh: weight-derived state recomputed in place from the live tensors.

Runs after every load, cold start included, in this order:

1. kernel refresh, per module: a reload-safe quant method's `refresh(layer)`
   (MoE kernel -> quant config -> prepare/finalize -> experts), right after
   its PWAL at cold start and after landing on reload;
2. attention: the impl's `refresh(layer)`, from attention PWAL;
3. layer refreshes that framework layers declare with `register_refresh`
   (state spanning modules, e.g. the MoE runner's fused router gate);
4. `SupportsReload.refresh_model()` of every module that implements it,
   children first;
5. cold start only: `SupportsReload.post_load_setup()`.
"""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

__all__ = [
    "register_refresh",
    "refresh_quant_method",
    "run_model_refresh",
    "run_post_load_setup",
]

_REFRESHES_ATTR = "_vllm_reload_refreshes"


def register_refresh(module: torch.nn.Module, fn: Callable[[], None]) -> None:
    """Declare an in-place refresh of derived state held by a framework layer
    and read across modules. The framework calls it after every load, after
    the kernel and attention refreshes."""
    module.__dict__.setdefault(_REFRESHES_ATTR, []).append(fn)


def refresh_quant_method(layer: torch.nn.Module, quant_method) -> None:
    if (
        isinstance(quant_method, QuantizeMethodBase)
        and getattr(quant_method, "reload_safe", False)
        and hasattr(quant_method, "refresh")
    ):
        quant_method.refresh(layer)


def _children_first(model: torch.nn.Module) -> list[torch.nn.Module]:
    return list(reversed(list(model.modules())))


@torch.no_grad()
def run_model_refresh(model: torch.nn.Module) -> None:
    """Phases 3 and 4: declared layer refreshes, then `refresh_model()`."""
    modules = _children_first(model)
    for module in modules:
        for fn in module.__dict__.get(_REFRESHES_ATTR, ()):
            fn()
    for module in modules:
        if getattr(module, "supports_reload", False):
            module.refresh_model()


def run_post_load_setup(model: torch.nn.Module) -> None:
    """Phase 5 (cold start only)."""
    for module in _children_first(model):
        if getattr(module, "supports_reload", False):
            module.post_load_setup()
