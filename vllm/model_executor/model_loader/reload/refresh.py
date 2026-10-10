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
    "declare_constant",
    "declare_derived",
    "declared_tensors",
    "register_refresh",
    "refresh_quant_method",
    "run_model_refresh",
    "run_post_load_setup",
]

_REFRESHES_ATTR = "_vllm_reload_refreshes"
_DERIVED_ATTR = "_vllm_reload_derived"
_CONSTANT_ATTR = "_vllm_reload_constants"


def declare_derived(holder: object, *tensors: torch.Tensor) -> None:
    """Declare weight-derived tensors an object (a kernel, a quant config)
    holds outside the module registry: allocated once, rewritten in place by a
    declared refresh. The reload verifier's ownership check accepts them."""
    holder.__dict__.setdefault(_DERIVED_ATTR, []).extend(tensors)


def declare_constant(holder: object, *tensors: torch.Tensor) -> None:
    """Declare tensors an object holds that do not depend on weights
    (workspaces, lookup tables): a weight update leaves them valid."""
    holder.__dict__.setdefault(_CONSTANT_ATTR, []).extend(tensors)


def declared_tensors(holder: object) -> list[torch.Tensor]:
    """Tensors `holder` declared (derived or constant, or #60009's
    `persistent_tensors()`)."""
    d = getattr(holder, "__dict__", None)
    if not isinstance(d, dict):
        return []
    out = [*d.get(_DERIVED_ATTR, ()), *d.get(_CONSTANT_ATTR, ())]
    persistent = getattr(type(holder), "persistent_tensors", None)
    if callable(persistent):
        out += list(persistent(holder).values())
    return out


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
