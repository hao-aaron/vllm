# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-specific steps around a generic layer's post-load processing."""

from collections.abc import Callable

import torch

__all__ = ["wrap_quant_method"]

Step = Callable[[torch.nn.Module], None]

# one wrapper class per quant method class
_WRAPPED_CLASSES: dict[type, type] = {}


def wrap_quant_method(
    layer: torch.nn.Module,
    *,
    before_pwal: Step | None = None,
    after_pwal: Step | None = None,
) -> None:
    """Run model-specific steps right before and after `layer`'s own
    `process_weights_after_loading`, on every load (cold start and reload).

    `before_pwal(layer)` sees the freshly loaded checkpoint-format tensors;
    `after_pwal(layer)` sees the PWAL results, which land like any PWAL
    result on reload (no double application, no extra copy). Steps follow the
    PWAL rules: transform the tensors they are given, allocate long-lived
    state once and refill it in place afterwards.

    This replaces monkeypatching `quant_method.process_weights_after_loading`
    on an instance and model-hook rewrites of a layer's processed tensors.
    The method keeps its class (as a subclass), so isinstance checks and its
    reload declarations are unchanged. Call it at construction, after the
    layer's quant method is set.
    """
    method = layer.quant_method
    if not getattr(method, "_pwal_wrapped", False):
        base = type(method)
        wrapped = _WRAPPED_CLASSES.get(base)
        if wrapped is None:
            wrapped = type(
                f"Wrapped{base.__name__}",
                (base,),
                {
                    "process_weights_after_loading": _wrapped_pwal,
                    "_pwal_wrapped": True,
                    "__module__": base.__module__,
                },
            )
            _WRAPPED_CLASSES[base] = wrapped
        method.__class__ = wrapped
        method._pwal_before = []
        method._pwal_after = []
    if before_pwal is not None:
        method._pwal_before.append(before_pwal)
    if after_pwal is not None:
        method._pwal_after.append(after_pwal)


def _wrapped_pwal(self, layer: torch.nn.Module) -> None:
    for step in self._pwal_before:
        step(layer)
    super(type(self), self).process_weights_after_loading(layer)
    for step in self._pwal_after:
        step(layer)
