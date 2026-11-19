# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core's automatic wrapper dispatching to a `vllm.specialized.Model`'s hooks."""

import functools
from typing import Any

from vllm.specialized.model import HOOKS, LIFECYCLE, Model, overridden
from vllm.v1.worker.gpu.model_states.interface import ModelState


def _forward(name: str):
    def hook(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.specialized, name)(*args, **kwargs)

    hook.__name__ = hook.__qualname__ = name
    return hook


@functools.cache
def dispatching(
    base: type[ModelState], model_cls: type[Model], wrapper_cls: type | None = None
) -> type[ModelState]:
    """`base` with each hook `model_cls` overrides routed to the model, except
    hooks a general wrapper class implements itself."""
    names = [
        n
        for n in HOOKS + LIFECYCLE
        if overridden(model_cls, n) and not hasattr(wrapper_cls, n)
    ]
    attrs: dict[str, Any] = {n: _forward(n) for n in names}

    def bind(self, model: Model, kind: type[ModelState]) -> None:
        self.specialized = model
        # The model's super() calls reach the kind's own implementation.
        model.__dict__["_core"] = (self, kind)

    attrs["bind"] = bind
    return type(f"{model_cls.__name__}State", (base,), attrs)
