# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The API specialized model directories use to talk to core.

`declare` is import-light; `Model`, `StepResult` and `services` load lazily so
importing a declaration never pulls in model or distributed code.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.specialized import services
    from vllm.specialized.model import Model, StepResult

__all__ = ["Model", "StepResult", "options", "services"]


def __getattr__(name: str) -> Any:
    if name in ("Model", "StepResult"):
        from vllm.specialized import model

        return getattr(model, name)
    if name == "services":
        import importlib

        return importlib.import_module("vllm.specialized.services")
    if name == "options":
        from vllm.specialized._routing import options

        return options
    raise AttributeError(name)
