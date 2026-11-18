# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash. Import-light: model classes load on first access, so routing
can read `routing.py` and the specialized declarations without them."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
    from .common.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        raise NotImplementedError("GLM-5.3-Flash does not currently support XPU.")
    if name == "Glm5NextMTP":
        from .common.mtp import Glm5NextMTP

        return Glm5NextMTP
    from .common import model

    return getattr(model, name)
