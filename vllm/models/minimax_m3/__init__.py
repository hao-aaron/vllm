# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 model — hardware-isolated entry point.

The implementation lives under `nvidia/` and `amd/`; classes load on first
access from the current platform's folder, so routing can read `routing.py`
and the specialized declarations without them.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.dspark import MiniMaxM3DSparkForCausalLM
    from .nvidia.model import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .nvidia.mtp import MiniMaxM3MTP

__all__ = [
    "MiniMaxM3DSparkForCausalLM",
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    import importlib

    from vllm.platforms import current_platform

    folder = "amd" if current_platform.is_rocm() else "nvidia"
    module = {"MiniMaxM3MTP": "mtp", "MiniMaxM3DSparkForCausalLM": "dspark"}.get(
        name, "model"
    )
    return getattr(importlib.import_module(f"{__name__}.{folder}.{module}"), name)
