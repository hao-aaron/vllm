# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executable contract of a shared op: the kernel, a plain-PyTorch
reference, and the cells it is tested on (tests/models/common/ops)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Cell:
    """One tested configuration: builds the op's arguments on a device."""

    reason: str
    args: Callable[[torch.device], tuple[Any, ...]]


class SharedOp:
    # Dtypes, scale layouts and fusions the op provides.
    SUPPORTS: Mapping[str, Any] = {}
    CELLS: tuple[Cell, ...] = ()
    # GPUs the cells ran on; "cpu" for ops without a device kernel.
    ARCHS: frozenset[str] = frozenset()

    def __call__(self, *args: Any) -> Any:
        raise NotImplementedError

    def reference(self, *args: Any) -> Any:
        """The same computation in plain PyTorch, accumulating in fp32."""
        raise NotImplementedError


REGISTRY: list[SharedOp] = []


def shared_op(cls: type[SharedOp]) -> SharedOp:
    """Register the op's instance, which callers use as the function."""
    op = cls()
    REGISTRY.append(op)
    return op
