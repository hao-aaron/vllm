# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The step metadata the executor plans and runs on: kept by the model's
`prepare_attn` hook, and current while a planned step is captured."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple

import torch


class AttnStep(NamedTuple):
    metadata: dict[str, Any]  # attention metadata by layer name
    slots: dict[str, torch.Tensor]  # slot mappings by layer name
    full_graph: bool  # a FULL-graph step or capture


_current: list[AttnStep | None] = [None]


def current() -> AttnStep:
    """The planned step being captured."""
    step = _current[0]
    assert step is not None, "no planned step"
    return step


@contextmanager
def planned(step: AttnStep) -> Iterator[None]:
    _current[0] = step
    try:
        yield
    finally:
        _current[0] = None
