# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load coverage: does every loadable param go through its `weight_loader`?

A streamed reload restores params on meta and only sees writes made through
their (wrapped) weight loaders. A `load_weights` that writes a param directly
(`param.data.copy_(...)`) writes a meta placeholder on reload and the param
silently keeps its old value. Observing one real cold load finds those:

- direct writes: the param changed, but no loader call reached it;
- never loaded: neither (absent from the checkpoint, or a mapper dropped it).

Opt-in (`VLLM_RELOAD_OBSERVE_LOAD=1`, for CI): it keeps a CPU copy of every
param during the load.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

__all__ = ["LoadCoverage", "OBSERVE_LOAD", "observe_load"]

OBSERVE_LOAD = os.getenv("VLLM_RELOAD_OBSERVE_LOAD", "0") == "1"


@dataclass
class LoadCoverage:
    loaded: set[str] = field(default_factory=set)
    direct_writes: list[str] = field(default_factory=list)
    never_loaded: list[str] = field(default_factory=list)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).contiguous().view(torch.uint8).to("cpu", copy=True)


@contextmanager
def observe_load(model: torch.nn.Module) -> Iterator[LoadCoverage]:
    coverage = LoadCoverage()
    names: dict[int, str] = {}
    params: dict[int, torch.nn.Parameter] = {}
    for module_name, module in model.named_modules():
        for pname, p in module._parameters.items():
            if p is None or p.is_meta or id(p) in names:
                continue
            names[id(p)] = f"{module_name}.{pname}" if module_name else pname
            params[id(p)] = p
    before = {key: _bytes(p) for key, p in params.items()}
    restore: list[tuple[torch.nn.Parameter, object]] = []
    for key, p in params.items():
        original = getattr(p, "weight_loader", None)
        restore.append((p, original))
        p.weight_loader = _observed(
            original or default_weight_loader, coverage, names[key]
        )
    try:
        yield coverage
    finally:
        for p, original in restore:
            if original is None:
                del p.weight_loader
            else:
                p.weight_loader = original
    for key, p in params.items():
        name = names[key]
        if name in coverage.loaded:
            continue
        if torch.equal(before[key], _bytes(p)):
            coverage.never_loaded.append(name)
        else:
            coverage.direct_writes.append(name)


def _observed(loader, coverage: LoadCoverage, name: str):
    @wraps(loader)
    def observed_loader(*args, **kwargs):
        coverage.loaded.add(name)
        return loader(*args, **kwargs)

    return observed_loader
