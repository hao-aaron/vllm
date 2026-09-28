# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct loading: host checkpoint-format bytes in live storage.

Check 1 (first touch): a live tensor whose own bytes can hold its
checkpoint-format tensor is given a checkpoint-layout view of those bytes as the
load target, instead of scratch. Check 2 (after PWAL): each result is found to
already sit in the live tensor ("in place"), or is copied (cloned first if it
overlaps live storage).

Off by default (stage-all); VLLM_RELOAD_DIRECT_LOAD=1 enables it.
"""

import os
from collections import Counter
from weakref import WeakKeyDictionary

import torch

DIRECT_LOAD = os.getenv("VLLM_RELOAD_DIRECT_LOAD", "0") == "1"

# Byte alignment required of a hosted view's start (kernels such as CUTLASS
# read 16-byte vectors; fresh allocations are far more aligned than this).
HOSTING_ALIGNMENT = 16

# storage data_ptr -> number of registered (module, name) users, rebuilt at start
_STORAGE_USERS: Counter[int] = Counter()

# Per-module outcomes of the last reload: name -> "hosted"/"scratch:<why>" (plan)
# and "in_place"/"copied"/"overlap"/"not_produced" (landing).
PLAN_OUTCOMES: WeakKeyDictionary[torch.nn.Module, dict[str, str]] = WeakKeyDictionary()
LANDING_OUTCOMES: WeakKeyDictionary[torch.nn.Module, dict[str, str]] = (
    WeakKeyDictionary()
)


def footprint(t: torch.Tensor) -> int:
    """Bytes the tensor itself spans (not the rest of its storage)."""
    if t.numel() == 0:
        return 0
    span = sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
    return span * t.element_size()


def build_storage_users(model: torch.nn.Module) -> None:
    """Count registered users per storage. A tensor object registered under two
    names (tied weights) counts twice, so it is never hosted."""
    _STORAGE_USERS.clear()
    for module in model.modules():
        for tensors in (module._parameters, module._buffers):
            for t in tensors.values():
                if t is None or t.is_meta or t.numel() == 0:
                    continue
                try:
                    _STORAGE_USERS[t.untyped_storage().data_ptr()] += 1
                except (RuntimeError, NotImplementedError):
                    continue


def plan_input(
    meta: torch.Tensor, live: torch.Tensor | None, device: torch.device
) -> str | None:
    """Return None if `live`'s bytes can host `meta` (checkpoint layout), else
    the reason for falling back to scratch. Depends only on metadata."""
    if live is None:
        return "deleted"
    if live.numel() == 0:
        return "released"
    if meta.numel() == 0:
        return "empty"
    if type(live.data) is not torch.Tensor:
        return "tensor subclass"  # e.g. torchao: may not expose plain storage
    if live.device != torch.device(device):
        return "device"
    need = footprint(meta)
    if footprint(live) < need:
        return "too small"
    byte_offset = live.storage_offset() * live.element_size()
    if byte_offset % meta.element_size() or byte_offset % HOSTING_ALIGNMENT:
        return "alignment"
    storage = live.untyped_storage()
    if byte_offset + need > storage.nbytes():
        return "too small"  # set_ would grow and move the storage
    if _STORAGE_USERS.get(storage.data_ptr(), 0) != 1:
        return "shared storage"
    return None


def checkpoint_view(live: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """A checkpoint-layout view of `live`'s bytes carrying the snapshot's
    class and attributes (subclass, output_dim, weight_loader, ...)."""
    storage = live.untyped_storage()
    offset = live.storage_offset() * live.element_size() // meta.element_size()
    view = torch.empty(0, dtype=meta.dtype, device=live.device)
    view.set_(storage, offset, meta.shape, meta.stride())
    assert view.untyped_storage().data_ptr() == storage.data_ptr()
    view.__class__ = meta.__class__
    view.__dict__ = meta.__dict__.copy()
    return view


def same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


def shares_storage(a: torch.Tensor, storages: set[int]) -> bool:
    try:
        return a.untyped_storage().data_ptr() in storages
    except (RuntimeError, NotImplementedError):
        return False


def landing_outcomes(model: torch.nn.Module) -> dict:
    """Summary of the last reload's hosting plan and landing outcomes."""
    plan: Counter[str] = Counter()
    landing: Counter[str] = Counter()
    by_tensor: dict[str, str] = {}
    for name, module in model.named_modules():
        for tname, outcome in PLAN_OUTCOMES.get(module, {}).items():
            plan[outcome] += 1
            by_tensor[f"{name}.{tname}:plan"] = outcome
        for tname, outcome in LANDING_OUTCOMES.get(module, {}).items():
            landing[outcome] += 1
            by_tensor[f"{name}.{tname}:landing"] = outcome
    return {"plan": dict(plan), "landing": dict(landing), "by_tensor": by_tensor}
