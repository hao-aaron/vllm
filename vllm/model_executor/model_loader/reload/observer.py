# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write observer: every mutating op on a model tensor during `load_weights`.

Two modes:

- Trace mode (always on, during the meta trace at init and the observed cold
  load): flags writes to model tensors while no wrapped loader runs. A model
  that writes outside its loaders (finalizes inside `load_weights`, or copies
  into params directly) can't be updated by the reload core.
- Full checking (CI, `VLLM_RELOAD_OBSERVE_WRITES=1`, on real updates): every
  loader call that reported units wrote its target, wrote nothing outside its
  own module, and a repeated unit wrote identical bytes.

Tensors are identified by their storage (`untyped_storage()._cdata`), which
`.data`, `narrow` and other views share, on meta tensors too.
"""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm.envs as envs

from .units import FRAMEWORK_WRITES, current_load

__all__ = ["OBSERVE_WRITES", "WriteObserver", "storage_key", "suspended"]

OBSERVE_WRITES = envs.VLLM_RELOAD_OBSERVE_WRITES


def storage_key(t: torch.Tensor) -> int | None:
    try:
        return t.untyped_storage()._cdata
    except (RuntimeError, NotImplementedError, AttributeError):
        return None


def _written_tensors(func, args, kwargs) -> list[torch.Tensor]:
    """The tensors `func` writes (its mutable arguments)."""
    schema = getattr(func, "_schema", None)
    if schema is None:
        return []
    out = []
    for i, arg in enumerate(schema.arguments):
        alias = arg.alias_info
        if alias is None or not alias.is_write:
            continue
        value = args[i] if i < len(args) else kwargs.get(arg.name)
        if isinstance(value, torch.Tensor):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(v for v in value if isinstance(v, torch.Tensor))
    return out


@dataclass
class WriteRecord:
    # "module.tensor" names written while no wrapped loader ran
    outside_loader: set[str] = field(default_factory=set)
    # id(LoadCall) -> owner names it wrote
    by_call: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))


class WriteObserver(TorchDispatchMode):
    """Attribute every write to a registered model tensor to the running
    wrapped loader call, or flag it as a write outside any loader."""

    def __init__(self, owners: dict[int, str], dry_run: bool = False):
        super().__init__()
        self.owners = owners  # storage key -> "module.tensor"
        self.record = WriteRecord()
        # a trace: data-less (meta) copies into tensors reload leaves live
        # (e.g. SKIP_LOAD_TENSORS) are dropped instead of failing
        self.dry_run = dry_run

    def __enter__(self):
        self._token = _OBSERVING.set(True)
        return super().__enter__()

    def __exit__(self, *args):
        _OBSERVING.reset(self._token)
        return super().__exit__(*args)

    def register(self, t: torch.Tensor, owner: str) -> None:
        key = storage_key(t)
        if key is not None:
            self.owners[key] = owner

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if FRAMEWORK_WRITES.get():
            return func(*args, **kwargs)
        if (
            self.dry_run
            and func is torch.ops.aten.copy_.default
            and args[1].is_meta
            and not args[0].is_meta
        ):
            return args[0]
        for t in _written_tensors(func, args, kwargs):
            key = storage_key(t)
            owner = None if key is None else self.owners.get(key)
            if owner is None:
                continue
            call = current_load()
            if call is None:
                self.record.outside_loader.add(owner)
            else:
                self.record.by_call[id(call)].add(owner)
        return func(*args, **kwargs)


@contextmanager
def suspended() -> Iterator[None]:
    """Run a wrapped loader without the observer: writes inside a loader are
    the loader's own, and loaders may compile (torch.compile under a dispatch
    mode recurses)."""
    if not _OBSERVING.get():
        yield
        return
    from torch.utils._python_dispatch import _disable_current_modes

    with _disable_current_modes():
        yield


_OBSERVING: ContextVar[bool] = ContextVar("vllm_reload_observing", default=False)
