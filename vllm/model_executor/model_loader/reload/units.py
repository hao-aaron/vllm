# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arrival units: what one weight-loader call delivered.

A unit names a part of a module's tensor that one loader call wrote:

- `(tensor, "*")`: the whole tensor (a plain parameter, a pre-fused checkpoint
  tensor, a stacked `[E, ...]` expert tensor);
- `(tensor, shard)`: one shard of a fused linear (`QKVParallelLinear`,
  `MergedColumnParallelLinear`), from the loader's `loaded_shard_id`;
- `(tensor, (shard, expert))`: one expert's shard of a routed-experts tensor.

The reload core wraps every loadable tensor's `weight_loader`. A loader may
report its units explicitly (`report_units`) or that it declined data that
isn't local to this rank (`report_declined`); otherwise the wrapper infers the
unit from the loader's arguments (`loaded_shard_id`, `shard_id` + `expert_id`),
or records `(tensor, "*")`. A unit is recorded only after the loader returned.

Each unit is recorded with the checkpoint name that produced it (the name the
model's `load_weights` pulled last, see `track_names`). The expected units of
an update are a multiset of `(unit, name)` pairs, so legitimate repeats (MTP
layers sharing `embed_tokens`, tied embeddings sent under both names) are
expected rather than exempted.
"""

import inspect
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

__all__ = [
    "ExpectedUnits",
    "LoadCall",
    "current_load",
    "infer_shard",
    "report_declined",
    "report_units",
    "track_names",
]

# (tensor name within its module, shard key)
Unit = tuple[str, Hashable]
WILDCARD = None  # a unit recorded without a checkpoint name matches any name


@dataclass
class LoadCall:
    """The wrapped loader call currently running."""

    module: torch.nn.Module
    tensor_name: str
    units: list[Unit] = field(default_factory=list)
    declined: bool = False


_CURRENT_LOAD: ContextVar[LoadCall | None] = ContextVar(
    "vllm_reload_current_load", default=None
)
_CURRENT_NAME: ContextVar[str | None] = ContextVar(
    "vllm_reload_current_name", default=None
)
# set while the framework itself writes (post-load processing, landing)
FRAMEWORK_WRITES: ContextVar[bool] = ContextVar(
    "vllm_reload_framework_writes", default=False
)


def current_load() -> LoadCall | None:
    """The (module, tensor) the running wrapped loader writes, if any."""
    return _CURRENT_LOAD.get()


def report_units(*shards: Hashable) -> None:
    """Report the shards of the current tensor this loader call wrote (call it
    from a loader that writes one tensor over several calls and whose
    arguments don't say which part each call writes)."""
    call = _CURRENT_LOAD.get()
    if call is not None:
        call.units.extend((call.tensor_name, s) for s in shards)


def report_declined() -> None:
    """Report that this loader call wrote nothing: its data belongs to another
    rank (e.g. a non-local expert)."""
    call = _CURRENT_LOAD.get()
    if call is not None:
        call.declined = True


def track_names(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass `weights` through, recording the name of the checkpoint tensor
    the model's `load_weights` pulled last (units are attributed to it)."""
    for name, tensor in weights:
        _CURRENT_NAME.set(name)
        yield name, tensor
    _CURRENT_NAME.set(None)


def current_checkpoint_name() -> str | None:
    return _CURRENT_NAME.get()


def infer_shard(bound: inspect.BoundArguments) -> Hashable:
    """The unit's shard key from a loader's arguments."""
    args = bound.arguments
    if "expert_id" in args and "shard_id" in args:
        return (args["shard_id"], args["expert_id"])
    for key in ("loaded_shard_id", "shard_id"):
        if key in args:
            shard = args[key]
            return "*" if shard is None else shard
    return "*"


@dataclass
class ExpectedUnits:
    """The units an update is expected to deliver, per module, from one
    source: a trace over the trainer's or the checkpoint's metadata, the cold
    load, or the module structure."""

    source: str
    # module name -> Counter of (tensor, shard, checkpoint name or WILDCARD)
    units: dict[str, Counter] = field(default_factory=dict)
    # module name -> (tensor, shard) pairs whose absence never fails an update
    optional: dict[str, set[tuple[str, Hashable]]] = field(default_factory=dict)
    # a sender's exact contract (trainer / checkpoint metadata), or inferred
    exact: bool = False

    def add(self, module: str, unit: Unit, name: str | None) -> None:
        self.units.setdefault(module, Counter())[(unit[0], unit[1], name)] += 1
