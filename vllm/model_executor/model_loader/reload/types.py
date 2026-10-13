# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from inspect import BoundArguments

import torch

__all__ = ["LayerTensors", "LayerReloadingInfo", "ReloadSession"]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


@dataclass
class ReloadSession:
    """Model-wide integrity state of weight updates (no module references).

    Modules are named relative to the model the session belongs to."""

    # between start and finish/abort: no forward may run
    active: bool = False
    # an update failed after it started; the engine must not serve until an
    # update succeeds that re-completes every module in `dirty_modules`
    dirty: bool = False
    # modules a failed update completed (landed new values in their live
    # tensors) or wrote outside a loader; accumulates until dirty clears
    dirty_modules: set[str] = field(default_factory=set)
    # this update writes only part of each module it touches, in kernel
    # layouts (sparse NCCL): it never clears dirty
    sparse: bool = False
    # True: a partial update. False: a full update, every required module must
    # receive a loader call. None: unspecified (untouched modules are logged)
    partial: bool | None = None
    # a partial update's module names (prefixes), agreed by all ranks at start
    modules: tuple[str, ...] | None = None
    # modules that received a loader call (or a planned write) in this update
    touched: set[str] = field(default_factory=set)
    # (module, tensor) pairs that received a loader call in this update
    touched_tensors: set[tuple[str, str]] = field(default_factory=set)
    # modules this update completed (every tensor the cold load wrote got a
    # loader call), or wrote by plan outside a loader
    completed: set[str] = field(default_factory=set)
    # modules this update landed without completing them (finalize processes
    # partially loaded modules): written, but they don't re-complete anything
    written: set[str] = field(default_factory=set)
    # the coverage check of this update already ran
    coverage_checked: bool = False


@dataclass
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # create-time value of each tensor created uniform (zeros, ones, scale
    # sentinels), recorded by `record_metadata_for_reloading`: reload targets
    # start from it, as at model init (others start zero-filled)
    init_fill: dict[str, float] = field(default_factory=dict)

    # track how many elements are ready for loading, used by `online_process_loader`
    load_numel: int = 0
    load_numel_total: int | None = None

    # used by `online_process_loader` to buffer args and tensors until ready to load
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # non-persistent buffer names captured with `kernel_tensors`, so buffer
    # persistence survives `_non_persistent_buffers_set` being mutated during reload
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)

    # the model's session and this module's name in it, set by
    # `initialize_layerwise_reload`; they survive `reset`
    session: ReloadSession | None = None
    name: str = ""

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
            init_fill=self.init_fill,
            session=self.session,
            name=self.name,
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
