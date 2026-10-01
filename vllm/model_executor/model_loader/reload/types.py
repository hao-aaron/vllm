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
    """Model-wide state of one weight update (holds no module references)."""

    # a live tensor was written and the update has not finished successfully;
    # the engine must not serve until a full update succeeds
    dirty: bool = False
    # names of modules left incomplete at finish (integrity report)
    incomplete: list[str] = field(default_factory=list)
    # between start and finish/abort: params are on meta / hold checkpoint bytes
    active: bool = True
    # True: a partial update (modules may receive nothing). False: a full
    # update (every module with checkpoint tensors must receive weights).
    # None: unspecified (modules that received nothing are reported)
    partial: bool | None = None
    # id(module) -> name, for modules a full update must touch
    required_modules: dict[int, str] = field(default_factory=dict)


@dataclass
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # create-time values of small tensors (e.g. FP8 scale sentinels), used to
    # initialize reload targets instead of zeros, recorded with the metadata
    init_values: dict[str, torch.Tensor] = field(default_factory=dict)

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

    # names of tensors materialized (first touch) this round, and their bytes
    materialized: set[str] = field(default_factory=set)
    scratch_bytes: int = 0

    # names of tensors that received at least one loader call this round
    loaded_names: set[str] = field(default_factory=set)

    # the update this module belongs to, set by `initialize_layerwise_reload`
    session: ReloadSession | None = None

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
            init_values=self.init_values,
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
