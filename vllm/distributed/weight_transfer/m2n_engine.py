# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-side weight transfer engine built on NCCL M2N (`nccl_m2n`).

Each parameter is moved with a single `nccl.m2n.reshard` from the trainer's
layout (FSDP / TP / EP / arbitrary DTensor sharding) to a layout over the
inference workers, so the trainer sends its local shards and never all-gathers
a full tensor. A parameter travels on the communicator of its source group
(its `owners`): those trainer ranks, then every inference worker.

Where each worker's bytes go comes from the load plan (`plan_loads`, RFC
#64418), built at init: the workers agree, per parameter, on the most sharded
layout that gives every worker the boxes it needs, then reshard straight into
the live tensor when it is one contiguous box, or into a staging buffer that
is copied into the plan's destinations. A parameter the plan cannot write
directly arrives in full and goes through `load_weights`.

Both meshes participate in every reshard, in the same order, so the worker must
be inside `receive_weights` while the trainer is sending. Driving that from the
trainer is the trainer engine's job.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    SKIP,
    Box,
    DstLayout,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    agree_destination_layouts,
    candidate_layouts,
    check_data_plane_agreement,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    check_transferable,
    create_m2n_communicators,
    expected_local_shape,
    import_m2n,
    layout_box,
    prepare_m2n_local_runtime,
    reshard,
    source_groups,
    source_plan_digest,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    decode_nccl_unique_id,
    worker_init_metadata_group,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

__all__ = [
    "M2NWeightTransferInitInfo",
    "M2NWeightTransferUpdateInfo",
    "M2NWeightTransferEngine",
]


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class M2NWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info: the rendezvous plus the full transfer plan.

    Layouts are static for the whole run, so they ride the one-time init
    handshake and the per-round update info stays a list of names. Everything is
    plain JSON so the HTTP control plane carries it unchanged.
    """

    schema_version: int
    master_address: str
    master_port: int
    rank_offset: int
    """First worker rank, i.e. the number of trainer ranks."""
    world_size: int
    """Trainer ranks + all inference workers."""
    source_digest: str
    """SHA-256 of the ordered, versioned source parameter plan."""
    params: list[dict[str, Any]]
    """Ordered per-parameter source metadata and layouts."""
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    max_cta: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.master_address, str) or not self.master_address:
            raise ValueError("`master_address` must name the metadata rendezvous")
        if (
            not isinstance(self.master_port, int)
            or isinstance(self.master_port, bool)
            or not 0 < self.master_port < 65536
        ):
            raise ValueError("`master_port` must be a valid metadata port")
        if (
            not isinstance(self.rank_offset, int)
            or isinstance(self.rank_offset, bool)
            or not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.rank_offset < 1
            or self.rank_offset >= self.world_size
        ):
            raise ValueError(
                f"`rank_offset` ({self.rank_offset}) must leave at least one "
                f"trainer rank and one worker in world_size {self.world_size}"
            )

    def parse_wire_params(self) -> tuple[M2NWireParam, ...]:
        """Parse and authenticate the ordered source plan."""
        if self.schema_version != M2N_WIRE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported nccl_m2n init schema version "
                f"{self.schema_version}; expected {M2N_WIRE_SCHEMA_VERSION}"
            )
        if not isinstance(self.params, list):
            raise TypeError("`params` must be a list of wire parameter mappings")
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise ValueError("`source_digest` must be a non-empty string")
        wire_params: list[M2NWireParam] = []
        names: set[str] = set()
        for index, value in enumerate(self.params):
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"`params[{index}]` must be a mapping, got {type(value).__name__}"
                )
            param = M2NWireParam.from_dict(value)
            if param.name in names:
                raise ValueError(
                    f"nccl_m2n init params contain duplicate parameter '{param.name}'"
                )
            names.add(param.name)
            wire_params.append(param)
        parsed = tuple(wire_params)
        computed_digest = source_plan_digest(parsed)
        if self.source_digest != computed_digest:
            raise ValueError(
                "nccl_m2n init source digest does not match params: "
                f"computed {computed_digest}, declared {self.source_digest}"
            )
        return parsed

    @property
    def nccl_unique_id_bytes(self) -> bytes | None:
        if self.nccl_unique_id_b64 is None:
            return None
        return decode_nccl_unique_id(
            master_address=None,
            master_port=None,
            nccl_unique_id_b64=self.nccl_unique_id_b64,
            ctx="M2NWeightTransferInitInfo data plane",
        )


@dataclass
class M2NWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info: which parameters this chunk carries, in order.

    Shapes, dtypes and layouts were fixed at init, so a round only needs to say
    what is coming and in what order — both sides must issue their reshards in
    exactly that order.
    """

    names: list[str]


# ---------------------------------------------------------------------------
# Destination plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Step:
    """How this worker receives one parameter, fixed at init."""

    kind: str  # "skip", "direct", "staged" or "fallback"
    layout: DstLayout = None
    box: Box = ()
    target: torch.Tensor | None = None  # direct: the live tensor, box-shaped
    copies: tuple[tuple[tuple[slice, ...], torch.Tensor], ...] = ()  # staged


def _covers(outer: Box, inner: Box) -> bool:
    return all(a <= c and d <= b for (a, b), (c, d) in zip(outer, inner))


def accepted_layouts(
    boxes: tuple[Box, ...] | None,
    dest: list[tuple[Box, torch.Tensor]] | None,
    shape: tuple[int, ...],
    candidates: list[DstLayout],
    worker: int,
) -> int:
    """Bit `i` set if candidate `i` delivers this worker everything it needs:
    any layout if it needs nothing (bit `len(candidates)` flags that), the full
    tensor if it falls back to `load_weights`, else a box covering its boxes."""
    if boxes is None:
        return (1 << (len(candidates) + 1)) - 1
    full = tuple((0, size) for size in shape)
    mask = 0
    for i, layout in enumerate(candidates):
        box = layout_box(layout, shape, worker)
        ok = box == full if dest is None else all(_covers(box, b) for b in boxes)
        mask |= ok << i
    return mask


def plan_step(
    layout: DstLayout,
    boxes: tuple[Box, ...] | None,
    dest: list[tuple[Box, torch.Tensor]] | None,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    worker: int,
) -> _Step:
    """This worker's step under the agreed `layout`. A destination in another
    dtype (the loader's `copy_` casts) is staged, so the copy casts too."""
    box = layout_box(layout, shape, worker)
    if dest is None and boxes is not None:
        return _Step("fallback", layout, box)
    dest = dest or []
    box_shape = tuple(b - a for a, b in box)
    if (
        len(dest) == 1
        and dest[0][0] == box
        and dest[0][1].is_contiguous()
        and dest[0][1].dtype == dtype
    ):
        return _Step("direct", layout, box, target=dest[0][1].view(box_shape))
    copies = tuple(
        (tuple(slice(c - a, d - a) for (a, _), (c, d) in zip(box, part)), tensor)
        for part, tensor in dest
    )
    return _Step("staged", layout, box, copies=copies)


def staged_view(staging: torch.Tensor, step: _Step, dtype: torch.dtype) -> torch.Tensor:
    """The front of the flat byte buffer `staging` as `step`'s box."""
    shape = tuple(b - a for a, b in step.box)
    nbytes = prod(shape) * dtype.itemsize
    return staging[:nbytes].view(dtype).view(shape)


def copy_staged(step: _Step, buffer: torch.Tensor) -> None:
    for slices, tensor in step.copies:
        tensor.copy_(buffer[slices].reshape(tensor.shape))


# ---------------------------------------------------------------------------
# Worker engine
# ---------------------------------------------------------------------------


class M2NWeightTransferEngine(
    WeightTransferEngine[M2NWeightTransferInitInfo, M2NWeightTransferUpdateInfo]
):
    """Inference-side engine: receives each parameter with one reshard, into
    the destinations of the load plan built at init."""

    init_info_cls = M2NWeightTransferInitInfo
    update_info_cls = M2NWeightTransferUpdateInfo
    # The load plan is built for the target model at init, so the engine cannot
    # be retargeted at a draft model mid-flight.
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._m2n: Any = None
        self._handle: Any = None
        self._comms: list[Any] = []
        self._metas: list[M2NParamMeta] = []
        self._group_of: list[int] = []
        self._steps: list[_Step] = []
        self._index: dict[str, int] = {}
        self._num_workers = 0
        self._staging_bytes = 0

    def init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Agree on the source plan and the destination layouts, then create one
        communicator per source group."""
        if self._handle is not None or self._comms:
            raise RuntimeError(
                "nccl_m2n weight transfer is already initialized; call "
                "shutdown_weight_transfer_engine() before initializing again"
            )
        rendezvous_info = NCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=init_info.rank_offset,
            world_size=init_info.world_size,
        )
        metadata_group = worker_init_metadata_group(
            rendezvous_info, self.parallel_config
        )
        num_trainers = init_info.rank_offset
        self._num_workers = init_info.world_size - num_trainers
        worker = metadata_group.rank - num_trainers

        wire_params: tuple[M2NWireParam, ...] = ()
        source_error: str | None = None
        try:
            self._m2n = import_m2n()
            wire_params = init_info.parse_wire_params()
            self._prepare_source_plan(wire_params, num_trainers)
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
            self._metas = []
        check_source_plan_agreement(
            metadata_group,
            wire_params,
            expected_digest=init_info.source_digest,
            local_error=source_error,
        )

        # Plan at init: nothing is in flight, and the plan holds no traffic.
        plan: Any = None
        candidates: list[list[DstLayout]] = []
        masks: list[int] | None = None
        destination_error: str | None = None
        try:
            plan = self._plan_loads()
            masks = []
            for meta in self._metas:
                candidates.append(candidate_layouts(meta, self._num_workers))
                masks.append(
                    accepted_layouts(
                        plan.get_boxes(meta.name),
                        plan.get_dest(meta.name),
                        meta.shape,
                        candidates[-1],
                        worker,
                    )
                )
            forbidden = [
                m.name
                for m in self._metas
                if not m.allow_full_fallback
                and plan.get_boxes(m.name) is not None
                and plan.get_dest(m.name) is None
            ]
            if forbidden:
                raise ValueError(
                    f"{len(forbidden)} parameters forbid full fallback but cannot "
                    f"be written directly (e.g. {forbidden[:3]}; reasons in the "
                    "load plan log)"
                )
        except Exception as exc:
            destination_error = f"{type(exc).__name__}: {exc}"
            masks = None
        chosen = agree_destination_layouts(
            metadata_group, self._metas, num_trainers, masks, destination_error
        )
        self._steps = [
            _Step("skip")
            if index == SKIP
            else plan_step(
                cands[index],
                plan.get_boxes(meta.name),
                plan.get_dest(meta.name),
                meta.shape,
                meta.dtype,
                worker,
            )
            for meta, cands, index in zip(self._metas, candidates, chosen)
        ]
        self._staging_bytes = max(
            (
                prod(b - a for a, b in step.box) * meta.dtype.itemsize
                for meta, step in zip(self._metas, self._steps)
                if step.kind == "staged"
            ),
            default=0,
        )
        self._report(worker)

        unique_id_bytes: bytes | None = None
        data_plane_error: str | None = None
        try:
            unique_id_bytes = init_info.nccl_unique_id_bytes
        except Exception as exc:
            data_plane_error = f"{type(exc).__name__}: {exc}"
        check_data_plane_agreement(metadata_group, unique_id_bytes, data_plane_error)

        handle = None
        comms: list[Any] = []
        try:
            nccl = None
            runtime_error: str | None = None
            try:
                nccl, _, handle = prepare_m2n_local_runtime(
                    self._m2n, init_info.max_cta
                )
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
            check_runtime_ready_agreement(metadata_group, runtime_error)
            groups = source_groups(self._metas)
            comms = create_m2n_communicators(
                nccl, metadata_group, unique_id_bytes, groups, num_trainers
            )
        except BaseException:
            for comm in comms:
                try:
                    comm.abort()
                except BaseException:
                    logger.exception("failed to abort an nccl_m2n communicator")
            if handle is not None:
                try:
                    handle.destroy()
                except BaseException:
                    logger.exception(
                        "failed to destroy partially initialized nccl_m2n handle"
                    )
            raise

        group_index = {owners: i for i, owners in enumerate(groups)}
        self._group_of = [group_index[meta.owners] for meta in self._metas]
        self._handle = handle
        self._comms = comms

    def _plan_loads(self) -> Any:
        from vllm.model_executor.model_loader.reload import plan_loads
        from vllm.model_executor.model_loader.reload.direct import (
            direct_reload_enabled,
        )

        if not direct_reload_enabled():
            raise ValueError("nccl_m2n requires VLLM_RELOAD_MODE=direct")
        if self.parallel_config.pipeline_parallel_size > 1:
            raise ValueError("nccl_m2n does not support pipeline parallel workers")
        return plan_loads(self.model, self._metas)

    def _report(self, worker: int) -> None:
        counts: dict[str, int] = {}
        nbytes = 0
        for meta, step in zip(self._metas, self._steps):
            counts[step.kind] = counts.get(step.kind, 0) + 1
            if step.kind != "skip":
                nbytes += prod(b - a for a, b in step.box) * meta.dtype.itemsize
        logger.info(
            "nccl_m2n destination plan (worker %d): %d direct, %d staged, "
            "%d fallback, %d skipped; receives %.3f GiB per update; staging "
            "buffer %.1f MiB",
            worker,
            counts.get("direct", 0),
            counts.get("staged", 0),
            counts.get("fallback", 0),
            counts.get("skip", 0),
            nbytes / (1 << 30),
            self._staging_bytes / (1 << 20),
        )

    def _prepare_source_plan(
        self,
        wire_params: tuple[M2NWireParam, ...],
        num_trainer_ranks: int,
    ) -> None:
        """Validate worker-visible source metadata without constructing NCCL."""
        self._metas = []
        for param in wire_params:
            dtype = getattr(torch, param.dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"parameter '{param.name}' has invalid dtype name "
                    f"'{param.dtype_name}'; expected the name of a torch.dtype"
                )
            check_transferable(param.name, dtype, param.shape)
            if not param.owners or max(param.owners) >= num_trainer_ranks:
                raise ValueError(
                    f"parameter '{param.name}' owners {param.owners} are not "
                    f"among the {num_trainer_ranks} trainer ranks"
                )
            layout = M2NLayout(M2NMesh(param.src_mesh_dims, 0), param.src_placements)
            meta = M2NParamMeta(
                param.name,
                dtype,
                param.shape,
                layout,
                param.allow_full_fallback,
                owners=param.owners,
            )
            expected_local_shape(meta)
            self._metas.append(meta)
        self._index = {meta.name: i for i, meta in enumerate(self._metas)}

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)

    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        """Receive each requested parameter as the init-time plan says."""
        if self._handle is None or not self._comms:
            raise RuntimeError(
                "nccl_m2n weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )
        if not isinstance(update_info.names, list) or any(
            not isinstance(name, str) for name in update_info.names
        ):
            raise TypeError("nccl_m2n update names must be a list of strings")
        missing = [name for name in update_info.names if name not in self._index]
        if missing:
            raise ValueError(
                f"parameter '{missing[0]}' was not declared at init; the "
                "trainer must send the same parameter set it announced"
            )

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        stream = torch.cuda.current_stream()
        # One buffer for every staged name: copies run on the reshard stream, so
        # the next reshard into it is ordered after them.
        staging = (
            torch.empty(self._staging_bytes, dtype=torch.uint8, device=self.device)
            if self._staging_bytes
            else None
        )
        with disable_mtp_completeness_check():
            for name in update_info.names:
                index = self._index[name]
                meta, step = self._metas[index], self._steps[index]
                if step.kind == "skip":
                    continue
                if step.kind == "direct":
                    buffer = step.target
                elif step.kind == "staged":
                    assert staging is not None
                    buffer = staged_view(staging, step, meta.dtype)
                else:
                    buffer = torch.empty(
                        meta.shape, dtype=meta.dtype, device=self.device
                    )
                reshard(
                    self._m2n,
                    self._handle,
                    self._comms[self._group_of[index]],
                    stream,
                    meta,
                    step.layout,
                    self._num_workers,
                    None,
                    buffer,
                )
                copy_staged(step, buffer)
                if step.kind == "fallback":
                    self.model.load_weights([(name, buffer)])

    def shutdown(self) -> None:
        """Finish pending GPU work and release M2N communication state."""
        handle, self._handle = self._handle, None
        comms, self._comms = self._comms, []
        self._m2n = None
        self._metas = []
        self._steps = []
        self._group_of = []
        self._index = {}
        self._staging_bytes = 0

        first_error: BaseException | None = None

        def attempt(action: Any, message: str) -> None:
            nonlocal first_error
            try:
                action()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                logger.exception(message)

        if handle is not None or comms:
            # M2N does not synchronize caller streams on finalize.
            attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
        if handle is not None:
            attempt(handle.destroy, "failed to destroy nccl_m2n handle")
        for comm in comms:
            attempt(comm.abort, "failed to abort nccl_m2n communicator")
        if first_error is not None:
            raise first_error
