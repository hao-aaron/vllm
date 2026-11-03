# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side weight transfer engine for the NCCL M2N backend.

Symmetric to `M2NWeightTransferEngine` but in the training process. Every
trainer rank joins the shared communicator and runs every reshard, sending its
own local shard; only rank 0 touches the inference control plane.

Workers join the communicator during init and run reshard from inside
`update_weights`, so those two RPCs overlap with work on this side. Both run on
a helper thread and are joined afterwards, which keeps that overlap inside the
engine -- where the `TrainerWeightTransferEngine` contract puts it -- rather than
in every caller.
"""

import socket
from collections.abc import Collection, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import torch

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    SKIP,
    DstLayout,
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
    m2n_param_meta,
    prepare_m2n_local_runtime,
    reshard,
    source_groups,
    source_plan_digest,
)
from vllm.distributed.weight_transfer.nccl_common import (
    decode_nccl_unique_id,
    stateless_init_metadata_group,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup

logger = init_logger(__name__)

__all__ = ["M2NTrainerInitInfo", "M2NTrainerWeightTransferEngine"]


@dataclass(kw_only=True)
class M2NTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for nccl_m2n.

    `rank` (from `TrainerInitInfo`) is this trainer process's rank; it is also
    its rank in the shared communicator, since the trainer occupies
    `[0, num_trainer_ranks)`. Rank 0 drives the control plane.
    """

    backend: ClassVar[str] = "nccl_m2n"

    master_address: str
    master_port: int
    world_size: int
    """Trainer ranks + all inference workers."""
    num_trainer_ranks: int = 1
    forbid_full_fallback: Collection[str] = ()
    """Names that must not arrive as full tensors for `load_weights` on any
    worker; init fails if the workers' load plan cannot write one directly."""
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    """Optional UID for the NCCL data plane; TCP remains the metadata plane."""
    max_cta: int | None = None
    listen_socket: socket.socket | None = field(default=None, repr=False, compare=False)
    """Optional rank-zero TCPStore reservation; never sent to workers."""

    def __post_init__(self) -> None:
        """Reject rank counts or destination meshes that cannot form a group."""
        if not isinstance(self.master_address, str) or not self.master_address:
            raise ValueError("`master_address` must name the metadata rendezvous")
        if (
            not isinstance(self.master_port, int)
            or isinstance(self.master_port, bool)
            or not 0 < self.master_port < 65536
        ):
            raise ValueError("`master_port` must be a valid metadata port")
        if not 0 < self.num_trainer_ranks < self.world_size:
            raise ValueError(
                f"`num_trainer_ranks` ({self.num_trainer_ranks}) must leave at "
                f"least one worker in world_size {self.world_size}"
            )
        if not 0 <= self.rank < self.num_trainer_ranks:
            raise ValueError(
                f"trainer `rank` ({self.rank}) must be below "
                f"`num_trainer_ranks` ({self.num_trainer_ranks})"
            )
        if self.listen_socket is not None:
            if self.rank != 0:
                raise ValueError("only trainer rank zero may reserve the TCPStore")
            if not isinstance(self.listen_socket, socket.socket):
                raise TypeError("listen_socket must be a socket.socket")
            if self.listen_socket.fileno() < 0:
                raise ValueError("listen_socket is already closed")

    @property
    def nccl_unique_id_bytes(self) -> bytes | None:
        if self.nccl_unique_id_b64 is None:
            return None
        return decode_nccl_unique_id(
            master_address=None,
            master_port=None,
            nccl_unique_id_b64=self.nccl_unique_id_b64,
            ctx="M2NTrainerInitInfo data plane",
        )


class M2NTrainerWeightTransferEngine(TrainerWeightTransferEngine[M2NTrainerInitInfo]):
    """Trainer-side engine: sends every rank's local shard, once per parameter.

    Called on every trainer rank. All ranks join the communicator and run every
    reshard; only rank 0 touches the control plane. The source's `iter_local()`
    supplies the shards and its metadata the layouts; every parameter must be
    owned by all trainer ranks, in rank order.
    """

    init_info_cls = M2NTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool,
        rank: int = 0,
    ) -> None:
        """Hold the client and source; `trainer_init` fills in the group."""
        super().__init__(client=client, source=source, is_sender=is_sender, rank=rank)
        self._comms: list[Any] = []
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._wire_params: list[M2NWireParam] = []
        self._source_digest = ""
        self._layouts: list[DstLayout | None] = []  # None also for skipped names
        self._skip: list[bool] = []
        self._group_of: list[int] = []
        self._metadata_group: StatelessProcessGroup | None = None
        self._num_trainer_ranks = 0
        self._num_workers = 0

    @classmethod
    def trainer_init(
        cls,
        init_info: M2NTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        """Build the engine and rendezvous with the inference side.

        Runs on *every* trainer rank. Rank 0 additionally ships the transfer
        plan to the workers and drives the control plane; the other ranks only
        build local state and join the communicator. Every rank participates
        in every reshard and sends its local shard.

        Build a local validation envelope, start the worker RPC without
        waiting, join the metadata rendezvous, and require unanimous source,
        data-plane, and runtime readiness before constructing NCCL.
        """
        if source is None:
            raise ValueError(
                "nccl_m2n trainer weight transfer requires a WeightSource."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            rank=init_info.rank,
        )
        engine._initialize(init_info, source)
        return engine

    def _initialize(
        self,
        init_info: M2NTrainerInitInfo,
        source: WeightSource,
    ) -> None:
        """Run source validation, metadata agreement, and NCCL rendezvous."""
        self._num_trainer_ranks = init_info.num_trainer_ranks
        local_source_error: str | None = None
        try:
            if not source.implements_iter_local():
                raise TypeError(
                    "nccl_m2n sends each rank's local shard, but "
                    f"{type(source).__name__} does not implement iter_local()"
                )
            self._m2n = import_m2n()
            self._prepare_source_plan(
                source, init_info.num_trainer_ranks, init_info.forbid_full_fallback
            )
        except Exception as exc:
            local_source_error = f"{type(exc).__name__}: {exc}"
            self._metas = []
            self._wire_params = []
            self._source_digest = source_plan_digest([])

        executor: ThreadPoolExecutor | None = None
        init_future: Future[None] | None = None
        if self.is_sender:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="m2n-weight-sync"
            )
            init_future = executor.submit(
                self.client.init_weight_transfer_engine,
                self._worker_init_info(init_info),
            )
        handle = None
        comms: list[Any] = []
        try:
            reservation = init_info.listen_socket
            try:
                metadata_group = stateless_init_metadata_group(
                    init_info.master_address,
                    init_info.master_port,
                    init_info.rank,
                    init_info.world_size,
                    listen_socket=reservation,
                )
            finally:
                init_info.listen_socket = None
                if reservation is not None:
                    reservation.close()
            plan_digest = check_source_plan_agreement(
                metadata_group,
                self._wire_params,
                expected_digest=self._source_digest,
                local_error=local_source_error,
            )

            # The workers pick each parameter's destination layout from their
            # load plans; every rank derives the same layouts from the gather.
            self._num_workers = init_info.world_size - init_info.num_trainer_ranks
            chosen = agree_destination_layouts(
                metadata_group, self._metas, init_info.num_trainer_ranks, None
            )
            self._skip = [index == SKIP for index in chosen]
            self._layouts = [
                None if index == SKIP else layouts[index]
                for index, layouts in zip(
                    chosen,
                    (candidate_layouts(m, self._num_workers) for m in self._metas),
                )
            ]
            unique_id_bytes: bytes | None = None
            data_plane_error: str | None = None
            try:
                unique_id_bytes = init_info.nccl_unique_id_bytes
            except Exception as exc:
                data_plane_error = f"{type(exc).__name__}: {exc}"
            check_data_plane_agreement(
                metadata_group,
                unique_id_bytes,
                data_plane_error,
            )

            runtime_error: str | None = None
            nccl = None
            device = None
            try:
                nccl, device, handle = prepare_m2n_local_runtime(
                    self._m2n, init_info.max_cta
                )
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
            check_runtime_ready_agreement(metadata_group, runtime_error)
            assert nccl is not None and device is not None and handle is not None
            groups = source_groups(self._metas)
            comms = create_m2n_communicators(
                nccl,
                metadata_group,
                unique_id_bytes,
                groups,
                init_info.num_trainer_ranks,
            )
            group_index = {owners: i for i, owners in enumerate(groups)}
            self._group_of = [group_index[m.owners] for m in self._metas]

            if init_future is not None:
                init_future.result()

            logger.info(
                "nccl_m2n trainer ready: %d parameters, trainer ranks [0, %d), "
                "inference ranks [%d, %d)",
                len(self._metas),
                init_info.num_trainer_ranks,
                init_info.num_trainer_ranks,
                init_info.world_size,
            )
            logger.info("nccl_m2n source plan digest: %s", plan_digest)
        except BaseException:
            for comm in comms:
                if comm is None:
                    continue
                try:
                    # Transport cleanup only; reload validity belongs to the
                    # model-update lifecycle, not this backend.
                    comm.abort()
                except BaseException:
                    logger.exception(
                        "failed to abort partially initialized nccl_m2n communicator"
                    )
            if handle is not None:
                try:
                    handle.destroy()
                except BaseException:
                    logger.exception(
                        "failed to destroy partially initialized nccl_m2n handle"
                    )
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

        self._metadata_group = metadata_group
        self._handle = handle
        self._comms = comms

    def _worker_init_info(self, init_info: M2NTrainerInitInfo) -> dict[str, Any]:
        """Handshake payload: rendezvous, both meshes, and the transfer plan."""
        payload: dict[str, Any] = {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "master_address": init_info.master_address,
            "master_port": init_info.master_port,
            "nccl_unique_id_b64": init_info.nccl_unique_id_b64,
            "rank_offset": init_info.num_trainer_ranks,
            "world_size": init_info.world_size,
            "source_digest": self._source_digest,
            "params": [param.to_dict() for param in self._wire_params],
            "max_cta": init_info.max_cta,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def _prepare_source_plan(
        self,
        source: WeightSource,
        num_trainer_ranks: int,
        forbid_full_fallback: Collection[str] = (),
    ) -> None:
        """Build and validate this rank's immutable source plan."""
        forbidden = set(forbid_full_fallback)
        self._metas = [
            m2n_param_meta(meta, num_trainer_ranks, meta.name not in forbidden)
            for meta in source.metadata
        ]
        names: set[str] = set()
        for meta in self._metas:
            if meta.name in names:
                raise ValueError(
                    "nccl_m2n source metadata contains duplicate parameter "
                    f"'{meta.name}'"
                )
            names.add(meta.name)
            check_transferable(meta.name, meta.dtype, meta.shape)
            expected_local_shape(meta)
        self._wire_params = [
            M2NWireParam(
                name=meta.name,
                dtype_name=str(meta.dtype).split(".")[-1],
                shape=meta.shape,
                src_mesh_dims=meta.source_layout.mesh.dims,
                src_placements=meta.source_layout.placements,
                owners=meta.owners,
                allow_full_fallback=meta.allow_full_fallback,
            )
            for meta in self._metas
        ]
        self._source_digest = source_plan_digest(self._wire_params)

    def send_weights(self) -> None:
        """Drive one update round: start, reshard concurrently, then finish."""
        if self._handle is None or not self._comms:
            raise RuntimeError("nccl_m2n trainer engine is not initialized")

        if not self.is_sender:
            try:
                self._send()
            except BaseException:
                try:
                    self._receive_round_outcome()
                except BaseException:
                    logger.exception(
                        "failed to consume nccl_m2n round outcome after local failure"
                    )
                raise
            self._receive_round_outcome()
            return

        try:
            self.client.start_weight_update()
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="m2n-weight-sync"
            )
            try:
                update = executor.submit(
                    self.client.update_weights,
                    {"names": [m.name for m in self._metas]},
                )
                # Surface an immediate worker-side rejection before trainers
                # enter the collective sequence.
                if update.done():
                    update.result()
                self._send()
                update.result()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            self.client.finish_weight_update()
        except BaseException as exc:
            try:
                self._publish_round_outcome(exc)
            except BaseException:
                logger.exception("failed to publish nccl_m2n round failure")
            raise

        self._publish_round_outcome(None)

    def _publish_round_outcome(self, failure: BaseException | None) -> None:
        """Tell non-sender trainer ranks whether worker finalization passed."""
        if self._num_trainer_ranks <= 1:
            return
        if self._metadata_group is None:
            raise RuntimeError("nccl_m2n metadata group is unavailable")
        envelope = {
            "ok": failure is None,
            "error": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        }
        for rank in range(1, self._num_trainer_ranks):
            self._metadata_group.send_obj(envelope, rank)
        # Keep rank zero's TCPStore alive until every trainer consumed its
        # outcome. This also keeps the per-rank message counters aligned.
        expected_ack = {"phase": "round_outcome_consumed"}
        for rank in range(1, self._num_trainer_ranks):
            ack = self._metadata_group.recv_obj(rank)
            if ack != expected_ack:
                raise RuntimeError(
                    f"trainer rank {rank} sent an invalid round-outcome acknowledgement"
                )

    def _receive_round_outcome(self) -> None:
        """Wait until trainer rank zero observed every worker's finish RPC."""
        if self._metadata_group is None:
            raise RuntimeError("nccl_m2n metadata group is unavailable")
        outcome = self._metadata_group.recv_obj(0)
        self._metadata_group.send_obj({"phase": "round_outcome_consumed"}, 0)
        if not isinstance(outcome, Mapping) or set(outcome) != {"ok", "error"}:
            raise RuntimeError("trainer rank zero sent an invalid round outcome")
        ok = outcome["ok"]
        error = outcome["error"]
        if not isinstance(ok, bool) or ok != (error is None):
            raise RuntimeError("trainer rank zero sent an invalid round outcome")
        if not ok:
            if not isinstance(error, str) or not error:
                raise RuntimeError("trainer rank zero sent an invalid round error")
            raise RuntimeError(
                "nccl_m2n worker update failed on trainer rank zero: " + error
            )

    def _send(self) -> None:
        """Reshard each local shard this rank owns into its agreed layout."""
        assert self.source is not None
        stream = torch.cuda.current_stream()
        owned = [i for i, m in enumerate(self._metas) if self.rank in m.owners]

        device_index = torch.accelerator.current_device_index()
        # Optimizer work may have used another CUDA stream. Make prior writes
        # visible before a provider materializes current-stream local shards.
        torch.accelerator.synchronize(device_index)
        sent = 0
        for name, shard in self.source.iter_local():
            if sent >= len(owned):
                raise RuntimeError(
                    "weight source yielded more parameters than this rank owns "
                    f"(first extra parameter: '{name}')"
                )
            index = owned[sent]
            meta = self._metas[index]
            if name != meta.name:
                raise RuntimeError(
                    f"weight source yielded '{name}' at position {sent}, but "
                    f"its metadata declared '{meta.name}'; the source must "
                    "iterate in a stable order"
                )
            local_shape = expected_local_shape(meta)
            if shard.dtype != meta.dtype or tuple(shard.shape) != local_shape:
                raise ValueError(
                    f"parameter '{name}' source returned {shard.dtype} "
                    f"{tuple(shard.shape)}, but its layout implies {meta.dtype} "
                    f"{local_shape}"
                )
            sent += 1
            if self._skip[index]:
                continue
            reshard(
                self._m2n,
                self._handle,
                self._comms[self._group_of[index]],
                stream,
                meta,
                self._layouts[index],
                self._num_workers,
                shard,
                None,
            )
            # Let the allocator recycle the shard's storage in stream order.
            shard.record_stream(stream)

        if sent != len(owned):
            missing = self._metas[owned[sent]].name
            raise RuntimeError(
                f"weight source yielded {sent} parameters, but this rank owns "
                f"{len(owned)} (first missing: '{missing}')"
            )

        # As in the other weight-transfer backends, wait before returning
        # rather than relying on same-stream ordering. The caller may then
        # safely mutate parameters or continue work on another stream.
        stream.synchronize()

    def shutdown(self) -> None:
        """Finish pending GPU work and release M2N communication state."""
        handle, self._handle = self._handle, None
        comms, self._comms = self._comms, []
        self._metadata_group = None
        self._layouts = []
        self._skip = []
        self._group_of = []
        self._m2n = None

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
            attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
        if handle is not None:
            attempt(handle.destroy, "failed to destroy nccl_m2n handle")
        for comm in comms:
            if comm is not None:
                attempt(comm.abort, "failed to abort nccl_m2n communicator")
        if first_error is not None:
            raise first_error
