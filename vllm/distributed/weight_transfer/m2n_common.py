# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the NCCL M2N (`nccl_m2n`) weight transfer backend.

M2N reshards a tensor between two disjoint meshes of ranks that live in one
communicator: the trainer occupies ranks `[0, T)` and the inference workers
`[T, T + N)`. This module holds everything both sides need — the optional
runtime import, layout descriptors, and the conversion to `nccl.m2n` types —
so the engine module stays about the transfer itself.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed.weight_transfer.base import ParamMeta, Shard
from vllm.utils.nccl import unpinned_nccl_env

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup

# Placement code for a mesh axis that replicates. Any other (non-negative) code
# is the tensor dim that axis shards. Ints rather than `nccl.m2n` objects so
# layouts survive the JSON init handshake without a custom encoder.
REPLICATE = -1

# `NCCL_RESHARD_MAX_TENSOR_DIMS`; 4-D and higher are rejected by the library.
MAX_TENSOR_DIMS = 3

MESH_NDIMS = 2

# Conservative capacities accepted by the supported nccl_m2n algorithms. They
# are not exposed by its Python API yet; remove these mirrors once nccl_m2n
# provides communication-free plan validation.
MAX_SOURCE_SHARDS = 16
MAX_DEST_SHARDS = 64

# Increment when the JSON init payload changes incompatibly.
M2N_WIRE_SCHEMA_VERSION = 3

SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in (
            "int8",
            "uint8",
            "float8_e4m3fn",
            "float8_e5m2",
            "float16",
            "bfloat16",
            "int32",
            "uint32",
            "float32",
            "int64",
            "uint64",
            "float64",
        )
    )
    if dtype is not None
)

_IMPORT_HINT = (
    "The nccl_m2n weight transfer backend requires the `nccl-extensions` "
    "package (and its `nccl4py` dependency), which is not installed by vLLM. "
    "See https://github.com/NVIDIA/nccl-extensions for build "
    "and install instructions. It needs NCCL 2.30.5 or newer."
)


def prepare_m2n_local_runtime(m2n: Any, max_cta: int | None) -> tuple[Any, int, Any]:
    """Prepare rank-local resources before any rank enters NCCL init."""
    import nccl.core as nccl

    device = torch.accelerator.current_device_index()
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise RuntimeError(f"invalid current accelerator device index: {device!r}")
    handle = m2n.Handle.create(m2n.Config(max_cta=max_cta))
    return nccl, device, handle


def create_m2n_communicators(
    nccl: Any,
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
    source_groups: Sequence[tuple[int, ...]],
    num_trainer_ranks: int,
) -> list[Any]:
    """Create one NCCL4Py communicator per source group: the group's trainer
    ranks, in order, then every inference worker. Returns this rank's
    communicator per group, None where it is not a member. Every rank joins its
    groups in the same order, so the blocking inits cannot cross."""
    if unique_id_bytes is not None:
        if len(source_groups) != 1:
            raise ValueError(
                "a pre-shared NCCL unique id serves one communicator, but the "
                f"source declares {len(source_groups)} owner groups"
            )
        unique_ids = [unique_id_bytes]
    else:
        envelope: dict[str, Any] | None = None
        if group.rank == 0:
            try:
                envelope = {
                    "unique_ids": [
                        bytes(nccl.get_unique_id().as_bytes) for _ in source_groups
                    ],
                    "error": None,
                }
            except Exception as exc:
                envelope = {"unique_ids": None, "error": f"{type(exc).__name__}: {exc}"}
        envelope = group.broadcast_obj(envelope, src=0)
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "unique_ids",
            "error",
        }:
            raise RuntimeError("invalid NCCL4Py unique-id envelope")
        if envelope["error"] is not None:
            raise RuntimeError(
                f"NCCL4Py unique-id generation failed: {envelope['error']}"
            )
        unique_ids = envelope["unique_ids"]
        if not isinstance(unique_ids, list) or len(unique_ids) != len(source_groups):
            raise RuntimeError("NCCL4Py unique-id broadcast returned a bad list")

    num_workers = group.world_size - num_trainer_ranks
    comms: list[Any] = []
    for owners, uid in zip(source_groups, unique_ids):
        if group.rank >= num_trainer_ranks:
            rank = len(owners) + group.rank - num_trainer_ranks
        elif group.rank in owners:
            rank = owners.index(group.rank)
        else:
            comms.append(None)
            continue
        with unpinned_nccl_env():
            comms.append(
                nccl.Communicator.init(
                    nranks=len(owners) + num_workers,
                    rank=rank,
                    unique_id=nccl.UniqueId.from_bytes(uid),
                )
            )
    return comms


def import_m2n() -> Any:
    """Import `nccl.m2n` lazily, with an actionable error when it is missing.

    Deferred so that importing vLLM — or any other weight transfer backend —
    never requires the m2n runtime to be present.
    """
    try:
        import nccl.m2n as m2n
    except ImportError as e:
        raise ImportError(f"{_IMPORT_HINT} (import failed: {e})") from e
    return m2n


@dataclass(frozen=True)
class M2NMesh:
    """One side's rank topology — pure topology, no tensor placement.

    Mirrors `ncclMesh_t`: a 2-axis mesh owning the contiguous rank interval
    `[start_rank, start_rank + dims[0] * dims[1])`. There is no 1-D mesh; a
    single-axis topology is spelled with a second axis of size 1.

    A model may use a different factorization for each tensor. For example,
    dense weights can use a DP x TP mesh while expert weights use EDP x EP.
    """

    dims: tuple[int, int]
    start_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        if len(self.dims) != MESH_NDIMS or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.dims
        ):
            raise ValueError(f"mesh dims must be {MESH_NDIMS} positive ints")
        if (
            not isinstance(self.start_rank, int)
            or isinstance(self.start_rank, bool)
            or self.start_rank < 0
        ):
            raise ValueError(
                f"mesh start_rank must be non-negative, got {self.start_rank}"
            )

    @property
    def size(self) -> int:
        return self.dims[0] * self.dims[1]


# A tensor's placement over its side's mesh: one code per mesh axis
# (`REPLICATE`, or the tensor dim that axis shards), or `REPLICATED` for a
# tensor every rank holds in full.
Placements = tuple[int, int]
REPLICATED: Placements | None = None


@dataclass(frozen=True)
class M2NLayout:
    """A tensor's mesh and placement on one side of a transfer."""

    mesh: M2NMesh
    placements: Placements | None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, M2NMesh):
            raise TypeError("M2N layout mesh must be an M2NMesh")
        if self.placements is not None:
            object.__setattr__(self, "placements", tuple(self.placements))


def check_placements(placements: Placements, context: str = "placements") -> None:
    """Reject placement pairs nccl_m2n cannot express."""
    if len(placements) != MESH_NDIMS:
        raise ValueError(f"{context} must have {MESH_NDIMS} entries")
    if any(not isinstance(code, int) or isinstance(code, bool) for code in placements):
        raise TypeError(f"{context} must contain integer placement codes")
    invalid = [code for code in placements if code < REPLICATE]
    if invalid:
        raise ValueError(
            f"{context} contains invalid placement code {invalid[0]}; valid "
            f"codes are {REPLICATE} (Replicate) and non-negative tensor dimensions"
        )
    num_sharded = sum(code != REPLICATE for code in placements)
    if num_sharded == 0:
        raise ValueError(
            f"{context}: a fully replicated tensor is carried as REPLICATED, "
            "not as two REPLICATE axes"
        )
    if num_sharded == MESH_NDIMS:
        raise ValueError(
            f"{context}: nccl_m2n needs one REPLICATE mesh axis, but "
            f"{placements} shards both"
        )


def resolve_layout(
    mesh: M2NMesh,
    placements: Placements | None,
    context: str = "placements",
) -> tuple[M2NMesh, Placements]:
    """Pair one tensor's placement with the mesh m2n should see for it.

    A replicated tensor needs a size-1 mesh axis to carry a no-op shard, since
    m2n has no `{REPLICATE, REPLICATE}`. It is therefore described over the
    *same rank interval* re-factored as `(size, 1)`. That re-factoring is sound
    precisely because replication is order-independent — every rank holds the
    whole tensor, so it does not matter that `(a, b)` and `(size, 1)` walk the
    interval in a different order. A sharded tensor keeps its side's own
    factorization, where rank order decides who owns which shard.
    """
    if placements is None:
        return M2NMesh((mesh.size, 1), mesh.start_rank), (REPLICATE, 0)
    check_placements(placements, context)
    return mesh, placements


def validate_layout(
    mesh: M2NMesh,
    placements: Placements,
    shape: Sequence[int],
    context: str,
) -> tuple[int, ...]:
    """Validate one resolved layout and return its required local shape."""
    local_shape = list(shape)
    for axis, tensor_dim in enumerate(placements):
        if tensor_dim == REPLICATE:
            continue
        if tensor_dim >= len(shape):
            raise ValueError(
                f"{context} shards tensor dim {tensor_dim}, but parameter has "
                f"rank {len(shape)}"
            )
        factor = mesh.dims[axis]
        if shape[tensor_dim] % factor:
            raise ValueError(
                f"{context} shards dim {tensor_dim} (size {shape[tensor_dim]}) "
                f"over {factor} ranks, which does not divide evenly"
            )
        local_shape[tensor_dim] //= factor
    return tuple(local_shape)


def expected_local_shape(meta: "M2NParamMeta") -> tuple[int, ...]:
    """Validate a parameter's source layout and return its local shape."""
    mesh, placements = resolve_layout(
        meta.source_layout.mesh,
        meta.source_layout.placements,
        f"parameter '{meta.name}' source placements",
    )
    return validate_layout(
        mesh, placements, meta.shape, f"parameter '{meta.name}' source layout"
    )


def check_transferable(name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
    """Reject parameters nccl_m2n cannot transfer."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"parameter '{name}' has dtype {dtype}, which nccl_m2n does not "
            f"support. Supported: {sorted(str(d) for d in SUPPORTED_DTYPES)}"
        )
    if not 1 <= len(shape) <= MAX_TENSOR_DIMS:
        raise ValueError(
            f"parameter '{name}' has rank {len(shape)}; nccl_m2n supports "
            f"rank 1..{MAX_TENSOR_DIMS}"
        )


def _shard_spec(mesh: M2NMesh, placements: Placements) -> tuple[int, int]:
    return next(
        (code, mesh.dims[axis])
        for axis, code in enumerate(placements)
        if code != REPLICATE
    )


def _max_partition_overlaps(
    extent: int,
    partitions: int,
    other_partitions: int,
) -> int:
    """Return the maximum other partitions intersected by any one partition."""
    if extent == 0:
        return 0
    width = extent // partitions
    other_width = extent // other_partitions
    max_overlaps = 0
    for index in range(partitions):
        start = index * width
        end = start + width
        first_other = start // other_width
        last_other = (end - 1) // other_width
        max_overlaps = max(max_overlaps, last_other - first_other + 1)
    return max_overlaps


def check_plan_limits(
    src: tuple[M2NMesh, Placements],
    dst: tuple[M2NMesh, Placements],
    name: str,
    shape: Sequence[int],
) -> None:
    """Reject plans whose per-shard fan-in or fan-out exceeds M2N capacity."""
    src_dim, src_shards = _shard_spec(*src)
    dst_dim, dst_shards = _shard_spec(*dst)
    if any(extent == 0 for extent in shape):
        max_sources = max_targets = 0
    elif src_dim != dst_dim:
        max_sources = src_shards
        max_targets = dst_shards
    else:
        extent = shape[src_dim]
        max_sources = _max_partition_overlaps(extent, dst_shards, src_shards)
        max_targets = _max_partition_overlaps(extent, src_shards, dst_shards)

    if max_sources > MAX_SOURCE_SHARDS:
        raise ValueError(
            f"parameter '{name}' would feed up to {max_sources} source shards "
            f"into one destination shard, over nccl_m2n MAX_SOURCES="
            f"{MAX_SOURCE_SHARDS}"
        )
    if max_targets > MAX_DEST_SHARDS:
        raise ValueError(
            f"parameter '{name}' would fan one source shard out to up to "
            f"{max_targets} destination shards, over nccl_m2n "
            f"MAX_TARGETS={MAX_DEST_SHARDS}"
        )


@dataclass(frozen=True)
class M2NParamMeta(ParamMeta):
    """`ParamMeta` with its m2n source layout spelled out, as the engines plan
    reshards from it. Built from a source's `ParamMeta` by `m2n_param_meta`.
    `owners` is always set: the trainer ranks of this parameter's source group,
    whose communicator carries it.
    """

    source_layout: M2NLayout
    allow_full_fallback: bool = True
    owners: tuple[int, ...] = field(default=(), kw_only=True)
    """Whether a worker may receive the full tensor for `load_weights`."""

    def __post_init__(self) -> None:
        if not isinstance(self.source_layout, M2NLayout):
            raise TypeError("source_layout must be an M2NLayout")
        if not self.owners or self.source_layout.mesh.size != len(self.owners):
            raise ValueError(
                f"parameter '{self.name}' source mesh covers "
                f"{self.source_layout.mesh.size} ranks, but it has owners "
                f"{self.owners}"
            )


def m2n_param_meta(
    meta: ParamMeta, num_trainer_ranks: int, allow_full_fallback: bool = True
) -> M2NParamMeta:
    """The m2n source layout of `meta` over its owners (every trainer rank by
    default), which form its source group.

    A 1-D mesh gains a size-1 second axis; a replicated parameter keeps
    `REPLICATED`. Placements m2n cannot express (both axes sharded, more than
    two axes) are rejected by `expected_local_shape`.
    """
    owners = meta.owners or tuple(range(num_trainer_ranks))
    if max(owners) >= num_trainer_ranks:
        raise ValueError(
            f"parameter '{meta.name}' is owned by trainer ranks {owners}, but "
            f"there are {num_trainer_ranks} trainer ranks"
        )
    mesh_shape = meta.mesh_shape or (len(owners),)
    if len(mesh_shape) > MESH_NDIMS:
        raise ValueError(
            f"parameter '{meta.name}' has a {len(mesh_shape)}-D source mesh; "
            f"nccl_m2n supports at most {MESH_NDIMS} axes"
        )
    dims = (mesh_shape[0], mesh_shape[1] if len(mesh_shape) == 2 else 1)
    placements: Placements | None = REPLICATED
    if meta.placements is not None:
        codes = [p.dim if isinstance(p, Shard) else REPLICATE for p in meta.placements]
        codes += [REPLICATE] * (MESH_NDIMS - len(codes))
        placements = cast(Placements, tuple(codes))
    return M2NParamMeta(
        meta.name,
        meta.dtype,
        meta.shape,
        M2NLayout(M2NMesh(dims, 0), placements),
        allow_full_fallback,
        owners=tuple(owners),
    )


@dataclass(frozen=True)
class M2NWireParam:
    """JSON-safe source metadata for one parameter in the init handshake."""

    name: str
    dtype_name: str
    shape: tuple[int, ...]
    src_mesh_dims: tuple[int, int]
    src_placements: Placements | None
    owners: tuple[int, ...]
    allow_full_fallback: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "owners", tuple(self.owners))
        shape = tuple(self.shape)
        dims = tuple(self.src_mesh_dims)
        placements = None if self.src_placements is None else tuple(self.src_placements)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "src_mesh_dims", dims)
        object.__setattr__(self, "src_placements", placements)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("M2N wire parameter name must not be empty")
        if not isinstance(self.dtype_name, str) or not self.dtype_name:
            raise ValueError(f"parameter '{self.name}' has an empty wire dtype name")
        if not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in shape
        ):
            raise ValueError(
                f"parameter '{self.name}' shape must contain positive integers"
            )
        M2NMesh(cast(tuple[int, int], dims), 0)
        if placements is not None:
            check_placements(
                cast(Placements, placements),
                f"parameter '{self.name}' source placements",
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M2NWireParam":
        """Parse one strict wire record from its JSON-decoded mapping."""
        fields = {
            "name",
            "dtype_name",
            "shape",
            "src_mesh_dims",
            "src_placements",
            "owners",
            "allow_full_fallback",
        }
        missing = fields - value.keys()
        extra = value.keys() - fields
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid M2N wire parameter: {', '.join(details)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record using only JSON-compatible values."""
        return {
            "name": self.name,
            "dtype_name": self.dtype_name,
            "shape": list(self.shape),
            "src_mesh_dims": list(self.src_mesh_dims),
            "src_placements": (
                None if self.src_placements is None else list(self.src_placements)
            ),
            "owners": list(self.owners),
            "allow_full_fallback": self.allow_full_fallback,
        }


def source_plan_digest(params: Sequence[M2NWireParam]) -> str:
    """Return a rank-independent digest of the ordered source wire plan."""
    payload = json.dumps(
        {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "params": [param.to_dict() for param in params],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def check_source_plan_agreement(
    group: "StatelessProcessGroup",
    params: Sequence[M2NWireParam],
    expected_digest: str | None = None,
    local_error: str | None = None,
) -> str:
    """Verify every participant entered with the same source plan."""
    prepared_error = local_error
    digest: str | None = None
    declared: str | None = expected_digest
    if prepared_error is None:
        try:
            digest = source_plan_digest(params)
            declared = digest if expected_digest is None else expected_digest
            if not isinstance(declared, str) or not declared:
                raise ValueError("declared source digest must be a non-empty string")
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local source error must be a non-empty string"

    gathered = group.all_gather_obj(
        {
            "phase": "source",
            "digest": digest,
            "declared_digest": declared,
            "error": prepared_error,
        }
    )
    errors: list[str] = []
    expected_fields = {"phase", "digest", "declared_digest", "error"}
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    agreed: list[tuple[int, str, str]] = []
    for rank, item in enumerate(gathered):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            errors.append(f"rank {rank}: invalid source preflight envelope")
            continue
        if item["phase"] != "source":
            errors.append(f"rank {rank}: invalid source preflight phase")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid source preflight error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        computed = item["digest"]
        claimed = item["declared_digest"]
        if (
            not isinstance(computed, str)
            or not computed
            or not isinstance(claimed, str)
            or not claimed
        ):
            errors.append(f"rank {rank}: invalid source digest")
            continue
        agreed.append((rank, computed, claimed))
    if errors:
        raise RuntimeError(
            "nccl_m2n source preflight rejected before NCCL init: " + "; ".join(errors)
        )

    canonical = agreed[0][1]
    for rank, computed, claimed in agreed:
        if computed != claimed:
            errors.append(
                f"rank {rank}: computed digest {computed} does not match "
                f"declared digest {claimed}"
            )
        if computed != canonical:
            errors.append(f"rank {rank}: source digest disagrees with rank 0")
    if errors:
        raise RuntimeError(
            "nccl_m2n source preflight rejected before NCCL init: " + "; ".join(errors)
        )
    return canonical


def check_runtime_ready_agreement(
    group: "StatelessProcessGroup", local_error: str | None
) -> None:
    """Require every rank to prepare local resources before NCCL init."""
    prepared_error = local_error
    if prepared_error is not None and (
        not isinstance(prepared_error, str) or not prepared_error
    ):
        prepared_error = "TypeError: local runtime error must be a non-empty string"
    gathered = group.all_gather_obj({"phase": "runtime_ready", "error": prepared_error})
    errors: list[str] = []
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"phase", "error"}
            or item.get("phase") != "runtime_ready"
        ):
            errors.append(f"rank {rank}: invalid runtime readiness envelope")
            continue
        error = item.get("error")
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid runtime readiness error")
            else:
                errors.append(f"rank {rank}: {error}")
    if errors:
        raise RuntimeError(
            "nccl_m2n local runtime preflight failed before NCCL init: "
            + "; ".join(errors)
        )


def check_data_plane_agreement(
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
    local_error: str | None = None,
) -> None:
    """Agree on the NCCL rendezvous identity before NCCL init."""
    prepared_error = local_error
    uid_digest: str | None = None
    if prepared_error is None:
        try:
            if unique_id_bytes is not None:
                if not isinstance(unique_id_bytes, bytes):
                    raise TypeError("NCCL unique id must decode to bytes")
                uid_digest = hashlib.sha256(unique_id_bytes).hexdigest()
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local data-plane error must be non-empty"

    gathered = group.all_gather_obj(
        {
            "phase": "data_plane",
            "mode": "uid" if unique_id_bytes is not None else "tcp",
            "uid_digest": uid_digest,
            "error": prepared_error,
        }
    )
    expected_fields = {
        "phase",
        "mode",
        "uid_digest",
        "error",
    }
    errors: list[str] = []
    valid: list[tuple[int, str, str | None]] = []
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_fields
            or item.get("phase") != "data_plane"
        ):
            errors.append(f"rank {rank}: invalid data-plane preflight envelope")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid data-plane preflight error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        mode = item["mode"]
        uid = item["uid_digest"]
        if mode not in {"tcp", "uid"} or (mode == "uid") != isinstance(uid, str):
            errors.append(f"rank {rank}: invalid data-plane identity")
            continue
        valid.append((rank, mode, uid))
    if not errors:
        _, mode, uid = valid[0]
        for rank, other_mode, other_uid in valid[1:]:
            if (other_mode, other_uid) != (mode, uid):
                errors.append(f"rank {rank}: NCCL data-plane identity disagrees")
    if errors:
        raise RuntimeError(
            "nccl_m2n data-plane preflight rejected before NCCL init: "
            + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Destination layouts (load plan, RFC #64418 §9)
# ---------------------------------------------------------------------------

# A destination layout: mesh dims over the N workers and their placements, or
# REPLICATED. Destination meshes start right after the source group's ranks.
DstLayout = tuple[tuple[int, int], Placements] | None
Box = tuple[tuple[int, int], ...]
# Agreed layout index for a name no worker needs: nothing is sent.
SKIP = -1


def candidate_layouts(meta: M2NParamMeta, num_workers: int) -> list[DstLayout]:
    """Every destination layout `meta` could take, the same list on every rank:
    each 2-D factorization of the workers with one axis `Shard(d)` (evenly) and
    the other `Replicate`, most shards first, then `REPLICATED`. Layouts over
    M2N's fan-in/out limits for `meta`'s source layout are left out."""
    found: list[tuple[int, DstLayout]] = []
    for a in range(1, num_workers + 1):
        if num_workers % a:
            continue
        dims = (a, num_workers // a)
        for dim, size in enumerate(meta.shape):
            for axis in range(MESH_NDIMS):
                if dims[axis] > 1 and size % dims[axis] == 0:
                    placements = [REPLICATE] * MESH_NDIMS
                    placements[axis] = dim
                    layout: DstLayout = (dims, cast(Placements, tuple(placements)))
                    found.append((dims[axis], layout))
    found.sort(key=lambda item: -item[0])  # stable: ties keep enumeration order
    src = resolve_layout(meta.source_layout.mesh, meta.source_layout.placements)
    layouts: list[DstLayout] = []
    candidates: list[DstLayout] = [item[1] for item in found] + [None]  # REPLICATED
    for candidate in candidates:
        dst = resolve_layout(*_mesh_and_placements(candidate, num_workers))
        try:
            check_plan_limits(src, dst, meta.name, meta.shape)
        except ValueError:
            continue
        layouts.append(candidate)
    return layouts


def _mesh_and_placements(
    layout: DstLayout, num_workers: int, start_rank: int = 0
) -> tuple[M2NMesh, Placements | None]:
    if layout is None:
        return M2NMesh((num_workers, 1), start_rank), REPLICATED
    return M2NMesh(layout[0], start_rank), layout[1]


def layout_box(layout: DstLayout, shape: Sequence[int], worker: int) -> Box:
    """The part of a `shape` tensor that `worker` receives under `layout`."""
    box = [(0, size) for size in shape]
    if layout is not None:
        dims, placements = layout
        for code, index, count in zip(placements, divmod(worker, dims[1]), dims):
            if code != REPLICATE:
                step = shape[code] // count
                box[code] = (index * step, (index + 1) * step)
    return tuple(box)


def agree_destination_layouts(
    group: "StatelessProcessGroup",
    metas: Sequence[M2NParamMeta],
    num_trainer_ranks: int,
    masks: Sequence[int] | None,
    local_error: str | None = None,
) -> list[int]:
    """AND every worker's per-name candidate masks and pick, per name, the first
    candidate all workers accept: an index into `candidate_layouts`, or `SKIP`.

    A worker's mask for a name has bit `i` set if it accepts candidate `i`, and
    bit `len(candidates)` if it needs nothing of the name. Trainer ranks pass
    `masks=None`. A local error on any rank fails every rank."""
    gathered = group.all_gather_obj({"masks": masks, "error": local_error})
    errors = [
        f"rank {rank}: {item['error']}"
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if errors:
        raise RuntimeError(
            "nccl_m2n destination agreement failed: " + "; ".join(errors)
        )
    workers = [item["masks"] for item in gathered[num_trainer_ranks:]]
    num_workers = len(workers)
    chosen = []
    for i, meta in enumerate(metas):
        common = -1
        for worker_masks in workers:
            common &= worker_masks[i]
        num_candidates = len(candidate_layouts(meta, num_workers))
        if common >> num_candidates & 1:
            chosen.append(SKIP)
        elif common & ((1 << num_candidates) - 1):
            chosen.append((common & -common).bit_length() - 1)
        else:
            raise RuntimeError(
                f"nccl_m2n: no destination layout of '{meta.name}' fits every "
                "worker within the M2N limits"
            )
    return chosen


def to_mesh(m2n: Any, mesh: M2NMesh) -> Any:
    return m2n.Mesh(mesh.dims, start_rank=mesh.start_rank)


def to_placements(m2n: Any, placements: Placements) -> list[Any]:
    return [
        m2n.Replicate() if code == REPLICATE else m2n.Shard(code) for code in placements
    ]


def reshard(
    m2n: Any,
    handle: Any,
    comm: Any,
    stream: torch.cuda.Stream,
    meta: M2NParamMeta,
    layout: DstLayout,
    num_workers: int,
    src: torch.Tensor | None,
    dst: torch.Tensor | None,
) -> None:
    """One `m2n.reshard` of `meta` over its source group's communicator, from
    its source layout into `layout` over the workers."""
    src_mesh, src_placements = resolve_layout(
        meta.source_layout.mesh, meta.source_layout.placements
    )
    dst_mesh, dst_placements = resolve_layout(
        *_mesh_and_placements(layout, num_workers, len(meta.owners))
    )
    m2n.reshard(
        src,
        dst,
        comm,
        stream,
        src_mesh=to_mesh(m2n, src_mesh),
        src_placements=to_placements(m2n, src_placements),
        dst_mesh=to_mesh(m2n, dst_mesh),
        dst_placements=to_placements(m2n, dst_placements),
        handle=handle,
    )


def source_groups(metas: Sequence[M2NParamMeta]) -> list[tuple[int, ...]]:
    """The distinct owner groups of `metas`, in order of first appearance; one
    communicator each."""
    return list(dict.fromkeys(meta.owners for meta in metas))
