# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core-owned process groups and collectives for specialized models."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from vllm.distributed.device_communicators.peer_transport import (
    MAX_PEERS,
    PeerBuffer,
    PeerTransport,
)

__all__ = ["MAX_PEERS", "PeerBuffer", "PeerTransport", "comm", "groups", "step"]


@dataclass(frozen=True)
class Group:
    """One process group: rank, size, and its device and CPU groups."""

    name: str
    rank: int
    size: int
    device_group: Any
    cpu_group: Any
    coordinator: Any  # vLLM's GroupCoordinator; for core-backed collectives


def _coordinator(name: str) -> Any:
    from vllm.distributed import parallel_state as ps

    getters = {
        "tp": ps.get_tp_group,
        "pp": ps.get_pp_group,
        "dp": ps.get_dp_group,
        "ep": ps.get_ep_group,
        "world": ps.get_world_group,
    }
    if name not in getters:
        raise KeyError(f"no process group named {name!r}")
    return getters[name]()


class _Groups:
    def get(self, name: str) -> Group:
        c = _coordinator(name)
        return Group(
            name=name,
            rank=c.rank_in_group,
            size=c.world_size,
            device_group=c.device_group,
            cpu_group=c.cpu_group,
            coordinator=c,
        )

    @property
    def tp(self) -> Group:
        return self.get("tp")

    @property
    def pp(self) -> Group:
        return self.get("pp")

    @property
    def dp(self) -> Group:
        return self.get("dp")

    @property
    def ep(self) -> Group:
        return self.get("ep")

    @property
    def world(self) -> Group:
        return self.get("world")


class _Comm:
    def all_reduce(self, x: torch.Tensor, group: str = "tp") -> torch.Tensor:
        return _coordinator(group).all_reduce(x)

    def all_gather(self, x: torch.Tensor, dim: int = -1, group: str = "tp"):
        return _coordinator(group).all_gather(x, dim)

    def reduce_scatter(self, x: torch.Tensor, dim: int = -1, group: str = "tp"):
        return _coordinator(group).reduce_scatter(x, dim)

    def cpu_all_reduce_min(self, value: int, group: str = "tp") -> int:
        """MIN over the group's CPU group, for rank agreement on host decisions."""
        t = torch.tensor([value], dtype=torch.int64)
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MIN, group=_coordinator(group).cpu_group
        )
        return int(t.item())

    def peer_arena(
        self, capacity: int, group: str = "tp", kind: str = "auto"
    ) -> PeerTransport:
        """Collective: an arena shared with the group's peers over CUDA IPC or
        symmetric memory; `alloc(name, nbytes)` carves buffers in rank order."""
        from vllm.distributed.device_communicators.peer_transport import (
            make_transport,
        )

        g = groups.get(group)
        return make_transport(
            kind,
            group=g.device_group,
            cpu_group=g.cpu_group,
            rank=g.rank,
            world=g.size,
            capacity=capacity,
        )

    def peer_buffer(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype, group: str = "tp"
    ) -> PeerBuffer:
        """Collective: one buffer shared with the group's peers over IPC."""
        nbytes = math.prod(shape) * dtype.itemsize
        arena = self.peer_arena(nbytes + 8192, group)
        _arenas.append(arena)  # the arena owns the memory
        return arena.alloc(name, nbytes)

    def symmetric(self, name: str) -> Any:
        raise NotImplementedError(
            "symmetric buffers from resources.memory are not in the prototype"
        )


_arenas: list[PeerTransport] = []


class _Step:
    """Read-only metadata of the step (or capture) being run; core sets it per
    step and per capture. Empty mappings and mode NONE outside a step."""

    @staticmethod
    def _context() -> Any:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        return get_forward_context() if is_forward_context_available() else None

    @property
    def attn_metadata(self) -> Mapping[str, Any]:
        """Attention metadata by layer name."""
        md = getattr(self._context(), "attn_metadata", None)
        return MappingProxyType(md) if isinstance(md, dict) else _EMPTY

    @property
    def slot_mapping(self) -> Mapping[str, torch.Tensor]:
        """KV-cache slot mapping by layer name."""
        sm = getattr(self._context(), "slot_mapping", None)
        return MappingProxyType(sm) if isinstance(sm, dict) else _EMPTY

    @property
    def is_capturing(self) -> bool:
        """Whether this forward is being captured into a CUDA graph."""
        return torch.cuda.is_current_stream_capturing()

    @property
    def graph_mode(self) -> Any:
        """The step's `CUDAGraphMode`; FULL captures run with NONE."""
        from vllm.config import CUDAGraphMode

        context = self._context()
        return CUDAGraphMode.NONE if context is None else context.cudagraph_runtime_mode


_EMPTY: Mapping[str, Any] = MappingProxyType({})

groups = _Groups()
comm = _Comm()
step = _Step()
