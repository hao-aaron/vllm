# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core-owned process groups and collectives for specialized models."""

from dataclasses import dataclass
from typing import Any

import torch


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

    def symmetric(self, name: str) -> Any:
        raise NotImplementedError(
            "symmetric buffers from resources.memory are not in the prototype"
        )


groups = _Groups()
comm = _Comm()
