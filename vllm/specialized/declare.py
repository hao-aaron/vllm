# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declaration types for specialized model directories.

Import-light by contract: no model, kernel, distributed or CUDA code, so routing
can read every declaration on CPU before anything is built.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

API_VERSION = 1

# Engine features in use that a directory must list in `Support.features`.
FEATURES = (
    "lora",
    "eplb",
    "sleep_mode",
    "weight_reload",
    "batch_invariance",
    "multimodal",
    "prompt_embeds",
    "kv_transfer",
)

# The flag that sets each field, named in decline messages.
FLAGS = {
    "kv_cache_dtype": "--kv-cache-dtype",
    "attention_backend": "--attention-backend",
    "moe_backend": "--moe-backend",
    "block_size": "--block-size",
    "dtype": "--dtype",
    "max_model_len": "--max-model-len",
    "tp": "--tensor-parallel-size",
    "pp": "--pipeline-parallel-size",
    "dp": "--data-parallel-size",
    "dcp": "--decode-context-parallel-size",
    "graph_mode": "-cc.cudagraph_mode",
    "speculation": "--speculative-config",
    "num_speculative_tokens": "--speculative-config",
}


@dataclass(frozen=True)
class RouteConfig:
    """The deployment facts routing decides on; identical on every rank."""

    platform: str
    arch: str
    compute_units: int | None
    machine: frozenset[str]
    libraries: Mapping[str, str]
    dtype: str
    max_model_len: int
    checkpoint: str
    hf_config: Any
    expert_quantization: str | None
    kv_cache_dtype: str
    attention_backend: str | None
    moe_backend: str | None
    tp: int
    pp: int
    dp: int
    ep: int
    dcp: int
    pcp: int
    micro_batching: bool
    speculation: str | None
    num_speculative_tokens: int
    graph_mode: str | None
    compile: bool
    role: str
    features: frozenset[str]
    block_size: int | None
    # Read-only view of the whole engine config, for `check`.
    settings: Any = field(repr=False, compare=False)


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("+")[0].split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _describe(want: Any) -> str:
    if isinstance(want, range):
        return f"{want.start}..{want.stop - 1}"
    shown = sorted(want, key=str)
    return repr(shown[0]) if len(shown) == 1 else f"one of {shown}"


def _hint(name: str, need: object) -> str:
    flag = FLAGS.get(name)
    return f" (set {flag} {need})" if flag else ""


@dataclass(frozen=True)
class Support:
    """The deployments a directory runs. Omitted field = any value."""

    arch: frozenset[str] | set[str] | None = None
    compute_units: range | None = None
    machine: frozenset[str] | set[str] | None = None
    libraries: Mapping[str, str] | None = None
    checkpoint: frozenset[str] | set[str] | None = None
    expert_quantization: frozenset[str] | set[str] | None = None
    model_shape: Mapping[str, Any] | None = None
    kv_cache_dtype: frozenset[str] | set[str] | None = None
    attention_backend: frozenset[str | None] | set[str | None] | None = None
    moe_backend: frozenset[str] | set[str] | None = None
    block_size: frozenset[int] | set[int] | None = None
    dtype: frozenset[str] | set[str] | None = None
    max_model_len: range | None = None
    tp: frozenset[int] | set[int] | None = None
    pp: frozenset[int] | set[int] | None = None
    dp: frozenset[int] | set[int] | None = None
    ep: frozenset[int] | set[int] | None = None
    dcp: frozenset[int] | set[int] | None = None
    pcp: frozenset[int] | set[int] | None = None
    micro_batching: bool | None = None
    graph_mode: frozenset[str] | set[str] | None = None
    compile: bool | None = None
    speculation: frozenset[str | None] | set[str | None] | None = None
    num_speculative_tokens: frozenset[int] | set[int] | None = None
    role: frozenset[str] | set[str] | None = None
    # Engine features handled; any feature in use and not listed declines.
    features: frozenset[str] | set[str] = frozenset()
    default_on: bool = True
    check: Callable[[RouteConfig], list[str]] | None = None

    def reasons(self, cfg: RouteConfig) -> list[str]:
        """Every way `cfg` falls outside this declaration, in words."""
        out: list[str] = []
        for f in fields(self):
            want = getattr(self, f.name)
            if want is None or f.name in ("features", "default_on", "check"):
                continue
            if f.name == "machine":
                if missing := set(want) - cfg.machine:
                    out.append(f"machine lacks {sorted(missing)}")
            elif f.name == "libraries":
                for lib, minimum in want.items():
                    have = cfg.libraries.get(lib)
                    if have is None or _version_tuple(have) < _version_tuple(minimum):
                        out.append(f"{lib} is {have}, needs >= {minimum}")
            elif f.name == "model_shape":
                for key, value in want.items():
                    have = getattr(cfg.hf_config, key, None)
                    if have != value:
                        out.append(f"{key} is {have!r}, needs {value!r}")
            else:
                have = getattr(cfg, f.name)
                if isinstance(want, bool):
                    # A bool field says whether the setting is allowed.
                    if have and not want:
                        out.append(f"{f.name} is on, not supported")
                elif have not in want:
                    need = _describe(want)
                    out.append(
                        f"{f.name} is {have!r}, needs {need}" + _hint(f.name, need)
                    )
        if extra := cfg.features - set(self.features):
            out.append(f"engine features {sorted(extra)} not supported")
        if self.check is not None:
            out.extend(self.check(cfg))
        return out


@dataclass(frozen=True)
class GroupRequest:
    name: str
    over: str  # "tp", "dp", "ep" or "world"


@dataclass(frozen=True)
class MemoryRequest:
    name: str
    bytes_per_rank: int
    kind: str = "weights"  # "weights", "workspace" or "symmetric"


@dataclass(frozen=True)
class KVGroup:
    layers: tuple[str, ...]
    block_size: int
    bytes_per_token: int
    payload: Any = None


@dataclass(frozen=True)
class Resources:
    groups: tuple[GroupRequest, ...] = ()
    memory: tuple[MemoryRequest, ...] = ()
    kv: tuple[KVGroup, ...] = ()
    split_points: tuple[str, ...] = ()
