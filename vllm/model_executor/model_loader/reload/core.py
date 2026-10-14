# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reload core: direct loading with a modulewise fallback, and explicit
arrival units.

Every module that owns loadable tensors takes one path per update:

- DIRECT: loaders write live memory, either the live tensor itself ("as-is":
  same object, shape, stride and dtype as the snapshot taken at model init) or
  a checkpoint-layout view of its bytes ("view"). Tensors with no room in
  live memory but a create-time value (scales with a smaller live form, KV
  scales the attention PWAL deleted) are staged in side tensors.
- MODULEWISE: scratch for a whole module (checkpoint format, materialized on
  first touch); CPU incoming is buffered in host memory, credited by
  checkpoint name and replayed once when the module's units arrived. Live
  bytes are untouched until the module lands, with a plain copy.

Attention layers complete in the attention phase at finish. The cold load of
online-quantized modules also runs here (`cold_online_load`): MODULEWISE in
cold mode, where the quantized results simply stay registered.

The path is chosen from metadata at start; `VLLM_RELOAD_DIRECT=0` forces
MODULEWISE. A module completes when its expected units arrived (see
`units.py`): its quant method's process_weights_after_loading (PWAL) runs, the
results land in the live tensors, and the original tensor objects are put
back. Finish checks coverage, completes what is left, then runs the refresh
phases (kernel refresh, attention, layer and model refreshes).

Layerwise reload (`layerwise.py`) stays as the legacy path: selected with
`VLLM_RELOAD_LEGACY=1`, and automatically for models that run post-load work
inside `load_weights` or whose loads could not be observed.
"""

import inspect
import weakref
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from . import layerwise as lw
from .meta import (
    SKIP_LOAD_TENSORS,
    SKIP_MODULES,
    SKIP_TENSORS,
    fill_init_value,
    get_numel_loaded,
    materialize_meta_tensor,
    to_meta_tensor,
)
from .observer import OBSERVE_WRITES, WriteObserver, storage_key, suspended
from .refresh import refresh_quant_method, run_model_refresh
from .sanitize import restore_layer_refs
from .units import (
    _CURRENT_LOAD,
    FRAMEWORK_WRITES,
    WILDCARD,
    ExpectedUnits,
    LoadCall,
    Unit,
    current_checkpoint_name,
    infer_shard,
)
from .utils import get_tensor_load_numel

logger = init_logger(__name__)

__all__ = [
    "ReloadUnitError",
    "abort_core",
    "complete_module",
    "ensure_materialized",
    "record_units",
    "core_active",
    "finish_core",
    "legacy_reason",
    "observe_cold_load",
    "start_core",
    "trace_loads",
]

LEGACY = envs.VLLM_RELOAD_LEGACY
DIRECT_ENABLED = envs.VLLM_RELOAD_DIRECT
# Development (R1): also count copied elements as layerwise reload does, and
# record where count-based completion would disagree with the units
SHADOW_COUNT = envs.VLLM_RELOAD_SHADOW_COUNT


class ReloadUnitError(RuntimeError):
    """A weight update delivered units its sender's contract rules out
    (a duplicate, or data after its module completed)."""


# ---------------------------------------------------------------------------
# Per-update state
# ---------------------------------------------------------------------------

AS_IS, VIEW, SIDE, SCRATCH = "as-is", "view", "side", "scratch"


@dataclass
class Target:
    name: str
    kind: str
    is_buffer: bool
    live: torch.Tensor | None  # the original live object (None: not live)
    standin: torch.Tensor  # registered during the update
    fill: float = 0.0
    filled: bool = False  # started from `fill` in this update
    # as-is targets: the live tensor's own weight_loader attribute
    had_loader: bool = False
    loader_attr: Callable | None = None
    # as-is targets: loader attributes (e.g. MoE scales' `quant_method`) the
    # recorded tensor has and the live one lost (`replace_parameter` keeps only
    # `weight_loader`), set on the live tensor for this update
    added_attrs: list[str] = field(default_factory=list)


@dataclass
class ModuleState:
    name: str
    module: torch.nn.Module
    path: str  # "direct" or "modulewise"
    post: str  # "pwal", "pwal_at_finish", "none", or "attention" (finish phase)
    targets: dict[str, Target] = field(default_factory=dict)
    # every live param / buffer at start (original objects), to put back
    live_params: dict[str, torch.Tensor] = field(default_factory=dict)
    live_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    non_persistent: set[str] = field(default_factory=set)
    # plain attributes under a gone param's name (e.g. Triton MXFP4's wrapped
    # weights), set aside while its stand-in is registered
    live_attrs: dict[str, object] = field(default_factory=dict)
    # aliases of the live tensors' storage, held across the PWAL: a PWAL that
    # rebinds a live tensor's `.data` (FlashInfer TRT-LLM's per-tensor FP8
    # shuffle) would otherwise move it
    held: dict[str, torch.Tensor] = field(default_factory=dict)
    remaining: Counter = field(default_factory=Counter)
    # the sum of `remaining`, and its keys by (tensor, shard): each loader
    # call does constant work, not a scan of the module's units
    pending: int = 0
    by_unit: dict[tuple[str, Hashable], tuple[tuple, ...]] = field(default_factory=dict)
    expected: Counter = field(default_factory=Counter)
    optional: set[tuple[str, Hashable]] = field(default_factory=set)
    arrived: Counter = field(default_factory=Counter)
    touched: bool = False
    materialized: bool = False
    completed: bool = False
    # MODULEWISE with CPU incoming: calls buffered until the module completes
    buffering: bool = False
    buffered: list[tuple[str, inspect.BoundArguments, Callable]] = field(
        default_factory=list
    )
    # units arrived that the expected set doesn't list: completes at finish
    format_mismatch: bool = False
    # R1 shadow counting (VLLM_RELOAD_SHADOW_COUNT=1)
    counted: int = 0
    count_total: int = 0
    counter_completed: bool = False

    def required_missing(self) -> list[tuple]:
        return sorted(
            (
                k
                for k, n in self.remaining.items()
                if n > 0 and k[:2] not in self.optional
            ),
            key=str,
        )


@dataclass
class CoreSession:
    model: torch.nn.Module | None = None
    modules: dict[str, ModuleState] = field(default_factory=dict)
    by_id: dict[int, ModuleState] = field(default_factory=dict)
    expected: ExpectedUnits = field(
        default_factory=lambda: ExpectedUnits(source="none")
    )
    exact: bool = False
    trace: "TraceState | None" = None
    observer: WriteObserver | None = None
    # cold start of online-quantized modules: no live tensors to land into
    cold: bool = False


def _core(model: torch.nn.Module) -> CoreSession | None:
    return model.__dict__.get("_reload_core")


def core_active(model: torch.nn.Module) -> bool:
    return _core(model) is not None


# ---------------------------------------------------------------------------
# Routing: the core, or the legacy (layerwise) path
# ---------------------------------------------------------------------------


@dataclass
class TraceState:
    """What a trace or an observed cold load found."""

    expected: ExpectedUnits
    reload_active_queried: bool = False
    outside_writes: set[str] = field(default_factory=set)
    error: str | None = None


_TRACING: ContextVar[TraceState | None] = ContextVar("vllm_reload_trace", default=None)


def note_reload_active_query() -> bool:
    """Called by `is_reload_active`: a model that asks during a trace or the
    cold load guards post-load work on it. Returns True while tracing (a
    trace stands in for a reload)."""
    state = _TRACING.get()
    if state is None:
        return False
    state.reload_active_queried = True
    return True


def _evidence(model: torch.nn.Module, source: str | None) -> TraceState | None:
    found = model.__dict__.get("_reload_traces", {})
    if source is not None and source in found:
        return found[source]
    return found.get("cold")


def legacy_reason(model: torch.nn.Module, source: str | None = None) -> str | None:
    """Why this update takes the legacy (layerwise) path, or None for the
    core. A model that finalizes inside `load_weights` would run that work on
    live bytes mid-stream under the core; a model whose loads were never
    observed may do so unseen."""
    if LEGACY:
        return "VLLM_RELOAD_LEGACY=1"
    if getattr(model, "finalizes_weights_during_load", False):
        return "finalizes_weights_during_load"
    state = _evidence(model, source)
    if state is None:
        return (
            "its loads were never observed (no trainer or checkpoint metadata, and "
            "the cold load was not observed)"
        )
    if state.error is not None:
        return f"the {state.expected.source} trace failed ({state.error})"
    if state.reload_active_queried:
        return "load_weights asks is_reload_active() (post-load work in load_weights)"
    if state.outside_writes:
        shown = sorted(state.outside_writes)[:4]
        return f"load_weights writes model tensors outside weight loaders ({shown})"
    return None


# ---------------------------------------------------------------------------
# Loadable tensors and their snapshot
# ---------------------------------------------------------------------------


def _loadable(module: torch.nn.Module) -> dict[str, tuple[torch.Tensor, bool]]:
    """Params and persistent buffers a weight loader may write."""
    out = {}
    for name, t in module._parameters.items():
        if t is not None and name not in SKIP_LOAD_TENSORS:
            out[name] = (t, False)
    for name, t in module._buffers.items():
        if (
            t is not None
            and name not in SKIP_LOAD_TENSORS
            and name not in module._non_persistent_buffers_set
        ):
            out[name] = (t, True)
    return out


def _snapshot(
    module: torch.nn.Module, name: str, live: torch.Tensor | None
) -> torch.Tensor:
    """A fresh meta stand-in in checkpoint format: the tensor as recorded at
    model init, with its loader attributes bound to `module`."""
    info = lw.LAYERWISE_INFO.get(module)
    meta = None
    if info is not None:
        params, buffers = info.restore_metadata
        meta = params.get(name, buffers.get(name))
    if meta is None:  # registered after init (e.g. `bias`): its live format
        assert live is not None
        meta = live
    return restore_layer_refs(to_meta_tensor(meta), module)


def _fill(module: torch.nn.Module, name: str) -> float:
    info = lw.LAYERWISE_INFO.get(module)
    return 0.0 if info is None else info.init_fill.get(name, 0.0)


def _plain(t: torch.Tensor) -> bool:
    return (
        type(t).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
        and not t.is_meta
        and t.numel() > 0
    )


def _view_over(live: torch.Tensor, snap: torch.Tensor) -> torch.Tensor | None:
    """A checkpoint-layout tensor over `live`'s bytes, or None if they can't
    hold it (footprint, alignment)."""
    elem = snap.element_size()
    offset_bytes = live.storage_offset() * live.element_size()
    storage = live.untyped_storage()
    needed = elem * (
        1 + sum((n - 1) * s for n, s in zip(snap.shape, snap.stride()) if n > 0)
    )
    if offset_bytes % elem or storage.nbytes() < offset_bytes + needed:
        return None
    view = torch.empty(0, dtype=snap.dtype, device=live.device)
    view.set_(storage, offset_bytes // elem, tuple(snap.shape), snap.stride())
    view.__class__ = snap.__class__
    view.__dict__ = snap.__dict__.copy()
    return view


def _post_step(module: torch.nn.Module) -> str:
    method = getattr(module, "quant_method", None)
    if not isinstance(method, QuantizeMethodBase):
        return "none"
    return getattr(method, "reload_post_step", "pwal")


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


def _storage_users(model: torch.nn.Module) -> Counter:
    users: Counter = Counter()
    for module in model.modules():
        for t in (*module._parameters.values(), *module._buffers.values()):
            if t is not None and not t.is_meta:
                key = storage_key(t)
                if key is not None:
                    users[key] += 1
    return users


def _plan_module(
    name: str,
    module: torch.nn.Module,
    users: Counter,
    trace: bool,
    cold: bool = False,
) -> ModuleState | None:
    if type(module).__name__ in SKIP_MODULES:
        return None
    tensors = _loadable(module)
    info = lw.LAYERWISE_INFO.get(module)
    # params recorded at init that are no longer live: deleted by their PWAL
    # (attention KV scales) or set to None. Loaders still write them.
    snap_params = info.restore_metadata[0] if info is not None else {}
    gone = [n for n in snap_params if n not in tensors and n not in SKIP_TENSORS]
    if not tensors and not gone:
        return None
    if cold and not any(t.is_meta for t, _ in tensors.values()):
        return None  # cold start: only online-quantized (meta) modules
    post = "attention" if is_deferred_attention_layer(module) else _post_step(module)
    state = ModuleState(name, module, "direct", post)
    # (None entries too: `register_parameter("bias", None)` must survive)
    state.live_params = dict(module._parameters)
    state.live_buffers = dict(module._buffers)
    state.non_persistent = set(module._non_persistent_buffers_set)
    state.live_attrs = {n: module.__dict__[n] for n in gone if n in module.__dict__}
    fills = info.init_fill if info is not None else {}

    kinds: dict[str, tuple[str, torch.Tensor | None, bool, torch.Tensor, float]] = {}
    for tname, (live, is_buffer) in tensors.items():
        snap = _snapshot(module, tname, live)
        fill = _fill(module, tname)
        if trace or cold or live.is_meta:
            kinds[tname] = (SCRATCH, live, is_buffer, snap, fill)
            continue
        exclusive = users[storage_key(live)] == 1
        if tname in SKIP_TENSORS or (
            type(live) is type(snap)
            and live.shape == snap.shape
            and live.stride() == snap.stride()
            and live.dtype == snap.dtype
            and (live.numel() == 0 or exclusive)
        ):
            # (`bias` is built after create_weights and never snapshotted: it
            # is always written in place)
            kinds[tname] = (AS_IS, live, is_buffer, live, fill)
            continue
        view = _view_over(live, snap) if _plain(live) and exclusive else None
        if view is not None:
            kinds[tname] = (VIEW, live, is_buffer, view, fill)
        elif tname in fills:
            kinds[tname] = (SIDE, live, is_buffer, snap, fill)
        else:
            kinds[tname] = (SCRATCH, live, is_buffer, snap, fill)
    for tname in gone:
        snap = _snapshot(module, tname, None)
        kind = SIDE if tname in fills else SCRATCH
        kinds[tname] = (kind, None, False, snap, _fill(module, tname))

    # (a module without a post step can only take the direct path as-is: its
    # live layout must be the checkpoint layout)
    direct = (
        DIRECT_ENABLED
        and not trace
        and not cold
        and all(k[0] in (AS_IS, VIEW, SIDE) for k in kinds.values())
        and any(k[0] in (AS_IS, VIEW) for k in kinds.values())
        and not (post == "none" and any(k[0] != AS_IS for k in kinds.values()))
    )
    if not direct:
        state.path = "modulewise"
        for tname, (kind, live, is_buffer, snap, fill) in kinds.items():
            if kind == AS_IS and tname in SKIP_TENSORS:
                continue  # `bias` stays live even in a MODULEWISE module
            if kind in (AS_IS, VIEW):  # a stand-in of the snapshot, not live
                # (an as-is live tensor has the recorded layout, but may have
                # lost loader attributes such as `quant_method`)
                snap = _snapshot(module, tname, live)
            if kind != SCRATCH:
                kinds[tname] = (SCRATCH, live, is_buffer, snap, fill)
    for tname, (kind, live, is_buffer, standin, fill) in kinds.items():
        state.targets[tname] = Target(tname, kind, is_buffer, live, standin, fill)
    return state


def _register(module: torch.nn.Module, name: str, t: torch.Tensor, buffer: bool):
    module.__dict__.pop(name, None)  # set aside in `live_attrs`
    if name in module._parameters:
        del module._parameters[name]
    if name in module._buffers:
        del module._buffers[name]
    if buffer:
        module.register_buffer(name, t, persistent=True)
    else:
        module.register_parameter(name, t)  # type: ignore[arg-type]


def _original_loader(t: torch.Tensor) -> Callable:
    loader = getattr(t, "weight_loader", default_weight_loader)
    while getattr(loader, "__name__", None) in ("online_process_loader", "core_loader"):
        loader = loader.__wrapped__  # type: ignore[union-attr]
    return loader


def _install(cs: CoreSession, state: ModuleState) -> None:
    """Register the stand-ins and wrap their loaders."""
    for target in state.targets.values():
        if target.kind == AS_IS:
            assert target.live is not None
            target.had_loader = "weight_loader" in target.live.__dict__
            target.loader_attr = target.live.__dict__.get("weight_loader")
            recorded = _snapshot(state.module, target.name, target.live).__dict__
            for key, value in recorded.items():
                if key != "weight_loader" and key not in target.live.__dict__:
                    target.live.__dict__[key] = value
                    target.added_attrs.append(key)
        else:
            _register(state.module, target.name, target.standin, target.is_buffer)
        target.standin.weight_loader = _make_loader(  # type: ignore[attr-defined]
            cs, state, target.name, _original_loader(target.standin)
        )


def _uninstall(state: ModuleState) -> None:
    """Put the original live objects back and unwrap loaders."""
    for target in state.targets.values():
        if target.kind == AS_IS:
            assert target.live is not None
            if target.had_loader:
                target.live.weight_loader = target.loader_attr  # type: ignore[attr-defined]
            else:
                target.live.__dict__.pop("weight_loader", None)
            for key in target.added_attrs:
                target.live.__dict__.pop(key, None)
            target.added_attrs.clear()
    _place_originals(state)


def _place_originals(state: ModuleState) -> None:
    module = state.module
    names = set(module._parameters) | set(module._buffers)
    names |= set(state.live_params) | set(state.live_buffers)
    for name in names:
        module._parameters.pop(name, None)
        module._buffers.pop(name, None)
    for name, t in state.live_params.items():
        module.register_parameter(name, t)  # type: ignore[arg-type]
    for name, b in state.live_buffers.items():
        module.register_buffer(name, b, persistent=name not in state.non_persistent)
    for name, value in state.live_attrs.items():
        module.__dict__[name] = value


def _expected_for(model: torch.nn.Module, source: str | None) -> ExpectedUnits:
    found = model.__dict__.get("_reload_traces", {})
    for key in (source, "cold"):
        if key is not None and key in found:
            return found[key].expected
    return _structure_expected(model)


def _structure_expected(model: torch.nn.Module) -> ExpectedUnits:
    """One wildcard `(tensor, "*")` per loadable tensor: used only when nothing
    else describes the update (e.g. after a dummy load)."""
    expected = ExpectedUnits(source="structure")
    for name, module in model.named_modules():
        for tname in _loadable(module):
            expected.add(name, (tname, "*"), WILDCARD)
            expected.optional.setdefault(name, set()).add((tname, "*"))
    return expected


def start_core(
    model: torch.nn.Module,
    partial: bool | None = None,
    modules: list[str] | tuple[str, ...] | None = None,
    source: str | None = None,
    cold: bool = False,
) -> None:
    """Begin an update on the core. `source` names the expected-unit source
    ("trainer" or a checkpoint key); the cold-load observation is the
    fallback. `cold`: the cold load of online-quantized modules (no live
    tensors, no integrity session; see `load_cold_online`)."""
    if not cold:
        lw._check_live_tensors_unchanged(model)
        lw._check_model_hook_reload_safe(model)
        lw._check_moe_methods_reload_safe(model)
        session = lw.begin_update(model, partial=partial, modules=modules)
        session.coverage_checked = True  # the core checks unit coverage itself
    cs = CoreSession(model=model, expected=_expected_for(model, source), cold=cold)
    cs.exact = cs.expected.exact
    model.__dict__["_reload_core"] = cs
    _ACTIVE.add(model)
    try:
        users = _storage_users(model)
        for name, module in model.named_modules():
            state = _plan_module(name, module, users, trace=False, cold=cold)
            if state is None:
                continue
            cs.modules[name] = state
            cs.by_id[id(module)] = state
            state.expected = Counter(cs.expected.units.get(name, Counter()))
            state.remaining = Counter(state.expected)
            state.pending = sum(state.remaining.values())
            by_unit: dict[tuple[str, Hashable], list[tuple]] = {}
            for key in state.remaining:
                by_unit.setdefault(key[:2], []).append(key)
            # (tuples of atomics: the garbage collector stops tracking them)
            state.by_unit = {k: tuple(v) for k, v in by_unit.items()}
            state.optional = set(cs.expected.optional.get(name, set()))
            if cold:  # layerwise's online-processing wrappers stand down
                lw.get_layerwise_info(module).reset()
            _install(cs, state)
        if OBSERVE_WRITES and not cold:
            cs.observer = _make_observer(cs)
            cs.observer.__enter__()
    except BaseException:
        abort_core(model)
        if not cold:
            lw.abort_reload(model, dirty=False)
        raise
    paths = Counter(s.path for s in cs.modules.values())
    kinds = Counter(t.kind for s in cs.modules.values() for t in s.targets.values())
    if not cold:
        model.__dict__["_reload_last_plan"] = {
            "paths": dict(paths),
            "targets": dict(kinds),
            "source": cs.expected.source,
        }
    logger.debug(
        "reload core: %s modules, targets %s, expected units from %s",
        dict(paths),
        dict(kinds),
        cs.expected.source,
    )


def _make_observer(cs: CoreSession) -> WriteObserver:
    owners = {}
    for state in cs.modules.values():
        for target in state.targets.values():
            for t in (target.standin, target.live):
                if t is not None and (key := storage_key(t)) is not None:
                    owners[key] = f"{state.name}.{target.name}"
    return WriteObserver(owners)


# ---------------------------------------------------------------------------
# receive
# ---------------------------------------------------------------------------


def _make_loader(
    cs: CoreSession, state: ModuleState, tname: str, original: Callable
) -> Callable:
    signature = inspect.signature(original)
    first = next(iter(signature.parameters), None)

    def core_loader(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if state.completed:
            raise ReloadUnitError(
                f"{state.name}.{tname}: weights arrived after the module completed "
                f"(its expected units, from {cs.expected.source}, were all in). "
                "If this sender sends more than the cold load did, ship its "
                "metadata at init_weight_transfer_engine."
            )
        if cs.trace is None and _buffers(state, bound, first):
            return _buffer(cs, state, tname, bound, original)
        _touch(cs, state)
        _first_write(cs, state, tname)
        if first is not None and first in bound.arguments:
            bound.arguments[first] = getattr(state.module, tname)
        call = LoadCall(state.module, tname)
        token = _CURRENT_LOAD.set(call)
        try:
            with suspended():
                if SHADOW_COUNT and cs.trace is None:
                    numel, ret = get_numel_loaded(original, bound)
                    state.counted += numel
                else:
                    ret = original(*bound.args, **bound.kwargs)
        finally:
            _CURRENT_LOAD.reset(token)
        if call.declined:
            return ret
        units = call.units or [(tname, infer_shard(bound))]
        _arrive(cs, state, units, current_checkpoint_name())
        if SHADOW_COUNT and cs.trace is None:
            _shadow(cs, state)
        _maybe_complete(cs, state)
        return ret

    core_loader.__wrapped__ = original  # type: ignore[attr-defined]
    return core_loader


def _take(state: ModuleState, key: tuple) -> None:
    state.remaining[key] -= 1
    state.pending -= 1


def _maybe_complete(cs: CoreSession, state: ModuleState) -> None:
    if (
        cs.trace is None
        and _completes_when_loaded(state)
        and not state.format_mismatch
        and state.pending == 0
    ):
        _complete(cs, state)


def _buffers(
    state: ModuleState, bound: inspect.BoundArguments, first: str | None
) -> bool:
    """CPU incoming into a MODULEWISE module is buffered (host memory) until
    the module's units arrived: first-touch scratch for every module a
    checkpoint shard touches would hold the checkpoint format of many modules
    on the device. Device incoming (weight transfer) loads into scratch."""
    if state.path != "modulewise" or state.post == "attention":
        return False
    if state.buffering:
        return True
    if state.materialized:
        return False
    incoming = [
        v
        for k, v in bound.arguments.items()
        if k != first and isinstance(v, torch.Tensor)
    ]
    return bool(incoming) and all(v.device.type in ("cpu", "meta") for v in incoming)


def _buffer(
    cs: CoreSession,
    state: ModuleState,
    tname: str,
    bound: inspect.BoundArguments,
    original: Callable,
):
    """Keep the call for replay at completion, credited now by the checkpoint
    name it arrived under: the loader runs once, at replay."""
    state.buffering = True
    _mark_touched(cs, state)
    state.buffered.append((tname, bound, original))
    name = current_checkpoint_name()
    shard = infer_shard(bound)
    credited = True
    if name is None:
        _arrive(cs, state, [(tname, shard)], None)
    else:
        key: tuple | None = (tname, shard, name)
        if state.remaining[key] <= 0:  # another shard under this name
            key = next(
                (
                    k
                    for k, n in state.remaining.items()
                    if n > 0 and k[0] == tname and k[2] == name
                ),
                None,
            )
        if key is not None:
            _take(state, key)
            state.arrived[key[:2]] += 1
        else:
            # (declined at the trace, e.g. a non-local expert, or unexpected)
            credited = any(k[0] == tname and k[2] == name for k in state.expected)
            if not credited and not any(k[2] == name for k in state.expected):
                _arrive(cs, state, [(tname, shard)], name)
                credited = True
    _maybe_complete(cs, state)
    if bound.arguments.get("return_success"):
        return credited
    return None


def _shadow(cs: CoreSession, state: ModuleState) -> None:
    """R1: when would layerwise reload's element count have completed this
    module, compared with its units?"""
    if state.count_total == 0:
        # layerwise counts the tensors recorded at init (its meta restore),
        # not ones a PWAL added (e.g. online quantization's scales)
        info = lw.LAYERWISE_INFO.get(state.module)
        recorded = (
            {**info.restore_metadata[0], **info.restore_metadata[1]}
            if info is not None
            else {}
        )
        state.count_total = sum(
            get_tensor_load_numel(recorded.get(n, t.standin))
            for n, t in state.targets.items()
            if not recorded or n in recorded
        )
    units_done = state.pending == 0
    counter_done = state.count_total > 0 and state.counted >= state.count_total
    if counter_done and not state.counter_completed:
        state.counter_completed = True
        if not units_done:
            _shadow_event(cs, state, "counter completes before the units arrived")
    if units_done and not counter_done and not state.format_mismatch:
        _shadow_event(cs, state, "units arrived, counter still incomplete")


def _shadow_event(cs: CoreSession, state: ModuleState, kind: str) -> None:
    events = _model_of(cs).__dict__.setdefault("_reload_shadow", Counter())
    events[(kind, type(state.module).__name__)] += 1


def _completes_when_loaded(state: ModuleState) -> bool:
    """Modules that complete as soon as their units arrived: PWAL ones, and
    modulewise ones without a post step (their scratch still lands). DIRECT
    modules without a post step have no completion event."""
    return state.post == "pwal" or (state.post == "none" and state.path == "modulewise")


def _mark_touched(cs: CoreSession, state: ModuleState) -> None:
    if not state.touched:
        state.touched = True
    if cs.trace is None and not cs.cold:
        session = lw.get_reload_session(_model_of(cs))
        if session is not None:
            session.touched.add(state.name)


def _model_of(cs: CoreSession) -> torch.nn.Module:
    assert cs.model is not None
    return cs.model


def _touch(cs: CoreSession, state: ModuleState) -> None:
    """First touch: materialize the module's side and scratch targets, and
    start DIRECT targets no loader call is expected to write from their
    create-time value. From here on a DIRECT module's live bytes hold this
    update."""
    _mark_touched(cs, state)
    if state.materialized or cs.trace is not None:
        return
    state.materialized = True
    module = state.module
    info = lw.LAYERWISE_INFO.get(module)
    default = info.restore_device if info is not None else torch.get_default_device()
    expected = {k[0] for k in state.expected}
    token = FRAMEWORK_WRITES.set(True)
    with torch.no_grad():
        for target in state.targets.values():
            if target.kind in (SIDE, SCRATCH):
                live = target.live
                device = (
                    live.device if live is not None and not live.is_meta else default
                )
                with device:
                    real = materialize_meta_tensor(target.standin)
                fill_init_value(real, target.fill)
                target.standin = real
                _register(module, target.name, real, target.is_buffer)
                target.filled = True
            elif target.name not in expected and state.path == "direct":
                # no loader call is expected to write it (a sibling's loader
                # may): it starts from its create-time value, as at a fresh load
                fill_init_value(target.standin, target.fill)
                target.filled = True
    FRAMEWORK_WRITES.reset(token)
    if state.path == "direct" and not cs.cold:
        session = lw.get_reload_session(_model_of(cs))
        if session is not None:
            session.written.add(state.name)


def _first_write(cs: CoreSession, state: ModuleState, tname: str) -> None:
    """A DIRECT target starts from its create-time value on the first loader
    call that writes it (a tensor no loader writes keeps its live value)."""
    target = state.targets[tname]
    if target.filled or cs.trace is not None:
        return
    if target.kind == AS_IS and not cs.cold:
        # live bytes change from here on (also `bias` in a MODULEWISE module)
        session = lw.get_reload_session(_model_of(cs))
        if session is not None:
            session.written.add(state.name)
    token = FRAMEWORK_WRITES.set(True)
    with torch.no_grad():
        fill_init_value(target.standin, target.fill)
    FRAMEWORK_WRITES.reset(token)
    target.filled = True


def _arrive(
    cs: CoreSession, state: ModuleState, units: list[Unit], name: str | None
) -> None:
    for tname, shard in units:
        state.arrived[(tname, shard)] += 1
        if cs.trace is not None:
            cs.trace.expected.add(state.name, (tname, shard), name)
            continue
        key = (tname, shard, name)
        if state.remaining[key] > 0:
            _take(state, key)
            continue
        alt = next(
            (
                k
                for k in state.by_unit.get((tname, shard), ())
                if state.remaining[k] > 0
            ),
            None,
        )
        if alt is not None:
            _take(state, alt)
            continue
        if cs.exact and any(k[:2] == (tname, shard) for k in state.expected):
            raise ReloadUnitError(
                f"{state.name}.{tname}: unit {shard!r} (from {name!r}) arrived more "
                "times than the sender's metadata declares"
            )
        if not state.format_mismatch:
            state.format_mismatch = True
            logger.warning_once(
                "%s: unit (%s, %r) from %r was not expected (expected units from "
                "%s); the module completes at finish",
                state.name,
                tname,
                shard,
                name,
                cs.expected.source,
            )


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------


def _complete(cs: CoreSession, state: ModuleState) -> None:
    """PWAL on the module's stand-ins, land the results in the live tensors,
    put the original objects back."""
    token = FRAMEWORK_WRITES.set(True)
    try:
        _complete_module(cs, state)
    finally:
        FRAMEWORK_WRITES.reset(token)


def _complete_module(cs: CoreSession, state: ModuleState) -> None:
    module = state.module
    session = None if cs.cold else lw.get_reload_session(_model_of(cs))
    if session is not None:
        session.completed.add(state.name)
        session.written.add(state.name)
    _replay(cs, state)
    if hasattr(module, "_already_called_process_weights_after_loading"):
        delattr(module, "_already_called_process_weights_after_loading")
    for target in state.targets.values():  # PWAL sees unwrapped loaders
        target.standin.weight_loader = _original_loader(  # type: ignore[attr-defined]
            target.standin
        )
    if state.post in ("pwal", "pwal_at_finish"):
        method = module.quant_method
        from vllm.model_executor.layers.quantization.online.fp8 import (
            OnlineLinearBase,
        )
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        timing = (
            online_quantization_timing_context(module, method)
            if isinstance(method, (OnlineLinearBase, OnlineMoEMethodBase))
            else nullcontext()
        )
        if not cs.cold:
            state.held = {
                n: t.detach()
                for n, t in (*state.live_params.items(), *state.live_buffers.items())
                if t is not None and not t.is_meta
            }
        with timing:
            method.process_weights_after_loading(module)
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()
    state.completed = True
    if cs.cold:
        # the PWAL results are the module's tensors: nothing to land into
        for t in (*module._parameters.values(), *module._buffers.values()):
            if t is not None and "weight_loader" in t.__dict__:
                t.weight_loader = _original_loader(t)  # type: ignore[attr-defined]
        _release(state)  # the checkpoint-format scratch the PWAL replaced
        return
    _land(state)
    _uninstall(state)
    _release(state)


def _release(state: ModuleState) -> None:
    """Drop scratch and side tensors now: their wrapped loaders reference the
    session, so a cycle would otherwise keep them until a GC collection."""
    for target in state.targets.values():
        if target.kind in (SIDE, SCRATCH):
            target.standin = target.standin.new_empty(0)


def _replay(cs: CoreSession, state: ModuleState) -> None:
    """Run the module's buffered loader calls on its (now materialized)
    targets, once."""
    if not state.buffered:
        return
    _touch(cs, state)
    module = state.module
    for tname, bound, original in state.buffered:
        first = next(iter(bound.signature.parameters), None)
        if first is not None and first in bound.arguments:
            bound.arguments[first] = getattr(module, tname)
        token = _CURRENT_LOAD.set(LoadCall(module, tname))
        try:
            with suspended():
                original(*bound.args, **bound.kwargs)
        finally:
            _CURRENT_LOAD.reset(token)
    state.buffered.clear()
    state.buffering = False


def _land(state: ModuleState) -> None:
    """Copy each result into its live tensor: in place when the result is the
    live tensor, an exact-checked copy otherwise, cloned first when it shares
    storage with live bytes. All sources are resolved before the first copy."""
    module = state.module
    live_all = {
        n: t
        for n, t in (*state.live_params.items(), *state.live_buffers.items())
        if t is not None
    }
    live_storages = {
        key for t in live_all.values() if (key := lw._storage_ptr(t)) is not None
    }
    pending = []
    for name, live in live_all.items():
        if live.is_meta:
            continue
        result = module._parameters.get(name, module._buffers.get(name))
        held = state.held.get(name)
        if held is not None and not lw._same_view(live, held):
            # the PWAL rebound `live.data`: its new bytes are the result, the
            # live object goes back to its storage
            rebound = live.data
            live.data = held
            if result is live:
                result = rebound
        if result is None or result is live or lw._same_view(result, live):
            continue
        if (
            name in state.non_persistent
            and name not in state.targets
            and lw.check_exact_landing(live, result) is not None
        ):
            # a derived buffer the PWAL recomputed in another layout: the
            # kernel refresh rewrites it
            continue
        # (scratch is a fresh allocation: only a DIRECT result can alias live)
        if (
            state.path == "direct"
            and result.numel() > 0
            and lw._storage_ptr(result) in live_storages
        ):
            result = result.clone()
        pending.append((name, live, result))
    for name, live, result in pending:
        lw._land(module, name, live, result)
    state.held = {}


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------


def finish_core(model: torch.nn.Module, model_config: ModelConfig | None) -> None:
    cs = _core(model)
    assert cs is not None, "finish_core without start_core"
    session = lw.get_reload_session(model)
    if cs.observer is not None:
        cs.observer.__exit__(None, None, None)
        observer, cs.observer = cs.observer, None
        _check_observed(observer)
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload

    if cs.cold:
        _finish_cold(model, cs)
        return
    # 1. coverage, before any finish-time transform
    _check_coverage(model, cs, session.partial if session is not None else None)
    order = list(cs.modules.values())
    # 2. modules still open (optional units missing, or a format mismatch)
    for state in order:
        if _completes_when_loaded(state) and state.touched and not state.completed:
            _complete(cs, state)
    # 3. finish-time PWAL, in tree order, after every other module landed
    for state in order:
        if state.post == "pwal_at_finish" and state.touched and not state.completed:
            _complete(cs, state)
    # 4. original objects back everywhere else
    for state in order:
        if state.post == "attention" or state.completed:
            continue
        if state.touched and session is not None:  # DIRECT, no post step
            session.completed.add(state.name)
        _uninstall(state)
        _release(state)
        state.completed = True
    # 5. kernel refresh, every reload-safe quant method, touched or not
    for layer in model.modules():
        refresh_quant_method(layer, getattr(layer, "quant_method", None))
    # 6. attention phase, every attention layer: KV scales that arrived sit in
    # side tensors and land here; MLA derives W_UK_T / W_UV from the landed
    # kv_b_proj; backends refresh their scale-derived state
    act_dtype = getattr(model_config, "dtype", None) or torch.get_default_dtype()
    for layer in model.modules():
        if is_deferred_attention_layer(layer):
            _attention_phase(cs, cs.by_id.get(id(layer)), layer, act_dtype)
    # 7. layer and model refreshes
    run_model_refresh(model)
    del model.__dict__["_reload_core"]
    _ACTIVE.discard(model)
    # 8. integrity
    if session is not None and session.active:
        lw.end_update(model)


def _attention_phase(
    cs: CoreSession,
    state: ModuleState | None,
    layer: torch.nn.Module,
    act_dtype: torch.dtype,
) -> None:
    old_floats = lw._kv_scale_floats(layer)
    if state is not None and not state.completed:
        if state.touched:
            # the KV-cache method consumes the arrived scales (side tensors)
            # into the layer's live scale buffers, then deletes them
            method = getattr(layer, "quant_method", None)
            token = FRAMEWORK_WRITES.set(True)
            try:
                if isinstance(method, QuantizeMethodBase):
                    method.process_weights_after_loading(layer)
                _land(state)
            finally:
                FRAMEWORK_WRITES.reset(token)
            session = lw.get_reload_session(_model_of(cs))
            if session is not None:
                session.completed.add(state.name)
        _uninstall(state)
        _release(state)
        state.completed = True
    lw._rerun_attention_pwal(layer, act_dtype, old_floats)


def _finish_cold(model: torch.nn.Module, cs: CoreSession) -> None:
    """End of an online-quantized cold load: every module completes (one the
    checkpoint didn't reach starts from its create-time values, as layerwise
    processing did); the rest of post-load processing runs as usual."""
    for state in cs.modules.values():
        if not state.completed:
            _touch(cs, state)
            _complete(cs, state)
    del model.__dict__["_reload_core"]
    _ACTIVE.discard(model)


def _check_coverage(
    model: torch.nn.Module, cs: CoreSession, partial: bool | None
) -> None:
    """A touched module missing a required unit fails the update (its live
    bytes would be mixed); untouched modules fail only a full update."""
    missing: dict[str, list] = {}
    untouched = []
    aliases = _aliased_modules(cs)
    for state in cs.modules.values():
        required = state.required_missing()
        if state.format_mismatch and state.touched:
            # the expected set doesn't describe this sender: every tensor the
            # expected set names must still have received data
            required = sorted(
                {k[0] for k in state.expected}
                - {t for (t, _), n in state.arrived.items() if n > 0}
            )
        if not required:
            continue
        if state.touched:
            missing[state.name] = required
        elif state.expected and state.name not in aliases:
            untouched.append(state.name)
    if missing:
        shown = {k: v[:4] for k, v in list(missing.items())[:6]}
        raise lw.ReloadIncompleteError(
            f"{len(missing)} module(s) received only part of their weights "
            f"(expected units from {cs.expected.source}): "
            f"{shown}"
        )
    if untouched and partial is False:
        raise lw.ReloadIncompleteError(
            f"Full weight update incomplete, {len(untouched)} module(s) received no "
            f"weights: {lw._shown(untouched)}. Start the update with partial=True "
            "to update a subset of the model."
        )
    if untouched and partial is None:
        logger.warning(
            "%d module(s) received no weights: %s. They keep their previous "
            "weights; start the update with partial=True to mark this intended, "
            "or partial=False to require a full update.",
            len(untouched),
            lw._shown(untouched),
        )


def _aliased_modules(cs: CoreSession) -> set[str]:
    """Modules whose params all belong to an earlier module (a tied `lm_head`,
    also one re-tied to the embeddings after the cold load): an update reaches
    them through that module's loader, as layerwise reload counts a shared
    param for its first owner. (From the live params: stand-ins replace them
    during the update.)"""
    first_owner: dict[int, str] = {}
    aliases = set()
    for name, state in cs.modules.items():
        owners = [
            first_owner.setdefault(id(t), name)
            for t in state.live_params.values()
            if t is not None
        ]
        if owners and all(o != name for o in owners):
            aliases.add(name)
    return aliases


def _check_observed(observer: WriteObserver) -> None:
    if observer.record.outside_loader:
        raise RuntimeError(
            "write observer: model tensors written outside weight loaders: "
            f"{sorted(observer.record.outside_loader)[:8]}"
        )


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


def abort_core(model: torch.nn.Module) -> None:
    """Put the original objects back on every module and drop the core's
    state (the integrity side is `layerwise.abort_reload`)."""
    cs = _core(model)
    if cs is None:
        return
    if cs.observer is not None:
        cs.observer.__exit__(None, None, None)
    for state in cs.modules.values():
        if state.completed:
            continue
        try:
            _uninstall(state)
        except Exception:  # noqa: BLE001
            logger.exception("reload core: failed to restore %s", state.name)
    del model.__dict__["_reload_core"]
    _ACTIVE.discard(model)


# ---------------------------------------------------------------------------
# trace and cold-load observation
# ---------------------------------------------------------------------------


def _owners(model: torch.nn.Module) -> dict[int, str]:
    owners = {}
    for name, module in model.named_modules():
        for tname, (t, _) in _loadable(module).items():
            if (key := storage_key(t)) is not None:
                owners[key] = f"{name}.{tname}"
    return owners


@contextmanager
def trace_loads(model: torch.nn.Module, source: str) -> Iterator[TraceState]:
    """Dry run: every loadable tensor is a meta stand-in, loaders run on them
    and record their units; nothing is written. A write to a model tensor
    outside a loader, or a call to `is_reload_active`, is recorded."""
    # not is_reload_active(): that records a query on an enclosing trace (the
    # cold load's, around the cold header trace)
    session = lw.get_reload_session(model)
    if _core(model) is not None or (session is not None and session.active):
        raise RuntimeError("trace_loads: a weight update is active")
    expected = ExpectedUnits(source=source, exact=True)
    state = TraceState(expected=expected)
    cs = CoreSession(model=model, expected=expected, exact=True, trace=state)
    model.__dict__["_reload_core"] = cs
    token = _TRACING.set(state)
    try:
        for name, module in model.named_modules():
            st = _plan_module(name, module, Counter(), trace=True)
            if st is None:
                continue
            cs.modules[name] = st
            _install(cs, st)
        observer = WriteObserver(_owners(model), dry_run=True)
        for st in cs.modules.values():
            for target in st.targets.values():
                observer.register(target.standin, f"{st.name}.{target.name}")
        with observer:
            yield state
        state.outside_writes |= observer.record.outside_loader
    finally:
        _TRACING.reset(token)
        for st in cs.modules.values():
            _uninstall(st)
        del model.__dict__["_reload_core"]


def run_trace(
    model: torch.nn.Module,
    metadata: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
    source: str,
) -> TraceState:
    """Trace `model.load_weights` over checkpoint names, shapes and dtypes
    (meta stand-ins, no data) and keep the result as the `source` expected
    set and routing evidence."""
    from .units import track_names

    meta = list(metadata)
    try:
        with trace_loads(model, source) as state:

            def stand_ins():
                for name, shape, dtype in meta:
                    yield name, torch.empty(shape, dtype=dtype, device="meta")

            model.load_weights(track_names(stand_ins()))
    except Exception as e:  # noqa: BLE001
        state = TraceState(expected=ExpectedUnits(source=source))
        state.error = f"{type(e).__name__}: {str(e)[:200]}"
        logger.warning(
            "Reload trace over %s metadata failed (%s); updates from this source "
            "use the legacy reload path.",
            source,
            state.error,
        )
    model.__dict__.setdefault("_reload_traces", {})[source] = state
    return state


@contextmanager
def observe_cold_load(model: torch.nn.Module) -> Iterator[None]:
    """Record the units of the real cold load (the fallback expected set) and
    whether `load_weights` writes model tensors outside loaders or asks
    `is_reload_active`. Loaders write the live tensors as usual."""
    if not envs.VLLM_RELOAD_OBSERVE_COLD:
        yield
        return
    expected = ExpectedUnits(source="cold")
    state = TraceState(expected=expected)
    cs = CoreSession(model=model, expected=expected, trace=state)
    wrapped: list[tuple[torch.Tensor, bool, Callable | None]] = []
    for name, module in model.named_modules():
        if type(module).__name__ in SKIP_MODULES:
            continue
        st = ModuleState(name, module, "observe", "none")
        for tname, (t, _) in _loadable(module).items():
            loader = getattr(t, "weight_loader", None)
            if getattr(loader, "__name__", "") == "core_loader":
                continue  # shared tensor, wrapped by its first owner
            wrapped.append(
                (t, "weight_loader" in t.__dict__, t.__dict__.get("weight_loader"))
            )
            t.weight_loader = _make_observing_loader(  # type: ignore[attr-defined]
                cs, st, tname, getattr(t, "weight_loader", default_weight_loader)
            )
    observer = WriteObserver(_owners(model))
    token = _TRACING.set(state)
    try:
        with observer:
            yield
    finally:
        _TRACING.reset(token)
        for t, had, attr in wrapped:
            if had:
                t.weight_loader = attr  # type: ignore[attr-defined]
            else:
                t.__dict__.pop("weight_loader", None)
    state.outside_writes |= observer.record.outside_loader
    # Modules the core loaded at cold start (online quantization) aren't
    # observed here: the trace over the same checkpoint describes them
    cold_trace = model.__dict__.get("_reload_traces", {}).get(COLD_CHECKPOINT)
    if cold_trace is not None:
        for module_name, units in cold_trace.expected.units.items():
            expected.units.setdefault(module_name, Counter(units))
    # A unit is optional when its tensor was created with a meaningful value
    # a fresh load keeps if the checkpoint lacks it (ones, scale sentinels,
    # -inf): the cold load's checkpoint may carry it where a sender doesn't
    for module_name, units in expected.units.items():
        module = model.get_submodule(module_name)
        info = lw.LAYERWISE_INFO.get(module)
        fills = info.init_fill if info is not None else {}
        expected.optional[module_name] = {
            (t, s) for (t, s, _) in units if fills.get(t, 0.0) != 0.0
        }
    model.__dict__.setdefault("_reload_traces", {})["cold"] = state


def _make_observing_loader(
    cs: CoreSession, state: ModuleState, tname: str, original: Callable
) -> Callable:
    signature = inspect.signature(original)

    def core_loader(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = LoadCall(state.module, tname)
        token = _CURRENT_LOAD.set(call)
        try:
            with suspended():
                ret = original(*bound.args, **bound.kwargs)
        finally:
            _CURRENT_LOAD.reset(token)
        if not call.declined:
            units = call.units or [(tname, infer_shard(bound))]
            for unit in units:
                cs.expected.add(state.name, unit, current_checkpoint_name())
        return ret

    core_loader.__wrapped__ = original  # type: ignore[attr-defined]
    return core_loader


# ---------------------------------------------------------------------------
# Public per-module API, for engines that route bytes themselves (sharded RDT,
# nccl_m2n's planned direct writes)
# ---------------------------------------------------------------------------


def _state_of(module: torch.nn.Module) -> tuple[CoreSession, ModuleState]:
    for model_ref in list(_ACTIVE):
        cs = _core(model_ref)
        state = None if cs is None else cs.by_id.get(id(module))
        if cs is not None and state is not None and state.module is module:
            return cs, state
    raise RuntimeError(
        f"{type(module).__name__} is not part of an active weight update "
        "(start_reload / start_weight_update must run first)"
    )


def ensure_materialized(
    module: torch.nn.Module, tensor_names: Iterable[str] | None = None
) -> dict[str, torch.Tensor]:
    """First touch for engines that write bytes themselves: start the module's
    targets (scratch materialized; the named DIRECT targets, all by default,
    from their create-time value) and return them by name. A write must land
    in these, at the target's own storage offset."""
    cs, state = _state_of(module)
    if state.completed:
        raise RuntimeError(f"{state.name}: already completed in this update")
    _touch(cs, state)
    for tname in state.targets if tensor_names is None else tensor_names:
        _first_write(cs, state, tname)
    return {name: getattr(module, name) for name in state.targets}


def record_units(
    module: torch.nn.Module,
    units: Iterable[Unit] | None = None,
    name: str | None = None,
) -> bool:
    """Record units a planned write delivered to `module` (all of its expected
    units when `units` is None: the engine's plan covers the module). Returns
    whether the module's expected units have all arrived."""
    cs, state = _state_of(module)
    _mark_touched(cs, state)
    if units is None:
        units = [k[:2] for k, n in state.remaining.items() for _ in range(n)]
    _arrive(cs, state, list(units), name)
    return state.pending == 0


def complete_module(module: torch.nn.Module) -> None:
    """Complete `module` now (PWAL, landing, original objects back), e.g. on an
    engine's post-processing thread once its plan delivered the module.
    Attention layers complete in the attention phase at finish."""
    cs, state = _state_of(module)
    if state.completed or not _completes_when_loaded(state):
        return
    _complete(cs, state)


# models with an active update, for the per-module API above
_ACTIVE: "weakref.WeakSet[torch.nn.Module]" = weakref.WeakSet()


# ---------------------------------------------------------------------------
# Cold start of online-quantized modules
# ---------------------------------------------------------------------------

COLD_CHECKPOINT = "cold-checkpoint"


def _has_online_modules(model: torch.nn.Module) -> bool:
    return any(
        getattr(getattr(m, "quant_method", None), "uses_meta_device", False)
        for m in model.modules()
    )


@contextmanager
def cold_online_load(model: torch.nn.Module, files: list[str]) -> Iterator[None]:
    """The cold load of online-quantized modules on the core: their BF16
    checkpoint tensors (CPU incoming) are buffered per module, credited by
    checkpoint name against a trace over the checkpoint's safetensors headers,
    and replayed into scratch when the module's units arrived; the module is
    quantized then (MODULEWISE, no live tensors). Without headers to trace
    (`.bin`, GGUF), or with VLLM_RELOAD_LEGACY=1, layerwise online processing
    runs as before."""
    if LEGACY or not files or not _has_online_modules(model):
        yield
        return
    state = run_trace(model, checkpoint_metadata(files), COLD_CHECKPOINT)
    if state.error is not None or legacy_reason(model, COLD_CHECKPOINT) is not None:
        yield
        return
    start_core(model, source=COLD_CHECKPOINT, cold=True)
    try:
        yield
    except BaseException:
        abort_core(model)
        raise
    finish_core(model, None)


_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def checkpoint_metadata(
    files: list[str],
) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """Names, shapes and dtypes of safetensors files (headers only)."""
    from safetensors import safe_open

    meta = []
    for file in files:
        with safe_open(file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                s = f.get_slice(name)
                dtype = _SAFETENSORS_DTYPES.get(s.get_dtype())
                if dtype is not None:  # e.g. F8 variants torch lacks
                    meta.append((name, tuple(s.get_shape()), dtype))
    return meta
