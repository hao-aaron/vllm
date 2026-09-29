# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import reload_mode

from . import direct as _direct
from . import meta as _meta
from . import per_expert as _per_expert
from .meta import (
    SKIP_LOAD_TENSORS,
    SKIP_MODULES,
    SKIP_TENSORS,
    capture_layer_to_meta,
    get_numel_loaded,
    restore_layer_on_meta,
    to_meta_tensor,
)
from .types import LayerReloadingInfo, ReloadSession
from .utils import (
    get_info_size,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "complete_module",
    "ensure_param_materialized",
    "scratch_bytes_in_flight",
    "start_reload",
    "finish_reload",
    "abort_reload",
    "trace_loads",
    "current_load",
    "ensure_materialized",
    "LoadTarget",
    "LoadTrace",
    "add_reload_transform",
    "get_reload_session",
    "is_model_dirty",
    "check_can_serve",
    "mark_dirty",
]


# Global dict storing information used for layerwise restoring, loading, and processing.
# For more information regarding what info is stored when, see `LayerReloadingInfo`
#
# Use a weak ref dictionary so that modules can be freed when the model is freed.
# Values are sanitized from references to the layer key in order to avoid circular refs
LAYERWISE_INFO: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
    WeakKeyDictionary()
)

# Global set used to track loading for logging purposes only
LOADING_LAYERS: WeakSet[torch.nn.Module] = WeakSet()
_LOADING_LOCK = threading.Lock()

# Warn once when checkpoint-format scratch held by incomplete modules exceeds
# this budget (0 disables).
RELOAD_SCRATCH_BUDGET_BYTES = int(
    float(os.getenv("VLLM_RELOAD_SCRATCH_BUDGET_MB", "0")) * 1e6
)
_SCRATCH_BUDGET_WARNED = False


@dataclass(frozen=True)
class LoadTarget:
    """The (module, tensor name) a wrapped loader is currently writing."""

    module: torch.nn.Module
    param_name: str
    # the tensor the loader was handed (meta during a trace)
    tensor: torch.Tensor | None = None


# Set around every wrapped loader call, real or traced.
_CURRENT_LOAD: ContextVar[LoadTarget | None] = ContextVar("current_load", default=None)


def current_load() -> LoadTarget | None:
    """(module, param_name) while a wrapped loader runs, else None."""
    return _CURRENT_LOAD.get()


class LoadTrace:
    """Result of `trace_loads`: loaders ran on meta, nothing was buffered,
    materialized or processed."""

    def __init__(self):
        self.copied_numel: dict[torch.nn.Module, int] = {}
        self._totals: dict[torch.nn.Module, int] = {}

    def _record(self, module: torch.nn.Module, numel: int, total: int) -> None:
        self.copied_numel[module] = self.copied_numel.get(module, 0) + numel
        self._totals[module] = total

    def complete_modules(self) -> list[torch.nn.Module]:
        """Modules whose copied numel reached their loadable size: the same rule
        the real path uses to trigger `complete_module`."""
        return [m for m, n in self.copied_numel.items() if n >= self._totals.get(m, 0)]


# Active trace, if inside `trace_loads`
_TRACE: LoadTrace | None = None


def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    if layer not in LAYERWISE_INFO:
        LAYERWISE_INFO[layer] = LayerReloadingInfo(
            restore_metadata=({}, {}),
            restore_device=torch.get_default_device(),
        )

    return LAYERWISE_INFO[layer]


def record_metadata_for_reloading(model: torch.nn.Module):
    """Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `initialize_layerwise_reload`.
    """
    for layer in model.modules():
        info = get_layerwise_info(layer)
        info.restore_metadata = capture_layer_to_meta(layer)
        info.restore_device = torch.get_default_device()
        info.init_values = _capture_init_values(layer, info)


# Small scale tensors keep their create-time value as the reload initial value:
# scales are created with deliberate values (ones, or FP8_SCALE_SENTINEL, which
# process_weights_after_loading uses to detect shards the checkpoint lacks).
# Everything else is zero-filled (weights are often created with torch.empty).
INIT_VALUE_MAX_NUMEL = 4096


def _capture_init_values(
    layer: torch.nn.Module, info: LayerReloadingInfo
) -> dict[str, torch.Tensor]:
    values = {}
    params, buffers = info.restore_metadata
    for name in (*params, *buffers):
        t = getattr(layer, name, None)
        if (
            "scale" in name
            and isinstance(t, torch.Tensor)
            and not t.is_meta
            and 0 < t.numel() <= INIT_VALUE_MAX_NUMEL
            and not isinstance(t, torch.nn.parameter.UninitializedParameter)
        ):
            try:
                values[name] = t.detach().clone()
            except (RuntimeError, NotImplementedError):
                continue
    return values


@torch.no_grad()
def initialize_layerwise_reload(
    model: torch.nn.Module,
    generation: int | None = None,
    partial: bool | None = None,
):
    """Set up layerwise weight loading with deferred processing.

    Must be called after `record_metadata_for_reloading`. This function:
    1. Saves current kernel tensors for later copying
    2. Restores layer parameters/buffers from metadata (on meta device)
    3. Wraps weight loaders to defer processing until all weights are loaded

    When all weights for a layer are loaded, the wrapped loaders will:
    1. Materialize the layer onto the target device
    2. Load all cached weights
    3. Run quantization processing if applicable
    4. Copy processed values back to original tensor storage
    """
    _check_model_hook_reload_safe(model)

    # A failed update leaves the model dirty until a full update succeeds
    previous = get_reload_session(model)
    session = ReloadSession(
        generation=generation,
        dirty=previous is not None and previous.dirty,
        partial=partial,
        required_modules=_required_modules(model),
    )
    model._reload_session = session

    if _direct.DIRECT_LOAD:
        _direct.build_storage_users(model)

    # disable torchao reloading to avoid infinite recursion
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        if info.can_load():
            continue

        info.session = session

        # Save current tensors for later copying
        info.kernel_tensors = get_layer_params_buffers(layer)
        # snapshot now: restore_layer_on_meta drops alias buffers from the live set
        info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)

        # Restore layer parameters/buffers onto meta device
        restore_layer_on_meta(layer, info)

        # Wrap weight loaders to buffer loading
        initialize_online_processing(layer)


# Reload never calls the free-form model hook `model.process_weights_after_loading()`
# (cold start only). A model that has one must declare `reload_safe = True`
# (its reload work lives in module-level PWAL / `refresh()`), else reload fails
# closed. VLLM_RELOAD_ALLOW_UNSAFE_MODEL_HOOK=1 downgrades this to a warning.
ALLOW_UNSAFE_MODEL_HOOK = os.getenv("VLLM_RELOAD_ALLOW_UNSAFE_MODEL_HOOK", "0") == "1"


class ReloadUnsafeModelError(RuntimeError):
    pass


def _check_model_hook_reload_safe(model: torch.nn.Module) -> None:
    hook = getattr(type(model), "process_weights_after_loading", None)
    if hook is None or getattr(model, "reload_safe", False):
        return
    msg = (
        f"{type(model).__name__} has a model-level process_weights_after_loading() "
        "hook, which reload does not call, and does not declare reload_safe. Its "
        "post-load work would not be redone on reload (stale or wrong weights). "
        "Set VLLM_RELOAD_ALLOW_UNSAFE_MODEL_HOOK=1 to reload anyway."
    )
    if ALLOW_UNSAFE_MODEL_HOOK:
        logger.warning_once(msg)
        return
    raise ReloadUnsafeModelError(msg)


# Integrity. A module left partially loaded at finish would keep zeros (or,
# with direct loading, bytes of its old kernel format) where data is missing.
# VLLM_RELOAD_REQUIRE_COMPLETE=1 raises; otherwise it is reported and logged.
# Direct loading always requires it for modules it hosted.
REQUIRE_COMPLETE = os.getenv("VLLM_RELOAD_REQUIRE_COMPLETE", "0") == "1"


class ReloadIncompleteError(RuntimeError):
    pass


def get_reload_session(model: torch.nn.Module) -> ReloadSession | None:
    return getattr(model, "_reload_session", None)


def is_model_dirty(model: torch.nn.Module) -> bool:
    """True after a failed update wrote live weights (until one succeeds)."""
    session = get_reload_session(model)
    return session is not None and session.dirty


def _required_modules(model: torch.nn.Module) -> dict[int, str]:
    """Modules a full update must send weights to: those that first own (in
    module order) a parameter or persistent buffer. A tensor shared by several
    modules is required once, from its first owner (a tied `lm_head` is loaded
    through `embed_tokens`). Deferred attention layers are exempt (their q/k/v
    scales are optional in checkpoints). Computed on the live tensors, before
    they are swapped for meta."""
    seen: set[int] = set()
    required = {}
    for name, module in model.named_modules():
        tensors = [
            t
            for n, t in module._parameters.items()
            if t is not None and n not in SKIP_LOAD_TENSORS
        ] + [
            t
            for n, t in module._buffers.items()
            if t is not None
            and n not in module._non_persistent_buffers_set
            and n not in SKIP_LOAD_TENSORS
        ]
        first_owned = [t for t in tensors if id(t) not in seen]
        seen.update(id(t) for t in tensors)
        if first_owned and not is_deferred_attention_layer(module):
            required[id(module)] = name
    return required


def _check_full_update(model: torch.nn.Module, session: ReloadSession) -> None:
    """Modules that received no weights: raise for a full update, report
    when unspecified, accept for a partial one."""
    if session.partial:
        return
    untouched = [
        name
        for name, layer in model.named_modules()
        if id(layer) in session.required_modules
        and (info := LAYERWISE_INFO.get(layer)) is not None
        and info.can_load()
        and info.kernel_tensors is not None
        and info.load_numel == 0
    ]
    if not untouched:
        return
    msg = (
        f"{len(untouched)} module(s) received no weights in this update: "
        f"{untouched[:8]}"
    )
    if session.partial is False:
        session.incomplete = untouched
        raise ReloadIncompleteError(
            msg + " (a full update was requested; start it with partial=True "
            "to update a subset of the model)"
        )
    logger.warning(
        "%s. They keep their previous weights; start the update with "
        "partial=True to mark this intended, or partial=False to require a "
        "full update.",
        msg,
    )


def mark_dirty(model: torch.nn.Module) -> None:
    """Mark the model's weights unusable until a full update succeeds (e.g. an
    update that succeeded here but failed on another rank)."""
    session = get_reload_session(model)
    if session is None:
        session = ReloadSession(active=False)
        model._reload_session = session
    session.dirty = True


def check_can_serve(model: torch.nn.Module) -> None:
    """Refuse to run the model mid-update or after a failed update."""
    session = getattr(model, "_reload_session", None)
    if session is None:
        return
    if session.active:
        raise RuntimeError(
            "Model is mid weight update (start_weight_update without "
            "finish_weight_update); pause generation during updates."
        )
    if session.dirty:
        raise RuntimeError(
            "A weight update failed after writing live weights; the engine "
            "refuses to serve until a full update succeeds."
        )


def _mark_dirty(info: LayerReloadingInfo) -> None:
    if info.session is not None:
        info.session.dirty = True


def _record_expert_unit(
    layer: torch.nn.Module,
    info: LayerReloadingInfo,
    param_name: str,
    original_loader,
    bound_args: inspect.BoundArguments,
) -> None:
    """Required keys for routed experts: one fused param holds every local
    expert, so a missing expert is invisible to name-based checks and a
    duplicate can make the numel count reach its total with another expert
    missing. Record each arrival per (param, shard, local expert)."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    session = info.session
    if session is None or not session.active:
        return
    if getattr(original_loader, "__func__", None) is not RoutedExperts.weight_loader:
        return
    args = bound_args.arguments
    shard_id, expert_id = args.get("shard_id"), args.get("expert_id")
    loaded = args.get("loaded_weight")
    if not isinstance(expert_id, int) or not isinstance(loaded, torch.Tensor):
        return
    stacked = loaded.dim() == 3 or (
        layer.quant_config is not None
        and layer.quant_config.get_name() == "gpt_oss_mxfp4"
    )
    if stacked:
        unit: int | str = "*"
    else:
        unit = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if unit == -1:
            return  # not local to this rank
    units = session.expert_units.setdefault(id(layer), Counter())
    units[(param_name, shard_id, unit)] += 1


def _expert_shards(layer: torch.nn.Module, param_name: str) -> tuple[str, ...]:
    if param_name.startswith("w2_"):
        return ("w2",)
    if param_name.startswith("w13_"):
        return ("w1", "w3") if layer.moe_config.is_act_and_mul else ("w1",)
    return ()


def check_expert_units(layer: torch.nn.Module, units: Counter) -> tuple[list, list]:
    """(missing, duplicated) (param, shard, local expert) units of one routed
    experts module, for every param this update touched. Input scales are
    exempt (per-tensor, reduced over experts; may be global)."""
    local = {
        layer._map_global_expert_id_to_local_expert_id(g)
        for g in range(layer.global_num_experts)
    } - {-1}
    touched = {param for param, _, _ in units if "input_scale" not in param}
    missing = []
    for param in sorted(touched):
        for shard in _expert_shards(layer, param):
            if units[(param, shard, "*")]:
                continue
            missing += [
                (param, shard, e) for e in sorted(local) if not units[(param, shard, e)]
            ]
    duplicated = sorted(k for k, n in units.items() if n > 1 and k[2] != "*")
    return missing, duplicated


def _check_expert_units(model: torch.nn.Module, session: ReloadSession) -> list[str]:
    problems = []
    for name, layer in model.named_modules():
        units = session.expert_units.get(id(layer))
        if not units:
            continue
        missing, duplicated = check_expert_units(layer, units)
        if missing:
            problems.append(
                f"{name}: {len(missing)} expert shard(s) never loaded, "
                f"e.g. {missing[:4]}"
            )
        if duplicated:
            msg = (
                f"{name}: {len(duplicated)} expert shard(s) loaded more than "
                f"once, e.g. {duplicated[:4]}"
            )
            if REQUIRE_COMPLETE:
                problems.append(msg)
            else:
                logger.warning(msg)
    session.expert_units.clear()
    return problems


def _check_complete(model: torch.nn.Module) -> None:
    incomplete: list[str] = []
    must_raise = False
    session = get_reload_session(model)
    if session is not None:
        # A missing expert holds zeros (or old bytes, with direct loading) in
        # every mode, and may sit in a module that already completed.
        _check_full_update(model, session)
        expert_problems = _check_expert_units(model, session)
        if expert_problems:
            session.incomplete = expert_problems
            raise ReloadIncompleteError(
                "Routed experts incomplete in this update: "
                + "; ".join(expert_problems[:8])
            )
    for name, layer in model.named_modules():
        info = LAYERWISE_INFO.get(layer)
        if info is None or not info.can_load() or info.kernel_tensors is None:
            continue
        if is_deferred_attention_layer(layer):
            continue
        if 0 < info.load_numel < info.load_numel_total:  # type: ignore[operator]
            # Required keys, by name: every loadable tensor must have received
            # a loader call. Counts alone can't tell a missing shard from
            # padding a loader never writes.
            restore_params, restore_buffers = info.restore_metadata
            never_loaded = sorted(
                t
                for t in (*restore_params, *restore_buffers)
                if t not in SKIP_LOAD_TENSORS
                and t not in info.kernel_non_persistent_buffers
                and t not in info.loaded_names
            )
            incomplete.append(
                f"{name or type(layer).__name__} "
                f"({info.load_numel}/{info.load_numel_total}"
                + (f", never loaded: {never_loaded}" if never_loaded else "")
                + ")"
            )
            # With direct loading a never-loaded hosted tensor holds zeros
            must_raise = must_raise or bool(
                set(never_loaded) & set(getattr(info, "hosted", {}))
            )
    session = get_reload_session(model)
    if session is not None:
        session.incomplete = incomplete
    if not incomplete:
        return
    msg = (
        f"{len(incomplete)} module(s) were only partially loaded by this update "
        f"(missing or miscounted names): {incomplete[:8]}"
    )
    if REQUIRE_COMPLETE or must_raise:
        raise ReloadIncompleteError(msg)
    logger.warning(msg)


def initialize_online_processing(layer: torch.nn.Module):
    """Wrap a layer's weight loaders with online processing loaders.
    Called by either `initialize_layerwise_reload` or an online quantization scheme,
    prevents double wrapping in the case of online quantization + reloading

    Args:
        layer: layer whose parameter weight loaders will be wrapped

    """
    info = get_layerwise_info(layer)

    # Track loading progress to determine when to process/copy
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)
    _wrap_parameters_weight_loader(layer)


def _wrap_parameters_weight_loader(layer: torch.nn.Module) -> None:
    """Wrap each parameter's weight loader."""
    # Note that nested wrapping will occur for shared tensors
    for name, tensor in get_layer_tensors(layer).items():
        if name in SKIP_LOAD_TENSORS:
            continue
        if _get_weight_loader(tensor).__name__ != "online_process_loader":
            tensor.weight_loader = make_online_process_loader(layer, name)


def make_online_process_loader(layer: torch.nn.Module, param_name: str) -> Callable:
    """Create a wrapped weight loader that defers processing."""
    info = get_layerwise_info(layer)
    param = getattr(layer, param_name)
    original_loader = _get_original_loader(param)
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def online_process_loader(*args, **kwargs):
        if not info.can_load():
            # Unfortunately, some qconfigs are set up to load the same weight
            # multiple times. For example, CT_WNA16 loads `weight_shape` for
            # each of the qkv partitions. This results in layers loading extra
            # weights (beyond load_numel_total) after it's already processed.
            #
            # Best solution is to ensure that `load_numel_total` reflects the
            # actual number of weights loaded, either by modifying qconfigs to
            # create as many weights as loaded (see padding issue as well)
            # or maybe capturing how many weights are loaded on first pass
            #
            # For now, `load_numel_total` is still safe to use as long as
            # there's no way to reach `load_numel_total` without loading all
            # necessary weights. `weight_shape` is very small, so this is safe.
            # see Limitations(4)
            logger.debug("%s: Excessive loading", layer.__class__.__name__)
            return

        # Re-run on each load: layers may register parameters later (e.g., `bias`).
        # Wrap late parameters and refresh `load_numel_total` so processing waits
        # until all parameters are loaded.
        info.load_numel_total = get_layer_size(layer)
        _wrap_parameters_weight_loader(layer)

        # Bind and normalize arguments
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        dest = bound_args.arguments.get("param")
        if _TRACE is not None and isinstance(dest, torch.Tensor) and not dest.is_meta:
            # Tensors kept live during reload (e.g. `bias`): trace against a meta
            # copy so a dry run never writes live storage, and is counted
            dest = to_meta_tensor(dest)
            bound_args.arguments["param"] = dest
        token = _CURRENT_LOAD.set(LoadTarget(layer, param_name, dest))
        try:
            if _TRACE is not None:
                # Dry run (`trace_loads`): run on meta and count, nothing else
                num_loaded, ret = get_numel_loaded(original_loader, bound_args)
                _TRACE._record(layer, num_loaded, info.load_numel_total)
                return ret

            info.loaded_names.add(param_name)
            if _should_buffer(layer, bound_args):
                # CPU/mmap incoming (cold-start online quant, file reload) or a
                # deferred attention layer: buffer the call as before and count
                # it with a dry run on the meta param.
                info.loaded_weights.append((param_name, bound_args))
                num_loaded, ret = get_numel_loaded(original_loader, bound_args)
            elif _use_per_expert(layer, info, param_name, original_loader, bound_args):
                # Per-expert completion: the loader writes one expert into a
                # checkpoint-format slot (the fused tensor stays on meta), and
                # the expert is quantized as soon as its pieces arrived.
                from vllm.model_executor.layers.fused_moe.routed_experts import (
                    expert_target_provider,
                )

                slots = info.expert_slots
                with expert_target_provider(slots.provider(param_name)):
                    num_loaded, ret = get_numel_loaded(original_loader, bound_args)
                slots.account(num_loaded)
            else:
                # First touch: materialize this param's rank-local
                # checkpoint-format target now and run the loader on it, so the
                # incoming tensor (including non-local experts, which the loader
                # declines) is not retained.
                target = ensure_param_materialized(layer, info, param_name)
                if info.kernel_tensors is not None and param_name not in (
                    info.materialized
                ):
                    _mark_dirty(info)  # writes live storage (e.g. `bias`)
                if info.loaded_weights:
                    _replay_buffered(layer, info)
                bound_args.arguments["param"] = target
                num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        finally:
            _CURRENT_LOAD.reset(token)
        info.load_numel += num_loaded
        _record_expert_unit(layer, info, param_name, original_loader, bound_args)

        logger.debug(
            "%s: %d / %d",
            layer.__class__.__name__,
            info.load_numel,
            info.load_numel_total,
        )

        # Do not online process attention layers, must wait until finalize
        if is_deferred_attention_layer(layer):
            return ret

        # Log warnings allocating excessive buffers on device
        if (
            _has_device_incoming(bound_args) or info.scratch_bytes > 0
        ) and layer not in LOADING_LAYERS:
            with _LOADING_LOCK:
                LOADING_LAYERS.add(layer)
                in_flight = list(LOADING_LAYERS)
            _warn_layers_in_flight(in_flight)

        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            complete_module(layer, info)

        return ret

    return online_process_loader


def finalize_layerwise_processing(model: torch.nn.Module, model_config: ModelConfig):
    """Apply processing to any layers which were not layerwise processed during loading.
    This includes attention layers and layers which have weight elements which are not
    loaded (due to padding).

    This function should be applied after `initialize_layerwise_reload` is applied
    unwrap the layerwise weight loaders.

    Args:
        model: model to finalize processing for
        model_config: config needed for applying processing to attention layers

    """
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload

    _check_complete(model)

    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []
    reloading = False

    for layer in model.modules():
        info = get_layerwise_info(layer)
        if not info.can_load():
            info.reset()
            continue
        reloading = reloading or info.kernel_tensors is not None

        # Deferred attention-like layers are processed after all other layers
        if is_deferred_attention_layer(layer):
            deferred_attn.append((layer, info))
            continue

        # No weights were loaded
        if info.load_numel <= 0:
            # first load: checkpoint did not contain weights for this layer
            if info.kernel_tensors is None:
                complete_module(layer, info)
                continue

            # reloading: place kernel tensors back as a fallback. Always place, even
            # when nothing is loadable (load_numel_total == 0), so parameter-alias
            # buffers on such layers are restored rather than left deleted.
            # (modules that own checkpoint tensors and received none are
            # reported by `_check_full_update`)
            if info.load_numel_total > 0:  # type: ignore[operator]
                logger.debug("%s: received no weights", layer.__class__.__name__)
            _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to excess memory usage
        # see Limitations(4)
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            complete_module(layer, info)

        info.reset()

    # Cross-module readers in the attention phase (MLA reads kv_b_proj) must
    # only see landed modules: every other module is closed by now.
    still_open = [
        layer
        for layer in model.modules()
        if not is_deferred_attention_layer(layer)
        and (layer_info := LAYERWISE_INFO.get(layer)) is not None
        and layer_info.hosted
    ]
    assert not still_open, f"modules not landed before attention: {still_open}"

    # Process attention layers after all other layers are done
    for layer, info in deferred_attn:
        reloading = reloading or info.kernel_tensors is not None
        _finalize_attention_layer(layer, info, model_config)
        info.reset()

    # Model phase: declared refresh() of model-local derived state (kind A),
    # after every module has landed. The free-form model hook is not called.
    if reloading:
        _refresh_model_local(model)

    LOADING_LAYERS.clear()

    # The update finished: the model may serve again
    session = get_reload_session(model)
    if session is not None:
        session.dirty = False
        session.active = False
        model._weights_generation = session.generation


def _refresh_model_local(model: torch.nn.Module) -> None:
    # post-order-ish: children before their parents, the model last
    for module in reversed(list(model.modules())):
        if isinstance(module, QuantizeMethodBase):
            continue  # refreshed in the quant phase, with their layer
        refresh = getattr(module, "refresh", None)
        if callable(refresh):
            with torch.no_grad():
                refresh()


def finalize_layerwise_reload(*args, **kwargs):
    finalize_layerwise_processing(*args, **kwargs)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    old_floats = _kv_scale_floats(layer)
    if info.kernel_tensors is None:
        if info.load_numel > 0:
            complete_module(layer, info)
    elif info.load_numel > 0:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info, model_config)
    else:
        _place_kernel_tensors(layer, info)
    reloading = info.kernel_tensors is not None
    if not reloading:
        layer.process_weights_after_loading(model_config.dtype)
        return
    # Attention PWAL runs on the live module (MLA derives W_UK_T / W_UV from
    # the landed kv_b_proj). Land anything it re-registers back into the
    # original tensors, so captured graphs keep reading valid storage.
    params, buffers = get_layer_params_buffers(layer)
    before = {**params, **buffers}
    with reload_mode():
        layer.process_weights_after_loading(model_config.dtype)
    # The layer PWAL refreshed backend state derived from the q/k/v scales
    # (FlashInfer bmm1/bmm2 scales, device copies filled in place). Decode
    # paths that consume host floats are baked into FULL CUDA graphs.
    impl = getattr(layer, "impl", None)
    if (
        old_floats != _kv_scale_floats(layer)
        and getattr(impl, "float_scales_in_decode", False)
        and _cudagraphs_captured()
    ):
        logger.warning_once(
            "Attention q/k/v scales changed on reload, but this attention "
            "backend's decode path (%s) reads them as host floats, which "
            "captured CUDA graphs bake in: decode uses the old scales until "
            "the graphs are re-captured.",
            type(impl).__name__,
        )
    for name, old in before.items():
        new = getattr(layer, name, None)
        if new is None or new is old:
            continue
        if _direct.same_view(new, old):
            continue
        if check_exact_landing(old, new) is None:
            old.data.copy_(new)
            if name in layer._parameters:
                layer._parameters[name] = old
            else:
                layer._buffers[name] = old
        else:
            logger.warning_once(
                "%s.%s: attention post-load processing re-allocated it with a "
                "different layout on reload; CUDA graphs may read stale data",
                type(layer).__name__,
                name,
            )


def _kv_scale_floats(layer: torch.nn.Module) -> tuple:
    return tuple(
        getattr(layer, name, None)
        for name in ("_q_scale_float", "_k_scale_float", "_v_scale_float")
    )


def _cudagraphs_captured() -> bool:
    try:
        from vllm.config import CUDAGraphMode, get_current_vllm_config

        mode = get_current_vllm_config().compilation_config.cudagraph_mode
        return mode is not None and mode != CUDAGraphMode.NONE
    except Exception:  # noqa: BLE001
        return False


def _reload_attention_scales(
    layer: torch.nn.Module,
    info: LayerReloadingInfo,
    model_config: ModelConfig | None = None,
) -> None:
    """Load and process attention scale weights (k_scale, v_scale, etc.)
    during reload.

    Assumes dtype/shapes of attention tensors do not change during
    processing, since we use .data.copy_() to preserve kernel tensor
    references."""
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None:
        # Re-create scale Parameters with sentinel values so unloaded scales
        # are correctly detected by process_weights_after_loading. Under the
        # model dtype, as at model init: the scales are 0-dim tensors of the
        # default dtype, so an fp32 checkpoint scale is rounded the same way
        # on reload as on a fresh load.
        from vllm.utils.torch_utils import set_default_torch_dtype

        dtype = model_config.dtype if model_config is not None else None
        if isinstance(dtype, torch.dtype):
            with set_default_torch_dtype(dtype):
                quant_method.create_weights(layer)
        else:
            quant_method.create_weights(layer)

    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        _get_weight_loader(param)(*args.args, **args.kwargs)

    if quant_method is not None:
        quant_method.process_weights_after_loading(layer)

    _copy_and_restore_kernel_tensors(layer, info)


def _should_buffer(layer: torch.nn.Module, bound_args: inspect.BoundArguments) -> bool:
    """Buffer-or-run policy. Deferred attention layers and CPU/mmap incoming
    tensors keep today's buffering: buffering CPU tensors costs no device memory,
    while first-touch there would allocate device scratch for every module a
    checkpoint shard touches. Device incoming tensors (NCCL/IPC reload) run
    the loader immediately."""
    if is_deferred_attention_layer(layer):
        return True
    return not _has_device_incoming(bound_args)


def _has_device_incoming(bound_args: inspect.BoundArguments) -> bool:
    """Like `has_device_tensors`, but ignores the destination `param`, which is
    a device tensor once materialized even if the incoming data is on CPU."""
    return any(
        isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
        for name, value in bound_args.arguments.items()
        if name != "param"
    )


def ensure_param_materialized(
    layer: torch.nn.Module, info: LayerReloadingInfo, name: str
) -> torch.Tensor:
    """Return the load target for one tensor of `layer`, materializing it
    (zero-filled) on first touch. Per tensor, not per module: some params are
    registered after a module's first loader call (e.g. `bias`)."""
    tensor = getattr(layer, name)
    if not tensor.is_meta or name in SKIP_TENSORS:
        return tensor
    target = None
    if (
        _direct.DIRECT_LOAD
        and info.kernel_tensors is not None
        # never host a non-persistent buffer: unless a loader writes it, landing
        # skips it, so zero-filling its live bytes would destroy its value
        and name not in info.kernel_non_persistent_buffers
    ):
        # Check 1: can the live tensor's own bytes hold the checkpoint tensor?
        params, buffers = info.kernel_tensors
        live = params.get(name, buffers.get(name))
        reason = _direct.plan_input(tensor, live, info.restore_device)
        plan = _direct.PLAN_OUTCOMES.setdefault(layer, {})
        if reason is None:
            assert live is not None
            target = _direct.checkpoint_view(live, tensor)
            storage = live.untyped_storage()
            info.hosted[name] = (storage.data_ptr(), storage.nbytes())
            plan[name] = "hosted"
            _mark_dirty(info)  # live bytes now hold checkpoint data
        else:
            plan[name] = f"scratch:{reason}"
    if target is None:
        with info.restore_device:
            target = _meta.materialize_meta_tensor(tensor)
        info.scratch_bytes += target.nbytes
    # Zero-filled, hosted or not: loaders don't write padding, and a slice no
    # loader writes must not read as plausible old kernel-format bytes. Small
    # tensors start from their create-time value instead (scale sentinels).
    init = info.init_values.get(name)
    if init is not None and init.shape == target.shape and init.dtype == target.dtype:
        target.data.copy_(init)
    else:
        target.data.zero_()
    setattr(layer, name, target)
    info.materialized.add(name)
    return target


def _materialize_all(
    layer: torch.nn.Module, info: LayerReloadingInfo, skip: tuple[str, ...] = ()
) -> None:
    if layer.__class__.__name__ in SKIP_MODULES:
        return
    for name, tensor in get_layer_tensors(layer).items():
        if name not in SKIP_TENSORS and name not in skip and tensor.is_meta:
            ensure_param_materialized(layer, info, name)


def _replay_buffered(
    layer: torch.nn.Module, info: LayerReloadingInfo, skip: tuple[str, ...] = ()
) -> None:
    """Replay buffered loader calls into materialized targets (already counted).
    Only needed when a module saw buffered calls (CPU incoming) before
    running ones, or at completion of a buffering module."""
    _materialize_all(layer, info, skip)
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        _get_original_loader(param)(*args.args, **args.kwargs)
    info.loaded_weights.clear()


def _warn_layers_in_flight(in_flight: list[torch.nn.Module]) -> None:
    if len(in_flight) < 2:
        return
    buffered = sum(get_info_size(LAYERWISE_INFO[layer]) for layer in in_flight)
    scratch = sum(LAYERWISE_INFO[layer].scratch_bytes for layer in in_flight)
    if len(in_flight) == 2:
        names = sorted(layer.__class__.__name__ for layer in in_flight)
        logger.warning_once(
            "Allocating %.1f MB of device memory to buffers and %.1f MB to "
            "checkpoint-format scratch to load %s layers. This extra memory usage "
            "can be avoided by ordering weights by their parent layer when "
            "reloading.",
            buffered / 1e6,
            scratch / 1e6,
            str(list(names)),
        )
    if RELOAD_SCRATCH_BUDGET_BYTES and scratch > RELOAD_SCRATCH_BUDGET_BYTES:
        global _SCRATCH_BUDGET_WARNED
        if not _SCRATCH_BUDGET_WARNED:
            _SCRATCH_BUDGET_WARNED = True
            logger.warning(
                "Reload scratch in flight (%.1f MB across %d incomplete modules) "
                "exceeds VLLM_RELOAD_SCRATCH_BUDGET_MB. Incomplete modules: %s",
                scratch / 1e6,
                len(in_flight),
                sorted({layer.__class__.__name__ for layer in in_flight}),
            )


def scratch_bytes_in_flight() -> int:
    """Checkpoint-format scratch currently held by incomplete modules."""
    with _LOADING_LOCK:
        in_flight = list(LOADING_LAYERS)
    return sum(LAYERWISE_INFO[layer].scratch_bytes for layer in in_flight)


def _warn_if_not_reload_safe(layer: torch.nn.Module, quant_method) -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    if isinstance(quant_method, FusedMoEMethodBase) and not quant_method.reload_safe:
        logger.warning_once(
            "%s (%s) does not declare reload_safe: its MoE kernel is rebuilt on "
            "reload, so weight-derived state held off-registry may be stale under "
            "CUDA graphs.",
            type(quant_method).__name__,
            getattr(quant_method, "fp8_backend", None)
            or getattr(quant_method, "mxfp4_backend", None)
            or "",
        )


def _refresh_quant_method(layer: torch.nn.Module, quant_method) -> None:
    if getattr(quant_method, "reload_safe", False) and hasattr(quant_method, "refresh"):
        quant_method.refresh(layer)


def add_reload_transform(
    module: torch.nn.Module, transform: Callable[[torch.nn.Module], None]
) -> None:
    """Register an output transform on `module`: on reload it runs on the
    module's PWAL results, before they land in the live tensors. At cold start
    the owner runs the same function where it does today."""
    transforms = module.__dict__.setdefault("_reload_output_transforms", [])
    transforms.append(transform)


def _use_per_expert(
    layer: torch.nn.Module,
    info: LayerReloadingInfo,
    param_name: str,
    original_loader: Callable,
    bound_args: inspect.BoundArguments,
) -> bool:
    if param_name not in _per_expert.EXPERT_WEIGHTS:
        return False
    if info.per_expert is None:
        info.per_expert = _per_expert.eligible(layer, original_loader, bound_args)
        if info.per_expert:
            info.expert_slots = _per_expert.ExpertSlots(layer, info)
    elif info.per_expert:
        loaded = bound_args.arguments.get("loaded_weight")
        if not (isinstance(loaded, torch.Tensor) and loaded.dim() == 2):
            raise NotImplementedError(
                f"{type(layer).__name__}: a stacked expert load after per-expert "
                "loads in the same update (set VLLM_RELOAD_PER_EXPERT=0)"
            )
    return bool(info.per_expert)


def _has_module_pwal(layer: torch.nn.Module) -> bool:
    """A plain module with its own zero-argument transform (not attention,
    whose PWAL takes the activation dtype and is deferred to finalize)."""
    return (
        not is_deferred_attention_layer(layer)
        and callable(getattr(layer, "process_weights_after_loading", None))
        and bool(getattr(layer, "reload_outputs", ()))
    )


def complete_module(layer: torch.nn.Module, info: LayerReloadingInfo | None = None):
    """Finish one module: PWAL on its checkpoint-format tensors, copy the
    results into the live tensors, restore the original tensor objects, reset.

    Called by the wrapped loader when a module's count reaches its total, by
    finalize for modules that never reached it, and by engines that route
    bytes themselves. It does not require the count to have reached its total.
    """
    if info is None:
        info = get_layerwise_info(layer)

    slots = info.expert_slots
    if slots is not None:
        # Per-expert completion: quantize every unit still open; the fused
        # checkpoint-format expert weights are never materialized
        slots.flush()
        info.expert_slots = None

    # Materialize anything not yet touched (zero-filled) and replay buffered
    # calls, if this module took the buffering path
    _replay_buffered(layer, info, skip=_per_expert.EXPERT_WEIGHTS if slots else ())

    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Unwrap layerwise loading wrappers
    for param in get_layer_tensors(layer).values():
        param.weight_loader = _get_original_loader(param)

    # Process weights (quantization, repacking, etc.). On reload this runs in
    # reload mode: kernel objects are built once, `replace_parameter` rebinds.
    reloading = info.kernel_tensors is not None
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        if reloading:
            _warn_if_not_reload_safe(layer, quant_method)
        with reload_mode() if reloading else nullcontext():
            if slots is not None:
                quant_method.finish_experts(layer, slots.staging)  # type: ignore[attr-defined]
                layer._already_called_process_weights_after_loading = True
            else:
                quant_method.process_weights_after_loading(layer)
        # Re-reconcile parameter TP state: process_weights_after_loading may
        # have re-created Parameters (stamped with the global rank), which would
        # otherwise break replicated (disable_tp) weights on a subsequent reload.
        if hasattr(layer, "update_param_tp_status"):
            layer.update_param_tp_status()
    elif reloading and _has_module_pwal(layer):
        # Module-level PWAL for modules without a quant method (model-local
        # transforms, e.g. mega-MoE): runs per module, on the fresh
        # checkpoint-format tensors, writing its declared `reload_outputs`.
        with reload_mode():
            layer.process_weights_after_loading()

    # Output transforms (kind B hooks, e.g. a parent's layout permute of this
    # module's weights): applied to the PWAL results before landing, so the
    # live tensors receive the final layout exactly once.
    if reloading:
        for transform in getattr(layer, "_reload_output_transforms", ()):
            with reload_mode(), torch.no_grad():
                transform(layer)

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    # this code is a no-op if not reloading (because kernel tensors is empty)
    if reloading:
        _copy_and_restore_kernel_tensors(layer, info)
        # Refresh weight-derived state now that the live tensors hold the new
        # values (quant phase; attention and model phases run in finalize)
        _refresh_quant_method(layer, quant_method)

    info.reset()
    with _LOADING_LOCK:
        LOADING_LAYERS.discard(layer)
    logger.debug("%s: Processed", layer.__class__.__name__)


# ---------------------------------------------------------------------------
# Public per-module reload API
# ---------------------------------------------------------------------------


def start_reload(
    model: torch.nn.Module,
    generation: int | None = None,
    partial: bool | None = None,
) -> None:
    """Begin a streaming reload (alias of `initialize_layerwise_reload`).

    `partial=False` requires every module with checkpoint tensors to receive
    weights; `partial=True` allows updating a subset; None reports modules
    that received nothing."""
    initialize_layerwise_reload(model, generation=generation, partial=partial)


def finish_reload(model: torch.nn.Module, model_config: ModelConfig) -> None:
    """Finish a streaming reload (alias of `finalize_layerwise_reload`)."""
    finalize_layerwise_reload(model, model_config)


def abort_reload(model: torch.nn.Module) -> None:
    """Put the original tensors back, reset reload state and unwrap loaders.

    Values are intact only for modules that were not completed yet (and, with
    direct loading, not hosted); see the dirty flag for the failure contract.
    """
    for layer in model.modules():
        info = LAYERWISE_INFO.get(layer)
        if info is None or not info.can_load():
            continue
        if info.kernel_tensors is not None:
            _place_kernel_tensors(layer, info)
        for tensor in get_layer_tensors(layer).values():
            loader = getattr(tensor, "weight_loader", None)
            if loader is not None and loader.__name__ == "online_process_loader":
                tensor.weight_loader = _get_original_loader(tensor)
        info.reset()
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload
    with _LOADING_LOCK:
        LOADING_LAYERS.clear()
    session = get_reload_session(model)
    if session is not None:
        session.active = False


@contextmanager
def trace_loads(model: torch.nn.Module) -> Iterator[LoadTrace]:
    """Dry run: loaders run on meta, nothing is buffered, materialized or
    processed. `start_reload` on enter, `abort_reload` on exit."""
    global _TRACE
    trace = LoadTrace()
    start_reload(model)
    _TRACE = trace
    try:
        yield trace
    finally:
        _TRACE = None
        abort_reload(model)


def ensure_materialized(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """First touch for engines that route bytes themselves: materialize every
    tensor of `module` and return its load targets by name. Touches only this
    module's state (safe to call from a scatter thread)."""
    info = LAYERWISE_INFO.get(module)
    if info is None or not info.can_load():
        raise RuntimeError(
            f"{type(module).__name__} was not set up for reload "
            "(start_reload / start_weight_update must run first)"
        )
    _materialize_all(module, info)
    return get_layer_tensors(module)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Return the weight loader with any layerwise wrappers removed."""
    loader = _get_weight_loader(tensor)
    while loader.__name__ == "online_process_loader":
        loader = loader.__wrapped__  # type: ignore[union-attr]

    return loader


def _get_weight_loader(tensor: torch.Tensor):
    return getattr(tensor, "weight_loader", default_weight_loader)


def _skip_landing(
    layer: torch.nn.Module, info: LayerReloadingInfo, name: str, is_buffer: bool
) -> bool:
    """Copy-back rules, shared by every landing: skip buffers that are no
    longer registered, and non-persistent buffers no loader wrote (#44371),
    unless a module-level PWAL declares them as outputs."""
    if not is_buffer:
        return False
    declared = name in getattr(layer, "reload_outputs", ())
    if name not in layer._buffers or layer._buffers[name] is None:
        if declared:
            raise RuntimeError(
                f"{type(layer).__name__}: declared reload output {name!r} was "
                "not produced by its process_weights_after_loading()"
            )
        return True
    loaded = info.loaded_names | {n for n, _ in info.loaded_weights}
    return (
        name in info.kernel_non_persistent_buffers
        and name not in loaded
        and not declared
    )


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Land PWAL results in the original kernel tensors and restore those tensor
    objects on the layer (preserves cudagraph references).

    Check 2, per tensor: the result already is the live tensor ("in place"),
    or it is copied with the exact-landing check; a result that shares storage
    with live storage is cloned first ("overlap"). All sources are resolved
    before the first copy, so no copy overwrites bytes a later one reads."""
    assert info.kernel_tensors is not None
    _mark_dirty(info)
    parameters, buffers = info.kernel_tensors
    live_storages = {
        t.untyped_storage().data_ptr()
        for t in (*parameters.values(), *buffers.values())
        if t.numel() > 0
    }
    outcomes = _direct.LANDING_OUTCOMES.setdefault(layer, {})
    pending: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for is_buffer, tensors in ((False, parameters), (True, buffers)):
        for name, live in tensors.items():
            if _skip_landing(layer, info, name, is_buffer):
                continue
            result = getattr(layer, name, None)
            if result is None:
                # Derived tensors (e.g. g1_alphas) a build-once PWAL no longer
                # produces keep their storage and are rewritten by refresh()
                outcomes[name] = "not_produced"
                continue
            if _direct.same_view(result, live):
                outcomes[name] = "in_place"
            elif _direct.shares_storage(result, live_storages):
                pending.append((name, live, result.clone()))
                outcomes[name] = "overlap"
            else:
                pending.append((name, live, result))
                outcomes[name] = "copied"
    for name, live, result in pending:
        _land(layer, name, live, result)

    # Hosting must never have grown or moved a live storage
    for name, (ptr, nbytes) in info.hosted.items():
        live = parameters.get(name, buffers.get(name))
        assert live is not None
        storage = live.untyped_storage()
        if storage.data_ptr() != ptr or storage.nbytes() != nbytes:
            raise RuntimeError(
                f"{type(layer).__name__}.{name}: hosted live storage moved "
                f"({ptr:#x}/{nbytes} -> {storage.data_ptr():#x}/{storage.nbytes()})"
            )

    _place_kernel_tensors(layer, info)


# Landing: strict by default in log-only mode; VLLM_RELOAD_STRICT_LANDING=1 raises
STRICT_LANDING_RAISE = os.getenv("VLLM_RELOAD_STRICT_LANDING", "0") == "1"


class LandingMismatchError(RuntimeError):
    pass


def check_exact_landing(dst: torch.Tensor, src: torch.Tensor) -> str | None:
    """Return a description of why `src` does not land exactly in `dst`
    (shape, stride, dtype or device differ), or None. `copy_` would silently
    broadcast or convert in those cases."""
    problems = []
    if tuple(dst.shape) != tuple(src.shape):
        problems.append(f"shape {tuple(src.shape)} -> {tuple(dst.shape)}")
    elif dst.numel() > 1 and _effective_strides(dst) != _effective_strides(src):
        problems.append(f"stride {src.stride()} -> {dst.stride()}")
    if dst.dtype != src.dtype:
        problems.append(f"dtype {src.dtype} -> {dst.dtype}")
    if dst.device != src.device:
        problems.append(f"device {src.device} -> {dst.device}")
    return ", ".join(problems) or None


def _effective_strides(t: torch.Tensor) -> tuple[int, ...]:
    # strides of size-1 dims carry no layout information
    return tuple(s for s, n in zip(t.stride(), t.shape) if n != 1)


def _land(
    layer: torch.nn.Module, name: str, dst: torch.Tensor, src: torch.Tensor
) -> None:
    problem = check_exact_landing(dst, src)
    if problem is not None:
        msg = f"reload landing {type(layer).__name__}.{name}: {problem}"
        if STRICT_LANDING_RAISE:
            raise LandingMismatchError(msg)
        logger.warning_once("Inexact %s (copy_ would broadcast/convert)", msg)
    dst.data.copy_(src)


def _place_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    for name in get_layer_tensors(layer):
        delattr(layer, name)

    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    for name, param in parameters.items():
        layer.register_parameter(name, param)
    for name, buffer in buffers.items():
        layer.register_buffer(name, buffer, persistent=name not in non_persistent)
