# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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

from . import meta as _meta
from .meta import (
    SKIP_LOAD_TENSORS,
    SKIP_MODULES,
    SKIP_TENSORS,
    capture_layer_to_meta,
    get_numel_loaded,
    restore_layer_on_meta,
    to_meta_tensor,
)
from .types import LayerReloadingInfo
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


@torch.no_grad()
def initialize_layerwise_reload(model: torch.nn.Module):
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
    # disable torchao reloading to avoid infinite recursion
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        if info.can_load():
            continue

        # Save current tensors for later copying
        info.kernel_tensors = get_layer_params_buffers(layer)
        # snapshot now: restore_layer_on_meta drops alias buffers from the live set
        info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)

        # Restore layer parameters/buffers onto meta device
        restore_layer_on_meta(layer, info)

        # Wrap weight loaders to buffer loading
        initialize_online_processing(layer)


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
            else:
                # First touch: materialize this param's rank-local
                # checkpoint-format target now and run the loader on it, so the
                # incoming tensor (including non-local experts, which the loader
                # declines) is not retained.
                target = ensure_param_materialized(layer, info, param_name)
                if info.loaded_weights:
                    _replay_buffered(layer, info)
                bound_args.arguments["param"] = target
                num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        finally:
            _CURRENT_LOAD.reset(token)
        info.load_numel += num_loaded

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

    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []

    for layer in model.modules():
        info = get_layerwise_info(layer)
        if not info.can_load():
            info.reset()
            continue

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
            if info.load_numel_total > 0:  # type: ignore[operator]
                logger.warning("%s: Failed to load weights", layer.__class__.__name__)
            _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to excess memory usage
        # see Limitations(4)
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            complete_module(layer, info)

        info.reset()

    # Process attention layers after all other layers are done
    for layer, info in deferred_attn:
        _finalize_attention_layer(layer, info, model_config)
        info.reset()

    LOADING_LAYERS.clear()


def finalize_layerwise_reload(*args, **kwargs):
    finalize_layerwise_processing(*args, **kwargs)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    if info.kernel_tensors is None:
        if info.load_numel > 0:
            complete_module(layer, info)
    elif info.load_numel > 0:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info)
    else:
        _place_kernel_tensors(layer, info)
    layer.process_weights_after_loading(model_config.dtype)


def _reload_attention_scales(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Load and process attention scale weights (k_scale, v_scale, etc.)
    during reload.

    Assumes dtype/shapes of attention tensors do not change during
    processing, since we use .data.copy_() to preserve kernel tensor
    references."""
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None:
        # Re-create scale Parameters with sentinel values so unloaded scales
        # are correctly detected by process_weights_after_loading
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
    with info.restore_device:
        target = _meta.materialize_meta_tensor(tensor)
        target.data.zero_()
    setattr(layer, name, target)
    info.materialized.add(name)
    info.scratch_bytes += target.nbytes
    return target


def _materialize_all(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    if layer.__class__.__name__ in SKIP_MODULES:
        return
    for name, tensor in get_layer_tensors(layer).items():
        if name not in SKIP_TENSORS and tensor.is_meta:
            ensure_param_materialized(layer, info, name)


def _replay_buffered(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Replay buffered loader calls into materialized targets (already counted).
    Only needed when a module saw buffered calls (CPU incoming) before
    running ones, or at completion of a buffering module."""
    _materialize_all(layer, info)
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


def complete_module(layer: torch.nn.Module, info: LayerReloadingInfo | None = None):
    """Finish one module: PWAL on its checkpoint-format tensors, copy the
    results into the live tensors, restore the original tensor objects, reset.

    Called by the wrapped loader when a module's count reaches its total, by
    finalize for modules that never reached it, and by engines that route
    bytes themselves. It does not require the count to have reached its total.
    """
    if info is None:
        info = get_layerwise_info(layer)

    # Materialize anything not yet touched (zero-filled) and replay buffered
    # calls, if this module took the buffering path
    _replay_buffered(layer, info)

    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Unwrap layerwise loading wrappers
    for param in get_layer_tensors(layer).values():
        param.weight_loader = _get_original_loader(param)

    # Process weights (quantization, repacking, etc.)
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)
        # Re-reconcile parameter TP state: process_weights_after_loading may
        # have re-created Parameters (stamped with the global rank), which would
        # otherwise break replicated (disable_tp) weights on a subsequent reload.
        if hasattr(layer, "update_param_tp_status"):
            layer.update_param_tp_status()

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    # this code is a no-op if not reloading (because kernel tensors is empty)
    if info.kernel_tensors is not None:
        _copy_and_restore_kernel_tensors(layer, info)

    info.reset()
    with _LOADING_LOCK:
        LOADING_LAYERS.discard(layer)
    logger.debug("%s: Processed", layer.__class__.__name__)


# ---------------------------------------------------------------------------
# Public per-module reload API
# ---------------------------------------------------------------------------


def start_reload(model: torch.nn.Module) -> None:
    """Begin a streaming reload (alias of `initialize_layerwise_reload`)."""
    initialize_layerwise_reload(model)


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


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Copy processed values into original kernel tensor storage and restore
    kernel tensor references on the layer. Preserves cudagraph references."""
    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    loaded_tensor_names = info.loaded_names | {name for name, _ in info.loaded_weights}
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        if name in non_persistent and name not in loaded_tensor_names:
            continue
        buffer.data.copy_(getattr(layer, name))

    _place_kernel_tensors(layer, info)


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
