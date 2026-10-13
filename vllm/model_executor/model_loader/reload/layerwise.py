# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

import torch

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .meta import (
    SKIP_LOAD_TENSORS,
    capture_init_fill,
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    restore_layer_on_meta,
)
from .refresh import refresh_quant_method, run_model_refresh
from .types import LayerReloadingInfo
from .utils import (
    get_info_size,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
    has_device_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "refresh_derived_state",
    "ReloadUnsafeModelError",
    "ReloadUnsupportedError",
    "LandingMismatchError",
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
        info.init_fill = capture_init_fill(layer)
    # module structure at init, for the verifier (deleted submodules)
    model.__dict__["_reload_module_names"] = [n for n, _ in model.named_modules()]


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
    # Before any live write: what this update cannot apply correctly
    _check_live_tensors_unchanged(model)
    _check_model_hook_reload_safe(model)
    _check_moe_methods_reload_safe(model)

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
            # see Limitations (3) in __init__.py
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

        # Buffer loaded weights, track loading progress
        info.loaded_weights.append((param_name, bound_args))
        num_loaded, ret = get_numel_loaded(original_loader, bound_args)
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
        if has_device_tensors(bound_args) and layer not in LOADING_LAYERS:
            LOADING_LAYERS.add(layer)
            if len(LOADING_LAYERS) == 2:
                names = sorted([layer.__class__.__name__ for layer in LOADING_LAYERS])
                mem_used = sum(
                    get_info_size(LAYERWISE_INFO[layer]) for layer in LOADING_LAYERS
                )
                logger.warning_once(
                    "Allocating %.1f MB of device memory to buffers to load %s layers. "
                    "This extra memory usage can be avoided by ordering weights "
                    "by their parent layer when reloading.",
                    mem_used / 1e6,
                    str(list(names)),
                )

        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)

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
                _layerwise_process(layer, info)
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
        # see Limitations (3) in __init__.py
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_process(layer, info)

        info.reset()

    reloading = reloading or any(
        info.kernel_tensors is not None for _, info in deferred_attn
    )

    # Kernel refresh, every reload-safe quant method, touched or not: the
    # built-once kernels hold the live tensors, which every module has landed
    # in by now. Before attention, in case a backend reads kernel-derived values.
    if reloading:
        for layer in model.modules():
            refresh_quant_method(layer, getattr(layer, "quant_method", None))

    # Process attention layers after all other layers are done
    for layer, info in deferred_attn:
        _finalize_attention_layer(layer, info, model_config)
        info.reset()

    # Layer and model refreshes, after every module has landed. The free-form
    # model hook is not called (a cold start runs both from
    # `process_weights_after_loading`).
    if reloading:
        run_model_refresh(model)
        _record_live_signatures(model)

    LOADING_LAYERS.clear()


def finalize_layerwise_reload(*args, **kwargs):
    finalize_layerwise_processing(*args, **kwargs)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    old_floats = _kv_scale_floats(layer)
    reloading = info.kernel_tensors is not None
    if not reloading:
        if info.load_numel > 0:
            _layerwise_process(layer, info)
    elif info.load_numel > 0:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info, model_config)
    else:
        _place_kernel_tensors(layer, info)
    if not reloading:
        layer.process_weights_after_loading(model_config.dtype)
        return
    _rerun_attention_pwal(layer, model_config.dtype, old_floats)


def _rerun_attention_pwal(
    layer: torch.nn.Module, act_dtype: torch.dtype, old_floats: tuple
) -> None:
    """Attention PWAL on the live module (MLA derives W_UK_T / W_UV from the
    landed kv_b_proj). Land anything it re-registers back into the original
    tensors, so captured graphs keep reading valid storage."""
    params, buffers = get_layer_params_buffers(layer)
    before = {**params, **buffers}
    layer.process_weights_after_loading(act_dtype)
    # The layer PWAL refreshed backend state derived from the q/k/v scales
    # (FlashInfer bmm1/bmm2 scales, device copies filled in place). Decode
    # paths that consume host floats are baked into captured CUDA graphs.
    impl = getattr(layer, "impl", None)
    if (
        old_floats != _kv_scale_floats(layer)
        and getattr(impl, "float_scales_in_decode", False)
        and _cudagraphs_captured()
    ):
        raise ReloadUnsupportedError(
            f"{type(layer).__name__}: attention q/k/v scales changed in this "
            f"update, but the decode path of {type(impl).__name__} reads them "
            "as host floats, which captured CUDA graphs bake in (FlashInfer "
            "native decode wrapper, DCP decode, or attention+output-quant "
            "fusion). Decode would keep the old scales. Keep KV-cache scales "
            "fixed across updates, or run with enforce_eager."
        )
    for name, old in before.items():
        new = getattr(layer, name, None)
        if new is None or new is old or _same_view(new, old):
            continue
        if check_exact_landing(old, new) is None:
            old.data.copy_(new)
            if name in layer._parameters:
                layer._parameters[name] = old  # type: ignore[assignment]
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
        # On the model's device, as at model init: the results land in the
        # live scales (reload runs in an RPC thread whose default is the CPU)
        with info.restore_device:
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
        from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        timing_context = (
            online_quantization_timing_context(layer, quant_method)
            if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
            else nullcontext()
        )
        with timing_context:
            quant_method.process_weights_after_loading(layer)

    _copy_and_restore_kernel_tensors(layer, info)


def _layerwise_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Finalize layer loading after all weights have been buffered.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all buffered weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Materialize layer tensors onto device
    materialize_layer(layer, info)

    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Unwrap layerwise loading wrappers
    for param in get_layer_tensors(layer).values():
        param.weight_loader = _get_original_loader(param)

    # Load all buffered weights into materialized layer (using original loaders)
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        param.weight_loader(*args.args, **args.kwargs)

    # Process weights (quantization, repacking, etc.)
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        timing_context = (
            online_quantization_timing_context(layer, quant_method)
            if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
            else nullcontext()
        )
        with timing_context:
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
    logger.debug("%s: Processed", layer.__class__.__name__)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Return the weight loader with any layerwise wrappers removed."""
    loader = _get_weight_loader(tensor)
    while loader.__name__ == "online_process_loader":
        loader = loader.__wrapped__  # type: ignore[union-attr]

    return loader


def _get_weight_loader(tensor: torch.Tensor):
    return getattr(tensor, "weight_loader", default_weight_loader)


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Land PWAL results in the original kernel tensors and restore those
    tensor objects on the layer (preserves cudagraph references).

    Per tensor: the result already is the live tensor, or it is copied with the
    exact-landing check; a result sharing storage with a live tensor is cloned
    first. All sources are resolved before the first copy, so no copy
    overwrites bytes a later one reads."""
    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    loaded_tensor_names = {name for name, _ in info.loaded_weights}
    live_storages = {
        _storage_ptr(t) for t in (*parameters.values(), *buffers.values())
    } - {None}
    pending: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for is_buffer, tensors in ((False, parameters), (True, buffers)):
        for name, live in tensors.items():
            if is_buffer and (
                layer._buffers.get(name) is None
                or (name in non_persistent and name not in loaded_tensor_names)
            ):
                continue
            result = getattr(layer, name, None)
            if result is None:
                # Derived tensors a build-once PWAL no longer produces keep
                # their storage and are rewritten by the kernel refresh
                continue
            if _same_view(result, live):
                continue
            if result.numel() > 0 and _storage_ptr(result) in live_storages:
                result = result.clone()
            pending.append((name, live, result))
    for name, live, result in pending:
        _land(layer, name, live, result)

    _place_kernel_tensors(layer, info)


# Landing: a result whose shape, dtype or device differs from the live tensor
# raises (copy_ would broadcast or convert); VLLM_RELOAD_STRICT_LANDING=0
# downgrades it to a warning. A stride-only difference always warns: copy_
# still writes the right values, in the live tensor's layout.
STRICT_LANDING_RAISE = envs.VLLM_RELOAD_STRICT_LANDING


class LandingMismatchError(RuntimeError):
    pass


def check_exact_landing(dst: torch.Tensor, src: torch.Tensor) -> str | None:
    """Return why `src` does not land exactly in `dst` (shape, stride, dtype or
    device differ), or None. `copy_` would silently broadcast or convert."""
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
        layout_only = (
            dst.shape == src.shape
            and dst.dtype == src.dtype
            and dst.device == src.device
        )
        if STRICT_LANDING_RAISE and not layout_only:
            raise LandingMismatchError(msg)
        logger.warning_once("Inexact %s", msg)
    dst.data.copy_(src)


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


def _storage_ptr(t: torch.Tensor) -> int | None:
    if t.numel() == 0 or t.is_meta:
        return None
    try:
        return t.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return None


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


# ---------------------------------------------------------------------------
# Checks before any live write
# ---------------------------------------------------------------------------


class ReloadUnsafeModelError(RuntimeError):
    """The model or one of its layers cannot be updated correctly in place."""


class ReloadUnsupportedError(RuntimeError):
    """An update this configuration cannot apply correctly."""


# Reload never calls the free-form model hook `model.process_weights_after_loading()`
# (cold start only). A model that has one should implement `SupportsReload`
# (its reload work lives in loaders, quant-method PWAL and `refresh_model()`).
# Others warn loudly on every update; VLLM_RELOAD_STRICT_MODEL_HOOK=1 raises.
STRICT_MODEL_HOOK = envs.VLLM_RELOAD_STRICT_MODEL_HOOK


def _check_model_hook_reload_safe(model: torch.nn.Module) -> None:
    hook = getattr(type(model), "process_weights_after_loading", None)
    if hook is None or getattr(model, "supports_reload", False):
        return
    msg = (
        f"{type(model).__name__} has a model-level process_weights_after_loading() "
        "hook, which reload does not call, and does not implement SupportsReload. "
        "Its post-load work is NOT redone by this weight update, so the model may "
        "serve stale or wrong weights. Move it to the weight loaders, quant-method "
        "PWAL (wrap_quant_method), refresh_model() and post_load_setup(), and "
        "implement SupportsReload; VLLM_RELOAD_STRICT_MODEL_HOOK=1 makes this an "
        "error."
    )
    if STRICT_MODEL_HOOK:
        raise ReloadUnsafeModelError(msg)
    logger.warning("UNSAFE WEIGHT UPDATE: %s", msg)


# MoE methods that don't declare `reload_safe` rebuild their kernel on reload.
# Captured CUDA graphs keep reading the old kernel's weight-derived state, so
# under graphs reload fails closed for them (before any live write).
# VLLM_RELOAD_ALLOW_UNSAFE_MOE=1 downgrades this to a warning.
ALLOW_UNSAFE_MOE = envs.VLLM_RELOAD_ALLOW_UNSAFE_MOE


def _check_moe_methods_reload_safe(model: torch.nn.Module) -> None:
    unsafe = unsafe_moe_methods(model)
    if not unsafe or not _cudagraphs_captured():
        return  # eager: a rebuilt kernel is read directly, nothing goes stale
    found = ", ".join(f"{label} (e.g. {name})" for label, name in unsafe.items())
    msg = (
        f"MoE methods that don't declare reload_safe: {found}. Reload would "
        "rebuild their kernel, and captured CUDA graphs would keep reading the "
        "old kernel's weight-derived state. To make a method reload-safe, let "
        "its PWAL always transform but build the kernel once (if "
        "self.moe_kernel is None), keep derived values in refresh(), and set "
        "reload_safe (a method with no derived state only sets it); "
        "Fp8MoEMethod and Fp8MoEProcessingPlan are the reference. Or run with "
        "enforce_eager, or set VLLM_RELOAD_ALLOW_UNSAFE_MOE=1 to reload anyway."
    )
    if ALLOW_UNSAFE_MOE:
        logger.warning_once(msg)
        return
    raise ReloadUnsafeModelError(msg)


def unsafe_moe_methods(model: torch.nn.Module) -> dict[str, str]:
    """MoE methods (labelled with their backend) that built a kernel without
    declaring `reload_safe`, each with the first module using it."""
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    unsafe: dict[str, str] = {}
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if (
            isinstance(method, FusedMoEMethodBase)
            and not method.reload_safe
            and getattr(method, "moe_kernel", None) is not None
        ):
            unsafe.setdefault(_moe_method_label(method), name)
    return unsafe


def _moe_method_label(method) -> str:
    backend = next(
        (
            getattr(method, attr)
            for attr in ("fp8_backend", "mxfp4_backend", "nvfp4_backend")
            if getattr(method, attr, None) is not None
        ),
        None,
    )
    name = type(method).__name__
    return f"{name} ({getattr(backend, 'name', backend)})" if backend else name


def _cudagraphs_captured() -> bool:
    try:
        from vllm.config import CUDAGraphMode, get_current_vllm_config

        mode = get_current_vllm_config().compilation_config.cudagraph_mode
        return mode is not None and mode != CUDAGraphMode.NONE
    except Exception:  # noqa: BLE001
        return False


# Live tensors as the last successful update left them, per module. Captured
# CUDA graphs and built-once kernels hold these objects, so a tensor replaced
# (or moved) between updates is one the next update would not write where they
# read. Checked at the start of each update, before any live write.
_LIVE_SIGNATURES: WeakKeyDictionary[torch.nn.Module, dict[str, tuple]] = (
    WeakKeyDictionary()
)

# Built-once kernel objects of reload-safe quant methods (and attention impls)
# as the last successful update left them: captured graphs read their state,
# so one rebuilt between updates would not be the one reload refreshes.
_KERNEL_IDS: WeakKeyDictionary[torch.nn.Module, dict[str, int]] = WeakKeyDictionary()
_KERNEL_ATTRS = ("moe_kernel", "moe_quant_config", "fp8_linear", "kernel")


def _live_signature(t: torch.Tensor) -> tuple:
    try:
        ptr = t.data_ptr()
    except (RuntimeError, NotImplementedError):  # e.g. tensor subclasses
        ptr = None
    return (id(t), ptr, tuple(t.shape), t.stride(), t.dtype)


def _kernel_ids(layer: torch.nn.Module) -> dict[str, int]:
    ids = {}
    method = getattr(layer, "quant_method", None)
    if isinstance(method, QuantizeMethodBase) and getattr(method, "reload_safe", False):
        ids["quant_method"] = id(method)
        for attr in _KERNEL_ATTRS:
            obj = getattr(method, attr, None)
            if obj is not None:
                ids[f"quant_method.{attr}"] = id(obj)
    impl = getattr(layer, "impl", None)
    if impl is not None and is_deferred_attention_layer(layer):
        ids["impl"] = id(impl)
    return ids


def _record_live_signatures(model: torch.nn.Module) -> None:
    for layer in model.modules():
        _LIVE_SIGNATURES[layer] = {
            name: _live_signature(t) for name, t in get_layer_tensors(layer).items()
        }
        _KERNEL_IDS[layer] = _kernel_ids(layer)


def _check_live_tensors_unchanged(model: torch.nn.Module) -> None:
    changed = []
    for module_name, layer in model.named_modules():
        expected = _LIVE_SIGNATURES.get(layer)
        if expected is None:
            continue
        current = get_layer_tensors(layer)
        for name, signature in expected.items():
            t = current.get(name)
            if t is None:
                changed.append(f"{module_name}.{name} (removed)")
            elif _live_signature(t) != signature:
                changed.append(f"{module_name}.{name}")
        current_ids = _kernel_ids(layer)
        for name, obj_id in _KERNEL_IDS.get(layer, {}).items():
            if current_ids.get(name) != obj_id:
                changed.append(f"{module_name}.{name} (kernel object replaced)")
    if not changed:
        return
    shown = ", ".join(changed[:5]) + (
        f" and {len(changed) - 5} more" if len(changed) > 5 else ""
    )
    msg = (
        f"Live tensors changed since the last weight update: {shown}. Something "
        "outside reload replaced, resized or moved them. Reload writes the "
        "tensors on the module now, so captured CUDA graphs and built-once "
        "kernels would keep reading the old ones. Update tensors in place "
        "(copy_ into the existing tensor) instead of replacing them."
    )
    if _cudagraphs_captured():
        raise ReloadUnsupportedError(msg)
    logger.warning_once(msg)


# ---------------------------------------------------------------------------
# Refresh without a reload session
# ---------------------------------------------------------------------------


@torch.no_grad()
def refresh_derived_state(
    model: torch.nn.Module, model_config: ModelConfig | None = None
) -> None:
    """Recompute weight-derived state in place after weights were written
    directly in live tensors (sparse patches, `is_checkpoint_format=False`
    reloads), without a reload session. Runs every refresh phase: kernel
    refresh (e.g. MoE alphas / activation gscales), attention (PWAL re-run on
    the live module and landed, so MLA's W_UK_T / W_UV follow a patched
    kv_b_proj and backends recompute their scale-derived state), then the
    layer and model refreshes. Without `model_config` the attention phase only
    runs the backends' `refresh(layer)`."""
    for layer in model.modules():
        refresh_quant_method(layer, getattr(layer, "quant_method", None))
    for layer in model.modules():
        if not is_deferred_attention_layer(layer):
            continue
        if model_config is not None:
            _rerun_attention_pwal(layer, model_config.dtype, _kv_scale_floats(layer))
        else:
            refresh = getattr(getattr(layer, "impl", None), "refresh", None)
            if callable(refresh):
                refresh(layer)
    run_model_refresh(model)
