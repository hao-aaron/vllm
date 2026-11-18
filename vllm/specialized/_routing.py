# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core-private routing: picks a specialized directory or the general model.

Specialized directories never import this module (Rule 1).
"""

import contextlib
import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.specialized.declare import DEFAULT_KEYS, FEATURES, RouteConfig, Support

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

MODES = ("auto", "off", "require")
LANGUAGE_MODEL = "language_model"
# Libraries probed for `Support.libraries`.
PROBED_LIBRARIES = ("flashinfer-python", "nvidia-cutlass-dsl", "torch")


@dataclass
class Decision:
    family: str | None
    choice: str | None = None
    lines: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)


def platform_key() -> str:
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        return "cuda"
    if current_platform.is_rocm():
        return "rocm"
    if current_platform.is_xpu():
        return "xpu"
    return "other"


def _arch() -> str:
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability()
    if cap is None:
        return "unknown"
    prefix = "gfx" if current_platform.is_rocm() else "sm"
    return f"{prefix}{cap.major}{cap.minor}"


def _libraries() -> dict[str, str]:
    out = {}
    for lib in PROBED_LIBRARIES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            out[lib] = importlib.metadata.version(lib)
    return out


def family_of(model_config: "ModelConfig") -> str | None:
    """The `vllm.models.<family>` package of the model's architecture, if any."""
    from vllm.model_executor.models.registry import ModelRegistry

    for arch in model_config.architectures or ():
        entry = ModelRegistry.models.get(arch)
        module = getattr(entry, "module_name", "")
        parts = module.split(".")
        if module.startswith("vllm.models.") and len(parts) >= 3:
            return ".".join(parts[:3])
    return None


def candidates(family: str, platform: str) -> list[str]:
    """Module paths of the family's specialized directories for `platform`."""
    if importlib.util.find_spec(f"{family}.routing") is None:
        return []
    table = importlib.import_module(f"{family}.routing").SPECIALIZED
    return [f"{family}.{rel}" for rel in table.get(platform, ())]


def _features(cfg: "VllmConfig") -> frozenset[str]:
    import vllm.envs as envs

    mc, pc = cfg.model_config, cfg.parallel_config
    used = {
        "lora": cfg.lora_config is not None,
        "eplb": pc.enable_eplb,
        "sleep_mode": mc.enable_sleep_mode,
        "weight_reload": cfg.weight_transfer_config is not None,
        "batch_invariance": envs.VLLM_BATCH_INVARIANT,
        "multimodal": mc.is_multimodal_model
        and not (mc.multimodal_config and mc.multimodal_config.language_model_only),
        "prompt_embeds": mc.enable_prompt_embeds,
        "kv_transfer": cfg.kv_transfer_config is not None,
    }
    return frozenset(k for k, v in used.items() if v)


def route_config(cfg: "VllmConfig") -> RouteConfig:
    from vllm.config import CompilationMode

    mc, pc, cc = cfg.model_config, cfg.parallel_config, cfg.cache_config
    spec = cfg.speculative_config
    backend = cfg.attention_config.backend
    graph_mode = cfg.compilation_config.cudagraph_mode
    kv = cfg.kv_transfer_config
    role = "colocated"
    if kv is not None and getattr(kv, "kv_role", None) in ("kv_producer",):
        role = "prefill"
    elif kv is not None and getattr(kv, "kv_role", None) in ("kv_consumer",):
        role = "decode"
    return RouteConfig(
        platform=platform_key(),
        arch=_arch(),
        compute_units=None,
        machine=frozenset(),
        libraries=_libraries(),
        dtype=str(mc.dtype).removeprefix("torch."),
        max_model_len=mc.max_model_len,
        checkpoint=mc.quantization or "none",
        hf_config=mc.hf_text_config,
        kv_cache_dtype=str(cc.cache_dtype),
        attention_backend=None if backend is None else backend.name,
        tp=pc.tensor_parallel_size,
        pp=pc.pipeline_parallel_size,
        dp=pc.data_parallel_size,
        ep=(pc.tensor_parallel_size * pc.data_parallel_size)
        if pc.enable_expert_parallel
        else 1,
        dcp=pc.decode_context_parallel_size,
        pcp=pc.prefill_context_parallel_size,
        micro_batching=bool(pc.use_ubatching),
        speculation=None if spec is None else spec.method,
        num_speculative_tokens=0 if spec is None else spec.num_speculative_tokens,
        graph_mode=None if graph_mode is None else graph_mode.name,
        compile=cfg.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
        role=role,
        features=_features(cfg),
        block_size=cc.block_size,
        settings=cfg,
    )


def _get(cfg: Any, key: str) -> Any:
    obj = cfg
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def _set(cfg: Any, key: str, value: Any) -> None:
    *path, last = key.split(".")
    obj = cfg
    for part in path:
        obj = getattr(obj, part)
    setattr(obj, last, value)


def _is_unset(cfg: Any, key: str) -> bool:
    if key == "cache_config.block_size":
        # Platforms fill a block size before routing; only the user's counts.
        return not cfg.cache_config.user_specified_block_size
    value = _get(cfg, key)
    return value is None or value == "auto"


def _coerce(key: str, value: Any) -> Any:
    if key == "compilation_config.cudagraph_mode" and isinstance(value, str):
        from vllm.config import CUDAGraphMode

        return CUDAGraphMode[value]
    if key == "attention_config.backend" and isinstance(value, str):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        return AttentionBackendEnum[value]
    return value


def _apply_defaults(cfg: "VllmConfig", mod: ModuleType) -> dict[str, Any]:
    """Fill the directory's defaults into unset keys; returns what was filled."""
    fn = getattr(mod, "defaults", None)
    if fn is None:
        return {}
    proposed = fn(route_config(cfg))
    filled = {}
    for key, value in proposed.items():
        if key not in DEFAULT_KEYS:
            raise ValueError(f"{mod.__name__}.defaults sets {key}, not allowed")
        if _is_unset(cfg, key):
            filled[key] = (_get(cfg, key), value)
            _set(cfg, key, _coerce(key, value))
        else:
            logger.info(
                "%s: keeping user setting %s=%s (default %s)",
                mod.__name__,
                key,
                _get(cfg, key),
                value,
            )
    return filled


def _undo(cfg: "VllmConfig", filled: dict[str, Any]) -> None:
    for key, (old, _) in filled.items():
        _set(cfg, key, old)


def _short(module: str, family: str) -> str:
    return module.removeprefix(family + ".").replace(".specialized.", "/specialized/")


def _matches_pin(module: str, pin: str) -> bool:
    return module.rsplit(".", 1)[-1] == pin or module.endswith("." + pin)


def route(cfg: "VllmConfig") -> Decision:
    """Pick a directory; applies its defaults to `cfg`. Runs once, at startup."""
    mc = cfg.model_config
    family = family_of(mc)
    d = Decision(family=family)
    mode = mc.specialized_model
    if family is None:
        if mode not in MODES[:2]:
            raise ValueError(f"--specialized-model {mode}: model has no family")
        return d
    names = candidates(family, platform_key())
    arch = (mc.architectures or ["?"])[0]
    if mode == "off" or not names:
        if mode == "require" or (mode not in MODES and not names):
            raise ValueError(f"--specialized-model {mode}: {arch} has none here")
        d.lines.append(f"{arch}: general" + (" (off)" if mode == "off" else ""))
        return d
    if mode not in MODES and not any(_matches_pin(n, mode) for n in names):
        raise ValueError(
            f"--specialized-model {mode}: not one of "
            f"{[_short(n, family) for n in names]}"
        )
    reasons_by: dict[str, list[str]] = {}
    for name in names:
        if mode not in MODES and not _matches_pin(name, mode):
            continue
        mod = importlib.import_module(name)
        support: Support = mod.SUPPORTS
        if mode == "auto" and not support.default_on:
            reasons_by[name] = ["default_on=False (pin it or use require)"]
            continue
        if isinstance(mod.MODEL, dict) and arch not in mod.MODEL:
            reasons_by[name] = [f"no MODEL for architecture {arch}"]
            continue
        filled = _apply_defaults(cfg, mod)
        reasons = support.reasons(route_config(cfg))
        if reasons:
            _undo(cfg, filled)
            reasons_by[name] = reasons
            continue
        d.choice, d.defaults = name, {k: v for k, (_, v) in filled.items()}
        break
    if d.choice is None and mode != "auto":
        detail = "; ".join(
            f"{_short(n, family)}: {', '.join(r)}" for n, r in reasons_by.items()
        )
        raise ValueError(f"--specialized-model {mode}: none qualifies. {detail}")
    d.lines.append(f"{arch}: {_short(d.choice, family) if d.choice else 'general'}")
    for n, r in reasons_by.items():
        d.lines.append(f"  {_short(n, family)}: {'; '.join(r)}")
    if d.defaults:
        d.lines.append(
            "  defaults: " + ", ".join(f"{k}={v}" for k, v in d.defaults.items())
        )
    return d


_LEGACY = {"stock": "off", "auto": "auto", "latency_specialized": "require"}


def _map_legacy_switch(cfg: "VllmConfig") -> None:
    """Deprecated: `decode_backend` (and `--monokernel`) for families that route."""
    extra = cfg.additional_config
    backend = extra.get("decode_backend") if isinstance(extra, dict) else None
    mc = cfg.model_config
    family = family_of(mc)
    if backend is None or family is None or not candidates(family, platform_key()):
        return
    mode = _LEGACY.get(backend)
    if mode is None:
        raise ValueError(f"decode_backend={backend!r}: use --specialized-model")
    if mc.specialized_model == "auto" and mode != "auto":
        mc.specialized_model = mode
    logger.warning(
        "decode_backend=%s is deprecated for this model; using --specialized-model %s.",
        backend,
        mc.specialized_model,
    )


def route_and_record(cfg: "VllmConfig") -> None:
    """VllmConfig hook: route, record the choice on the model config, print."""
    mc = cfg.model_config
    # Once per engine: derived configs (draft overrides) share this model config.
    if mc is None or mc.is_submodel_config or mc.specialized_choice is not None:
        return
    _map_legacy_switch(cfg)
    d = route(cfg)
    mc.specialized_choice = d.choice or ""
    for line in d.lines:
        logger.info("Specialized routing: %s", line)


def verify_resolved(cfg: "VllmConfig") -> None:
    """Re-check the choice once core has resolved every setting."""
    mc = cfg.model_config
    if mc is None or mc.is_submodel_config or mc.specialized_verified:
        return
    mc.specialized_verified = True
    if not mc.specialized_choice:
        return
    mod = importlib.import_module(mc.specialized_choice)
    reasons = mod.SUPPORTS.reasons(route_config(cfg))
    if not reasons:
        return
    msg = f"{mc.specialized_choice} declined after config resolution: " + "; ".join(
        reasons
    )
    if mc.specialized_model == "auto":
        logger.warning("Specialized routing: %s; using the general model.", msg)
        mc.specialized_choice = ""
    else:
        raise ValueError(msg)


def _rank_local(support: Support) -> Support:
    """Only the facts each rank probes; the engine checked the rest on the full
    config (a worker may hold a submodel's config here)."""
    return Support(
        arch=support.arch,
        compute_units=support.compute_units,
        machine=support.machine,
        libraries=support.libraries,
        features=FEATURES,
    )


def _worker_agrees(cfg: "VllmConfig", choice: str) -> bool:
    """Every rank re-checks the per-rank facts; one objection moves all ranks."""
    from vllm.specialized import services

    mod = importlib.import_module(choice)
    reasons = _rank_local(mod.SUPPORTS).reasons(route_config(cfg))
    if reasons:
        logger.warning(
            "Specialized routing: this rank declines %s: %s", choice, "; ".join(reasons)
        )
    ok = services.comm.cpu_all_reduce_min(0 if reasons else 1, group="world")
    return bool(ok)


_agreed: dict[str, bool] = {}


def _arch_of(model_config: "ModelConfig") -> str:
    return (model_config.architectures or [""])[0]


def model_class(
    vllm_config: "VllmConfig", model_config: "ModelConfig", general_cls: type
) -> type:
    """The class to build: the chosen directory's MODEL / DRAFT_MODEL, or general."""
    target = vllm_config.model_config
    choice = target.specialized_choice
    spec = vllm_config.speculative_config
    is_draft = spec is not None and model_config is spec.draft_model_config
    if not choice:
        return general_cls
    mod = importlib.import_module(choice)
    table = mod.MODEL if isinstance(mod.MODEL, dict) else {}
    if is_draft:
        pass
    elif model_config.is_submodel_config:
        # A wrapper's language model: specialized only when MODEL maps the
        # wrapper to "language_model" (the wrapper then stays general).
        if LANGUAGE_MODEL not in table.values() or general_cls.__name__ not in table:
            return general_cls
    elif model_config is not target or table.get(_arch_of(target)) == LANGUAGE_MODEL:
        return general_cls
    if choice not in _agreed:
        _agreed[choice] = _worker_agrees(vllm_config, choice)
        if not _agreed[choice]:
            if target.specialized_model != "auto":
                raise ValueError(f"a rank declines {choice}")
            logger.warning("Specialized routing: a rank declines %s; general", choice)
    if not _agreed[choice]:
        return general_cls
    if not is_draft:
        ref = mod.MODEL
        if isinstance(ref, dict):  # one class per registered architecture
            ref = ref.get(general_cls.__name__)
            if ref is None:
                raise ValueError(f"{choice} has no MODEL for {general_cls.__name__}")
    else:
        assert spec is not None
        ref = getattr(mod, "DRAFT_MODEL", {}).get(spec.method)
        if ref is None:
            return general_cls
    module, _, cls_name = ref.partition(":")
    cls = getattr(importlib.import_module(f"{choice}.{module}"), cls_name)
    logger.info_once("Specialized model: building %s.%s", choice, cls_name)
    return cls
