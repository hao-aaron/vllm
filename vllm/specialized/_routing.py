# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core-private routing: picks a specialized directory or the general model.

Specialized directories never import this module (Rule 1).
"""

import contextlib
import dataclasses
import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.specialized.declare import FEATURES, RouteConfig, Support

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

MODES = ("auto", "off", "require")
# Libraries probed for `Support.libraries`.
PROBED_LIBRARIES = ("flashinfer-python", "nvidia-cutlass-dsl", "torch")


@dataclass
class Decision:
    family: str | None
    choice: str | None = None
    lines: list[str] = field(default_factory=list)


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


def _checkpoint(cfg: "VllmConfig") -> str:
    """Core's canonical quantization name, after aliases are resolved."""
    if cfg.quant_config is not None:
        return cfg.quant_config.get_name()
    return cfg.model_config.quantization or "none"


def _expert_quantization(cfg: "VllmConfig") -> str | None:
    """The routed experts' format when the checkpoint names one apart."""
    quant = getattr(cfg.model_config.hf_config, "quantization_config", None)
    if quant is not None and not isinstance(quant, dict):
        quant = quant.to_dict()
    algo = (quant or {}).get("moe_quant_algo")
    return str(algo).lower() if algo else None


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
        checkpoint=_checkpoint(cfg),
        expert_quantization=_expert_quantization(cfg),
        hf_config=mc.hf_text_config,
        kv_cache_dtype=str(cc.cache_dtype),
        attention_backend=None if backend is None else backend.name,
        moe_backend=cfg.kernel_config.moe_backend,
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


def _short(module: str, family: str) -> str:
    return module.removeprefix(family + ".").replace(".specialized.", "/specialized/")


def _matches_pin(module: str, pin: str) -> bool:
    return module.rsplit(".", 1)[-1] == pin or module.endswith("." + pin)


def route(cfg: "VllmConfig") -> Decision:
    """Pick a directory on the resolved config. Never changes a setting."""
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
        reasons = support.reasons(route_config(cfg))
        if reasons:
            reasons_by[name] = reasons
            continue
        d.choice = name
        break
    if d.choice is None and mode != "auto":
        detail = "; ".join(
            f"{_short(n, family)}: {', '.join(r)}" for n, r in reasons_by.items()
        )
        raise ValueError(f"--specialized-model {mode}: none qualifies. {detail}")
    d.lines.append(f"{arch}: {_short(d.choice, family) if d.choice else 'general'}")
    for n, r in reasons_by.items():
        d.lines.append(f"  {_short(n, family)}: {'; '.join(r)}")
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
    """VllmConfig hook, last in `__post_init__`: route, record, print."""
    mc = cfg.model_config
    # Once per engine: derived configs (the draft's) reuse the stored choice.
    if mc is None or mc.is_submodel_config or mc.specialized_choice is not None:
        return
    _map_legacy_switch(cfg)
    d = route(cfg)
    mc.specialized_choice = d.choice or ""
    for line in d.lines:
        logger.info("Specialized routing: %s", line)
    if mc.specialized_model_options:
        if d.choice:
            options(cfg)  # unknown or mistyped options fail at startup
        else:
            logger.warning(
                "Specialized routing: the general model ignores "
                "--specialized-model-options %s",
                mc.specialized_model_options,
            )


def options(cfg: "VllmConfig") -> Any:
    """The chosen directory's `OPTIONS` from --specialized-model-options, with
    its defaults; None if the directory declares none."""
    mc = cfg.model_config
    if not mc.specialized_choice:
        return None
    cls = getattr(importlib.import_module(mc.specialized_choice), "OPTIONS", None)
    given = mc.specialized_model_options or {}
    if cls is None:
        if given:
            raise ValueError(
                f"{mc.specialized_choice} takes no --specialized-model-options"
            )
        return None
    names = {f.name for f in dataclasses.fields(cls)}
    if unknown := sorted(set(given) - names):
        raise ValueError(
            f"{mc.specialized_choice} has no options {unknown}; it takes "
            f"{sorted(names)}"
        )
    return cls(**given)


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


def model_class(
    vllm_config: "VllmConfig",
    model_config: "ModelConfig",
    general_cls: type,
    arch: str,
) -> type:
    """The class to build for one architecture resolution: the chosen directory's
    MODEL entry for it, DRAFT_MODEL for the draft, otherwise the general class."""
    target = vllm_config.model_config
    choice = target.specialized_choice
    if not choice:
        return general_cls
    mod = importlib.import_module(choice)
    spec = vllm_config.speculative_config
    if spec is not None and model_config is spec.draft_model_config:
        ref = getattr(mod, "DRAFT_MODEL", {}).get(spec.method)
    else:
        ref = mod.MODEL.get(arch)
    if ref is None:  # unlisted architectures stay general
        return general_cls
    if choice not in _agreed:
        _agreed[choice] = _worker_agrees(vllm_config, choice)
        if not _agreed[choice] and target.specialized_model != "auto":
            raise ValueError(f"a rank declines {choice}")
        if not _agreed[choice]:
            logger.warning("Specialized routing: a rank declines %s; general", choice)
    if not _agreed[choice]:
        return general_cls
    module, _, cls_name = ref.partition(":")
    cls = getattr(importlib.import_module(f"{choice}.{module}"), cls_name)
    logger.info_once("Specialized model: building %s.%s", choice, cls_name)
    return cls
