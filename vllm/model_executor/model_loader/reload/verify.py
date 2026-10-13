# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reload verifier: can this model's weights be updated in place correctly?

`verify_reload(model, vllm_config)` checks what can be checked from structure
and configuration, and reports per item:

- SAFE: checked and fine;
- UNSAFE: an update would serve stale or wrong weights (or hang);
- UNVERIFIED: not checkable here; a refit-equals-fresh test is the check.

Checks (numbered as in the SupportsReload RFC):

- 6.1 declarations: the model implements `SupportsReload` and has no root
  `process_weights_after_loading`; MoE methods that built a kernel are
  `reload_safe`; no instance monkeypatches of a quant method's PWAL; no
  module-level PWAL without a quant method (reload never runs it).
- 6.2 tensor ownership: device tensors held outside the module registry
  that no object declared (run it after CUDA graph capture).
- 6.3 snapshot vs live structure: params or submodules recorded at init that
  are now gone (set to None, deleted), which reload cannot write.
- 6.5 configuration: EPLB, weight offloading, LoRA, transport limits.

`check_reload_support` reports everything. At the start of every update,
`enforce` refuses only items known to produce wrong weights (configuration:
EPLB, weight offloading, sparse patches into kernel layouts), decided by all
ranks together; VLLM_RELOAD_ALLOW_UNSAFE=1 overrides.

The verifier proves structure, not values: a wrong refresh formula passes.
"""

import enum
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger(__name__)

__all__ = [
    "ReloadRefusedError",
    "enforce",
    "policy_report",
    "refuses",
    "find_unowned_tensors",
    "merge_reload_reports",
    "ReloadItem",
    "ReloadReport",
    "ReloadStatus",
    "verify_reload",
]


class ReloadStatus(str, enum.Enum):
    SAFE = "SAFE"
    UNVERIFIED = "UNVERIFIED"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True)
class ReloadItem:
    check: str
    module: str
    reason: str
    status: ReloadStatus
    # known to produce wrong weights: an update is refused (see `enforce`)
    refuse: bool = False


@dataclass
class ReloadReport:
    items: list[ReloadItem] = field(default_factory=list)

    @property
    def status(self) -> ReloadStatus:
        statuses = {item.status for item in self.items}
        for s in (ReloadStatus.UNSAFE, ReloadStatus.UNVERIFIED):
            if s in statuses:
                return s
        return ReloadStatus.SAFE

    def add(
        self,
        check: str,
        module: str,
        reason: str,
        status: ReloadStatus,
        refuse: bool = False,
    ):
        self.items.append(ReloadItem(check, module, reason, status, refuse))

    def of(self, status: ReloadStatus) -> list[ReloadItem]:
        return [item for item in self.items if item.status == status]

    def summary(self, limit: int = 20) -> str:
        lines = [f"reload verification: {self.status.value}"]
        problems = self.of(ReloadStatus.UNSAFE) + self.of(ReloadStatus.UNVERIFIED)
        for item in problems[:limit]:
            lines.append(
                f"  {item.status.value} [{item.check}] {item.module or '<model>'}: "
                f"{item.reason}"
            )
        if len(problems) > limit:
            lines.append(f"  ... and {len(problems) - limit} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "items": [
                {**asdict(item), "status": item.status.value} for item in self.items
            ],
        }


SAFE, UNSAFE, UNVERIFIED = (
    ReloadStatus.SAFE,
    ReloadStatus.UNSAFE,
    ReloadStatus.UNVERIFIED,
)


def verify_reload(
    model: torch.nn.Module, vllm_config: Any | None = None
) -> ReloadReport:
    """Reload checks of a loaded model: 6.1, 6.2, 6.3 and, given the config,
    6.5. Run it after CUDA graph capture (lazily built kernel state exists by
    then) and outside a weight update."""
    report = ReloadReport()
    _check_declarations(model, report)
    _check_ownership(model, report)
    _check_structure(model, report)
    if vllm_config is not None:
        _check_configuration(model, vllm_config, report)
    return report


# ---------------------------------------------------------------------------
# 6.1 declarations
# ---------------------------------------------------------------------------


def _reload_roots(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """The model and nested models (a VL wrapper's language model)."""
    return [
        (name, m)
        for name, m in model.named_modules()
        if m is model or getattr(m, "supports_reload", False)
    ]


def _check_declarations(model: torch.nn.Module, report: ReloadReport) -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    for name, root in _reload_roots(model):
        hook = getattr(type(root), "process_weights_after_loading", None)
        if not getattr(root, "supports_reload", False):
            if hook is not None:
                report.add(
                    "6.1",
                    name,
                    f"{type(root).__name__} has a model-level "
                    "process_weights_after_loading() that reload never calls, "
                    "and does not implement SupportsReload",
                    UNSAFE,
                )
            else:
                report.add(
                    "6.1",
                    name,
                    f"{type(root).__name__} does not implement SupportsReload",
                    UNVERIFIED,
                )
            continue
        if hook is not None:
            report.add(
                "6.1",
                name,
                "a SupportsReload model defines a root "
                "process_weights_after_loading(); its work must move to loaders, "
                "module PWAL, refresh_model() or post_load_setup()",
                UNSAFE,
            )
        if getattr(root, "finalizes_weights_during_load", False):
            report.add(
                "6.1",
                name,
                "finalizes_weights_during_load: load_weights runs post-load work, "
                "which a streamed reload cannot",
                UNSAFE,
            )

    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, QuantizeMethodBase):
            if "process_weights_after_loading" in vars(method):
                report.add(
                    "6.1",
                    name,
                    f"{type(method).__name__}.process_weights_after_loading is "
                    "monkeypatched on the instance; use wrap_quant_method",
                    UNSAFE,
                )
            if (
                isinstance(method, FusedMoEMethodBase)
                and getattr(method, "moe_kernel", None) is not None
                and not getattr(method, "reload_safe", False)
            ):
                from .layerwise import _moe_method_label

                report.add(
                    "6.1",
                    name,
                    f"{_moe_method_label(method)} built a kernel and is not "
                    "reload_safe (rebuilt on reload; captured graphs keep the "
                    "old one)",
                    UNSAFE,
                )
            continue
        if (
            module is model
            or is_deferred_attention_layer(module)
            or getattr(module, "supports_reload", False)
        ):
            continue
        pwal = getattr(type(module), "process_weights_after_loading", None)
        if callable(pwal):
            report.add(
                "6.1",
                name,
                f"{type(module).__name__} has its own process_weights_after_loading() "
                "without a quant method: reload never runs it",
                UNSAFE,
            )


# ---------------------------------------------------------------------------
# 6.2 tensor ownership (run after CUDA graph capture, so lazy caches exist)
# ---------------------------------------------------------------------------

# Attributes holding per-step runtime state, not weights (from #60009)
RUNTIME_STATE_ATTRS = frozenset(
    {
        "kv_cache",
        "_replayssm_ring_start",
        "_replayssm_prev_num_accepted",
        "_replayssm_prev_query_len",
        "deepstack_input_embeds",
        "per_layer_embeddings",
        "topk_indices_buffer",
        "_topk_indices_buffer",
        "_workspace_buffer",
        "index_group",
        "diffusion_states",
        "aot_compiled_fn",
    }
)


def _storage(t: torch.Tensor) -> int | None:
    try:
        return t.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return None


def find_unowned_tensors(model: torch.nn.Module) -> list[str]:
    """Paths of device tensors the model holds outside its parameters and
    buffers that no object declared (`declare_derived` / `declare_constant`
    / `register_derived` / `add_refresh` outputs / `persistent_tensors()`).
    A weight update leaves those stale, or at a new address captured graphs
    don't read. Walks module attributes and the plain objects they hold
    (quant methods, kernels, attention impls)."""
    from .refresh import declared_tensors

    owned = {_storage(t) for t in (*model.parameters(), *model.buffers())}
    modules = {id(m) for m in model.modules()}
    found: list[str] = []
    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type not in ("cpu", "meta") and (
                _storage(value) not in owned
            ):
                found.append(path)
        elif isinstance(value, torch.nn.Module):
            if id(value) not in modules:
                for name, t in (*value.named_parameters(), *value.named_buffers()):
                    visit(t, f"{path}.{name}")
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]")
        elif isinstance(value, dict):
            for key, item in list(value.items()):
                visit(item, f"{path}[{key!r}]")
        elif not isinstance(value, type) and isinstance(
            getattr(value, "__dict__", None), dict
        ):
            owner = getattr(type(value), "__module__", "") or ""
            if owner.startswith(("vllm.config", "transformers", "torch")):
                return
            owned.update(_storage(t) for t in declared_tensors(value))
            for name, item in list(value.__dict__.items()):
                if name not in RUNTIME_STATE_ATTRS and not name.startswith(
                    "_vllm_reload_"
                ):
                    visit(item, f"{path}.{name}")

    skip = {"_parameters", "_buffers", "_modules", *RUNTIME_STATE_ATTRS}
    for module_name, module in model.named_modules():
        owned.update(_storage(t) for t in declared_tensors(module))
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if name not in skip and not name.startswith(("_forward_", "_vllm_reload_")):
                visit(value, f"{module_name or '<root>'}.{name}")
    return found


def _check_ownership(model: torch.nn.Module, report: ReloadReport) -> None:
    from collections import Counter

    import regex as re

    # one item per attribute path, layer indices folded
    paths = Counter(re.sub(r"\.\d+\.", ".*.", p) for p in find_unowned_tensors(model))
    for path, n in sorted(paths.items()):
        report.add(
            "6.2",
            path,
            f"device tensor held outside the module registry, not declared"
            f"{f' (x{n})' if n > 1 else ''}: if weight-derived, a weight update "
            "leaves it stale (declare_derived + a refresh); if not, "
            "declare_constant",
            UNSAFE,
        )


# ---------------------------------------------------------------------------
# 6.3 snapshot vs live structure
# ---------------------------------------------------------------------------


def _check_structure(model: torch.nn.Module, report: ReloadReport) -> None:
    from .layerwise import LAYERWISE_INFO
    from .meta import SKIP_TENSORS

    recorded = model.__dict__.get("_reload_module_names")
    if recorded is not None:
        live = {name for name, _ in model.named_modules()}
        gone = [name for name in recorded if name not in live]
        # report the topmost deleted modules only
        top = [n for n in gone if not any(n.startswith(g + ".") for g in gone)]
        for name in top:
            report.add(
                "6.3",
                name,
                "submodule recorded at init is gone (deleted or replaced by "
                "None); reload cannot write its weights",
                UNSAFE,
            )
    for name, module in model.named_modules():
        info = LAYERWISE_INFO.get(module)
        if info is None or is_deferred_attention_layer(module):
            continue  # attention scale params are recreated by reload
        params, buffers = info.restore_metadata
        for pname in params:
            if pname in SKIP_TENSORS:
                continue
            if getattr(module, pname, None) is None:
                report.add(
                    "6.3",
                    f"{name}.{pname}" if name else pname,
                    "param recorded at init is now None or missing; reload "
                    "cannot write it (keep the Parameter, release its storage)",
                    UNSAFE,
                )


# ---------------------------------------------------------------------------
# 6.5 configuration
# ---------------------------------------------------------------------------

# MoE backends whose live weights are a checkpoint-layout copy (sparse patches
# in checkpoint coordinates land correctly)
_SPARSE_SAFE_MOE_BACKENDS = ("TRITON", "DEEPGEMM", "MARLIN")


def _check_configuration(
    model: torch.nn.Module, vllm_config: Any, report: ReloadReport
) -> None:
    parallel = vllm_config.parallel_config
    transfer = getattr(vllm_config, "weight_transfer_config", None)
    backend = getattr(transfer, "backend", None)

    if parallel.enable_eplb:
        report.add(
            "6.5",
            "",
            "EPLB is enabled: loaders write experts into their startup slots, "
            "not where a rebalance moved them",
            UNSAFE,
            refuse=True,
        )
    offload = getattr(vllm_config, "offload_config", None)
    uva = getattr(offload, "uva", None)
    prefetch = getattr(offload, "prefetch", None)
    if (uva is not None and uva.cpu_offload_gb > 0) or (
        prefetch is not None and prefetch.offload_group_size > 0
    ):
        report.add(
            "6.5",
            "",
            "weight offloading is not supported by reload",
            UNSAFE,
            refuse=True,
        )
    if getattr(vllm_config, "lora_config", None) is not None:
        report.add(
            "6.5",
            "",
            "LoRA: adapters are evicted on every weight update",
            UNVERIFIED,
        )
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "word_embeddings_untied_by_checkpoint", False):
        lm_head = getattr(getattr(model, "lm_head", None), "weight", None)
        embed = next(
            (m.weight for n, m in model.named_modules() if n.endswith("embed_tokens")),
            None,
        )
        if lm_head is not None and lm_head is embed:
            report.add(
                "6.5",
                "lm_head",
                "re-tied to embed_tokens (the checkpoint's lm_head was identical): "
                "an update that sends a different lm_head overwrites the shared "
                "tensor; train with tied embeddings or set "
                "tie_word_embeddings=False",
                UNVERIFIED,
            )
    if backend == "sharded_rdt" and parallel.enable_eplb:
        report.add("6.5", "", "sharded_rdt does not support EPLB", UNSAFE, refuse=True)
    if backend == "sparse_nccl":
        _check_sparse_layouts(model, report)


def _check_sparse_layouts(model: torch.nn.Module, report: ReloadReport) -> None:
    """Sparse NCCL patches write checkpoint coordinates into live tensors:
    only correct where the live layout is the checkpoint layout."""
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, FusedMoEMethodBase):
            continue
        backend = next(
            (
                getattr(method, attr)
                for attr in ("fp8_backend", "mxfp4_backend", "nvfp4_backend")
                if getattr(method, attr, None) is not None
            ),
            None,
        )
        label = getattr(backend, "name", str(backend))
        if backend is not None and not label.startswith(_SPARSE_SAFE_MOE_BACKENDS):
            report.add(
                "6.5",
                name,
                f"sparse_nccl patches checkpoint coordinates, but {label} keeps "
                "experts in a kernel layout",
                UNSAFE,
                refuse=True,
            )
        elif getattr(method, "uses_online_quant", False):
            report.add(
                "6.5",
                name,
                "sparse_nccl patches checkpoint coordinates into online-quantized "
                "weights",
                UNSAFE,
                refuse=True,
            )


def merge_reload_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-rank `ReloadReport.to_dict()`s: the worst status, and every
    non-SAFE item tagged with its rank."""
    order = [s.value for s in (SAFE, UNVERIFIED, UNSAFE)]
    status = max((r["status"] for r in reports), key=order.index, default="SAFE")
    items = [
        {**item, "rank": rank}
        for rank, r in enumerate(reports)
        for item in r["items"]
        if item["status"] != "SAFE"
    ]
    return {"status": status, "items": items}


# ---------------------------------------------------------------------------
# Policy at the start of an update
# ---------------------------------------------------------------------------

# VLLM_RELOAD_ALLOW_UNSAFE=1 lets updates run despite known-wrong items
ALLOW_UNSAFE = envs.VLLM_RELOAD_ALLOW_UNSAFE


class ReloadRefusedError(RuntimeError):
    pass


def policy_report(model: torch.nn.Module, vllm_config: Any) -> ReloadReport:
    """The checks an update's start enforces: configuration (6.5). Cheap
    enough to run before every update."""
    report = ReloadReport()
    _check_configuration(model, vllm_config, report)
    return report


def refuses(report: ReloadReport) -> bool:
    """Whether `enforce` would refuse (the override considered)."""
    return not ALLOW_UNSAFE and any(i.refuse for i in report.items)


def enforce(report: ReloadReport) -> None:
    """Refuse an update only for items known to produce wrong weights
    (VLLM_RELOAD_ALLOW_UNSAFE=1 runs it anyway, with a warning). Everything
    else is left to `check_reload_support`."""
    refusals = [i for i in report.items if i.refuse]
    if not refusals:
        return
    msg = "; ".join(
        f"[{i.check}] {i.module or '<model>'}: {i.reason}" for i in refusals[:5]
    )
    if ALLOW_UNSAFE:
        logger.warning_once("UNSAFE WEIGHT UPDATE (allowed): %s", msg)
        return
    raise ReloadRefusedError(
        f"Weight update refused, it would serve wrong weights: {msg}. Fix the "
        "configuration (see LLM.check_reload_support()), or set "
        "VLLM_RELOAD_ALLOW_UNSAFE=1 to update anyway."
    )
