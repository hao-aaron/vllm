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
  `reload_safe`; no instance monkeypatches of a quant method's PWAL; modules
  with their own PWAL and no quant method declare their outputs.
- 6.3 snapshot vs live structure: params or submodules recorded at init that
  are now gone (set to None, deleted), which reload cannot write.
- 6.5 configuration: EPLB, weight offloading, LoRA, transport limits.
- 6.6 after every update (`verify_post_update`): nothing left on meta, no
  module still holding checkpoint-format inputs.

The verifier proves structure, not values: a wrong refresh formula passes.
"""

import enum
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger(__name__)

__all__ = [
    "checkpoint_meta_from_safetensors",
    "verify_load_coverage",
    "merge_reload_reports",
    "ReloadItem",
    "ReloadReport",
    "ReloadStatus",
    "verify_post_update",
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
    ):
        self.items.append(ReloadItem(check, module, reason, status))

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
    model: torch.nn.Module,
    vllm_config: Any | None = None,
    checkpoint_meta: list[tuple[str, tuple[int, ...], torch.dtype]] | None = None,
) -> ReloadReport:
    """Reload checks of a loaded model: 6.1, 6.3, 6.5, and 6.4 from an
    observed cold load and, given checkpoint names / shapes / dtypes, from a
    dry run of `load_weights` (must not run during an update)."""
    report = ReloadReport()
    _check_declarations(model, report)
    _check_structure(model, report)
    _check_load_coverage(model, report)
    if vllm_config is not None:
        _check_configuration(model, vllm_config, report)
    if checkpoint_meta is not None:
        verify_load_coverage(model, checkpoint_meta, report)
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
        if not getattr(root, "reload_safe", True):
            report.add(
                "6.1",
                name,
                f"{type(root).__name__} opted this instance out (reload_safe=False)",
                UNSAFE,
            )
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
        if callable(pwal) and not getattr(module, "reload_outputs", ()):
            report.add(
                "6.1",
                name,
                f"{type(module).__name__} has its own process_weights_after_loading() "
                "without a quant method or declared reload_outputs: reload never "
                "runs it",
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
        if getattr(module, "reload_outputs", ()):
            continue  # module-level PWAL recreates and drops its inputs
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
# 6.4 load coverage (from an observed cold load, VLLM_RELOAD_OBSERVE_LOAD=1)
# ---------------------------------------------------------------------------


def _check_load_coverage(model: torch.nn.Module, report: ReloadReport) -> None:
    coverage = model.__dict__.get("_reload_load_coverage")
    if coverage is None:
        return
    for name in coverage.direct_writes:
        report.add(
            "6.4",
            name,
            "written during load_weights without its weight_loader: a reload "
            "writes a meta placeholder and keeps the old value",
            UNSAFE,
        )
    for name in coverage.never_loaded:
        report.add(
            "6.4",
            name,
            "no loader call reached it at cold start (absent from the "
            "checkpoint, or dropped by a mapper)",
            UNVERIFIED,
        )


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


def checkpoint_meta_from_safetensors(
    path: str,
) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """Names, shapes and dtypes of a local safetensors checkpoint (headers
    only)."""
    import glob
    import os

    from safetensors import safe_open

    meta = []
    for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                s = f.get_slice(name)
                meta.append(
                    (name, tuple(s.get_shape()), _SAFETENSORS_DTYPES[s.get_dtype()])
                )
    return meta


def verify_load_coverage(
    model: torch.nn.Module,
    checkpoint_meta: list[tuple[str, tuple[int, ...], torch.dtype]],
    report: ReloadReport | None = None,
) -> ReloadReport:
    """6.4 without data: run `model.load_weights` on meta stand-ins under the
    reload wrappers (`trace_loads`; nothing is written) and report modules a
    full checkpoint leaves without loader calls (direct writes in
    `load_weights`), partially counted modules, and names that reached no
    loader (dropped by a mapper, or buffered by the model)."""
    from .layerwise import get_reload_session, trace_loads

    report = report if report is not None else ReloadReport()
    with trace_loads(model) as trace:
        session = get_reload_session(model)
        assert session is not None
        required = dict(session.required_modules)

        def stand_ins():
            for name, shape, dtype in checkpoint_meta:
                trace.current_name = name
                yield name, torch.empty(shape, dtype=dtype, device="meta")
            trace.current_name = None

        try:
            model.load_weights(stand_ins())
        except Exception as e:  # noqa: BLE001
            report.add(
                "6.4",
                "",
                f"load_weights raised on meta stand-ins ({type(e).__name__}: "
                f"{str(e)[:200]}); a loader may branch on values. Reload runs "
                "loaders on real targets; the refit test is the check",
                UNVERIFIED,
            )
            return report
    modules = dict(model.named_modules())
    by_id = {id(m): m for m in modules.values()}
    for module_id, name in required.items():
        module = by_id.get(module_id)
        if module is None:
            continue
        copied = trace.copied_numel.get(module, 0)
        total = trace.total(module)
        if copied == 0:
            report.add(
                "6.4",
                name,
                "no loader call reached it from the checkpoint: written directly "
                "in load_weights (reload would keep the old value), or absent "
                "from the checkpoint",
                UNSAFE,
            )
        elif total is not None and copied < total:
            report.add(
                "6.4",
                name,
                f"loaders copied {copied}/{total} elements (padding a loader "
                "never writes, or a missing shard)",
                UNVERIFIED,
            )
    dropped = [name for name, _, _ in checkpoint_meta if not trace.name_hits[name]]
    if dropped:
        report.add(
            "6.4",
            "",
            f"{len(dropped)} checkpoint name(s) reached no loader (dropped by a "
            f"mapper, or buffered by the model), e.g. {dropped[:6]}",
            UNVERIFIED,
        )
    return report


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
        report.add("6.5", "", "sharded_rdt does not support EPLB", UNSAFE)
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
            )
        elif getattr(method, "uses_online_quant", False):
            report.add(
                "6.5",
                name,
                "sparse_nccl patches checkpoint coordinates into online-quantized "
                "weights",
                UNSAFE,
            )


# ---------------------------------------------------------------------------
# 6.6 after every update
# ---------------------------------------------------------------------------


def verify_post_update(
    model: torch.nn.Module, meta_before: set[tuple[int, str]] = frozenset()
) -> list[str]:
    """Invariants after a successful update: every tensor that was live
    before it is materialized, and no module holds checkpoint-format inputs
    or reload state. `meta_before`: (id(module), name) already on meta when
    the update started (never materialized, e.g. a disabled tower). Returns
    the violations."""
    from .layerwise import LAYERWISE_INFO

    problems = []
    for name, module in model.named_modules():
        for tname, t in list(module._parameters.items()) + list(
            module._buffers.items()
        ):
            if t is None or not t.is_meta or (id(module), tname) in meta_before:
                continue
            problems.append(f"{name}.{tname} left on meta")
        info = LAYERWISE_INFO.get(module)
        if info is not None and (info.can_load() or info.hosted or info.scratch_bytes):
            problems.append(f"{name or '<model>'} still holds reload state")
    return problems


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
