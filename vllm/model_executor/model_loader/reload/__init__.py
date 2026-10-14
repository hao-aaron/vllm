# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place weight reloading (weight transfer, `reload_weights`).

`start_reload` / `finish_reload` run an update on the reload core (`core.py`):
each module loads straight into its live memory when the live bytes can hold
the checkpoint format, otherwise into scratch, and completes when the units
its loaders report have arrived. Models that run post-load work inside
`load_weights`, or whose loads were never observed, take the legacy layerwise
path (`layerwise.py`; `VLLM_RELOAD_LEGACY=1` selects it everywhere).

Limitations: CPU weight offloading and EPLB rebalancing are not supported (an
update is refused).
"""

import torch

__all__ = [
    "start_reload",
    "finish_reload",
    "report_units",
    "report_declined",
    "current_load",
    "track_names",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "refresh_derived_state",
    "abort_reload",
    "begin_update",
    "check_can_serve",
    "end_update",
    "get_reload_session",
    "is_model_dirty",
    "is_reload_active",
    "mark_dirty",
    "record_written",
    "register_refresh",
    "declare_constant",
    "declare_derived",
    "merge_reload_reports",
    "ReloadReport",
    "ReloadStatus",
    "verify_reload",
    "set_torchao_reload_attrs",
    "support_quantized_model_reload_from_hp_weights",
]

from .layerwise import (
    abort_reload,
    begin_update,
    check_can_serve,
    end_update,
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    get_reload_session,
    initialize_layerwise_reload,
    is_model_dirty,
    is_reload_active,
    mark_dirty,
    record_metadata_for_reloading,
    record_written,
    refresh_derived_state,
)
from .refresh import declare_constant, declare_derived, register_refresh
from .torchao_decorator import (
    set_torchao_reload_attrs,
    support_quantized_model_reload_from_hp_weights,
)
from .units import current_load, report_declined, report_units, track_names
from .verify import ReloadReport, ReloadStatus, merge_reload_reports, verify_reload


def start_reload(
    model: torch.nn.Module,
    partial: bool | None = None,
    modules: list[str] | tuple[str, ...] | None = None,
    source: str | None = None,
) -> None:
    """Begin a weight update: the reload core, or the legacy layerwise path
    (`VLLM_RELOAD_LEGACY=1`, models that run post-load work inside
    `load_weights`, or whose loads were never observed). `source` names the
    expected-unit set ("trainer", or a checkpoint trace key)."""
    from .core import legacy_reason, start_core

    reason = legacy_reason(model, source)
    if reason is None:
        start_core(model, partial=partial, modules=modules, source=source)
        return
    _log_legacy(model, reason)
    initialize_layerwise_reload(model, partial=partial, modules=modules)


def finish_reload(model: torch.nn.Module, model_config) -> None:
    """Finish the weight update `start_reload` began."""
    from .core import core_active, finish_core

    if core_active(model):
        finish_core(model, model_config)
    else:
        finalize_layerwise_reload(model, model_config)


def _log_legacy(model, reason: str) -> None:
    from vllm.logger import init_logger

    logged = model.__dict__.setdefault("_reload_legacy_logged", set())
    if reason not in logged:
        logged.add(reason)
        init_logger(__name__).warning(
            "%s reloads through the legacy (layerwise) path: %s.",
            type(model).__name__,
            reason,
        )
