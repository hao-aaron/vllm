# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise weight reloading utilities for vLLM.

This module provides functionality to reload model weights layer-by-layer,
which is useful for weight updates without full model reconstruction.

Limitations:
1. Composition with CPU offloading has not been implemented
2. Tied parameters will only reflect processing from one of the parent layers (for
   example, only processing from embed_tokens will have an effect)
3. This design assumes that the number of weights loaded from disk is the same as the
   number of weights created at model init time. This is not true for quant methods
   which (1) pad weights or (2) load qkv weights into the same parameter. Both of these
   cases are non-issues for today's quant methods, but future quantizations may cause
   reloading to fail
"""

__all__ = [
    "start_reload",
    "finish_reload",
    "abort_reload",
    "trace_loads",
    "current_load",
    "ensure_materialized",
    "complete_module",
    "LoadTarget",
    "LoadTrace",
    "add_reload_transform",
    "get_reload_session",
    "is_model_dirty",
    "landing_outcomes",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "set_torchao_reload_attrs",
    "support_quantized_model_reload_from_hp_weights",
]

from .direct import landing_outcomes
from .layerwise import (
    LoadTarget,
    LoadTrace,
    abort_reload,
    add_reload_transform,
    complete_module,
    current_load,
    ensure_materialized,
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    finish_reload,
    get_reload_session,
    initialize_layerwise_reload,
    is_model_dirty,
    record_metadata_for_reloading,
    start_reload,
    trace_loads,
)
from .torchao_decorator import (
    set_torchao_reload_attrs,
    support_quantized_model_reload_from_hp_weights,
)
