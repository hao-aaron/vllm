# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many tokens a MoE may leave its top-k finalize to the next layer's fused
all-reduce + norm; import-light, so routing can read it."""

from typing import Any

# The default; larger batches finalize in the MoE.
FUSED_MOE_FINALIZE_MAX_TOKENS = 32
# The `additional_config` key that raises it (it sizes a collective buffer).
FUSED_MOE_FINALIZE_MAX_TOKENS_KEY = "minimax_m3_fused_finalize_max_tokens"


def fused_finalize_max_tokens(vllm_config: Any) -> int:
    extra = vllm_config.additional_config
    extra = extra if isinstance(extra, dict) else {}
    return int(
        extra.get(FUSED_MOE_FINALIZE_MAX_TOKENS_KEY, FUSED_MOE_FINALIZE_MAX_TOKENS)
    )
