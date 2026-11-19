# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many tokens a MoE may leave its top-k finalize to the next layer's fused
all-reduce + norm; import-light, so routing can read it."""

from typing import Any


def fused_finalize_max_tokens(vllm_config: Any) -> int:
    """Rows of the largest decode step: max_num_seqs x (1 + draft length),
    capped by the step's token budget."""
    sched = vllm_config.scheduler_config
    spec = vllm_config.speculative_config
    k = 0 if spec is None else int(spec.num_speculative_tokens or 0)
    return min(sched.max_num_seqs * (1 + k), sched.max_num_batched_tokens)
