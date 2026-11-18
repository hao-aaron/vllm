# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `Model` base class a specialized directory implements."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class StepResult:
    """A fused step's outcome; see `Model.fused_step`."""

    sampled: torch.Tensor  # [num_reqs, k + 1] int32, left-aligned
    num_sampled: torch.Tensor  # [num_reqs] int32
    next_drafts: torch.Tensor | None = None  # [num_reqs, k] int32


class Model(nn.Module):
    """An `nn.Module` that owns its run-time state and hooks.

    Core wraps it in a pass-through ModelState whose defaults are the ones the
    general model would get, so a subclass overrides only the hooks it needs.
    """

    def warmup(self) -> None:
        """Build derived state, JIT and autotune; runs after load, before profiling."""

    def init_kv_cache(self, kv_cache_config: Any, block_tables: Any) -> None:
        """Runs after every KV cache bind (profiling cache, then the real one)."""

    def add_request(self, req_index: int, request: Any) -> None:
        pass

    def remove_request(self, req_id: str) -> None:
        pass

    def capture_inner_graphs(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Logits for sampling, or None for core's default."""
        return None

    def attach_draft(self, draft: nn.Module) -> None:
        """Called on the target once it and its `DRAFT_MODEL` are both loaded."""

    def fused_step(self, batch: Any, sampling: Any) -> StepResult | None:
        return None

    def count(self, path: str, n: int = 1) -> None:
        """Per-path step counter, printed by core at shutdown."""
        counts = self.__dict__.setdefault("_specialized_path_counts", Counter())
        counts[path] += n

    def path_counts(self) -> dict[str, int]:
        return dict(self.__dict__.get("_specialized_path_counts", {}))
