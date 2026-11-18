# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pass-through ModelState for a specialized `Model`: the general model's state
class, with the model's own hooks run after each default."""

import functools
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_states.interface import ModelState

logger = init_logger(__name__)


@functools.cache
def adapter_for(base: type[ModelState]) -> type[ModelState]:
    class SpecializedModelState(base):  # type: ignore[valid-type,misc]
        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            from vllm.specialized.model import Model

            if not isinstance(model, Model):  # a wrapper over a specialized LM
                model = model.get_language_model()
            self.specialized = model
            model.warmup()

        def initialize_kv_cache(self, kv_cache_config, block_tables):
            super().initialize_kv_cache(kv_cache_config, block_tables)
            self.specialized.init_kv_cache(kv_cache_config, block_tables)

        def add_request(self, req_index, new_req_data):
            super().add_request(req_index, new_req_data)
            self.specialized.add_request(req_index, new_req_data)

        def remove_request(self, req_id):
            super().remove_request(req_id)
            self.specialized.remove_request(req_id)

        def capture_inner_cudagraphs(self, *args: Any, **kwargs: Any) -> None:
            super().capture_inner_cudagraphs(*args, **kwargs)
            self.specialized.capture_inner_graphs(*args, **kwargs)

        def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            logits = self.specialized.sample_logits(hidden_states)
            if logits is None:
                return super().compute_sample_logits(hidden_states)
            return logits

    SpecializedModelState.__name__ = f"Specialized{base.__name__}"
    SpecializedModelState.__qualname__ = SpecializedModelState.__name__
    return SpecializedModelState
