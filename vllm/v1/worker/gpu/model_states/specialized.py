# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pass-through ModelState for a specialized `Model`: the general model's state
class, with the model's own hooks run after each default."""

import functools
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_states.interface import ModelState

logger = init_logger(__name__)

_REPORT_AT = frozenset({10, 100, 1000, 10_000})


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

        def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
            out = super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)
            if not kwargs.get("for_capture", False):
                self._step_path = self.specialized.step_path(
                    cudagraph_mode == CUDAGraphMode.FULL,
                    input_batch.num_tokens_after_padding,
                )
            return out

        def postprocess_state(self, *args, **kwargs):
            super().postprocess_state(*args, **kwargs)
            # Only real steps reach here; dummy and capture runs do not.
            path = getattr(self, "_step_path", None)
            if path is None:
                return
            self._step_path = None
            model = self.specialized
            model.count(path)
            counts = model.path_counts()
            steps = sum(counts.values())
            if steps in _REPORT_AT or steps % 100_000 == 0:
                logger.info("Specialized paths after %d steps: %s", steps, counts)

        def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            logits = self.specialized.sample_logits(hidden_states)
            if logits is None:
                return super().compute_sample_logits(hidden_states)
            return logits

    SpecializedModelState.__name__ = f"Specialized{base.__name__}"
    SpecializedModelState.__qualname__ = SpecializedModelState.__name__
    return SpecializedModelState
