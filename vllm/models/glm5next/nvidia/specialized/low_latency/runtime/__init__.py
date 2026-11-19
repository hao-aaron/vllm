# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash low-latency decode runtime settings."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    max_context: int
    # Tokens per request in a verify step (1 + draft tokens; 1 without speculation).
    q: int
    capture_sizes: tuple[int, ...]
    # Overrides of the per-token-count dispatch (see ``policy.DEFAULT``).
    policy: tuple[tuple[str, str], ...] = ()


def decode_config(vllm_config) -> DecodeConfig:
    """The executor settings for a deployment routing already accepted."""
    from vllm.specialized import options

    spec = vllm_config.speculative_config
    return DecodeConfig(
        max_context=vllm_config.model_config.max_model_len,
        q=1 + spec.num_speculative_tokens if spec is not None else 1,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
        policy=tuple(options(vllm_config).policy.items()),
    )
