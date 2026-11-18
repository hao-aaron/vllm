# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared end-to-end check: a pinned specialized directory against the general
model on the same checkpoint, TP4."""

import os

import pytest
import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "List the first ten prime numbers:",
    "Question: What is 17 * 23? Answer:",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
]


def _generate(model: str, specialized: str, **kwargs) -> tuple[str, list[list[int]]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=4,
        specialized_model=specialized,
        max_model_len=4096,
        **kwargs,
    )
    choice = llm.llm_engine.vllm_config.model_config.specialized_choice
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=32))
    tokens = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    from vllm.distributed import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
    return choice, tokens


def check_against_general(env_var: str, name: str, **kwargs) -> None:
    model = os.environ.get(env_var)
    if not model or torch.accelerator.device_count() < 4:
        pytest.skip(f"needs 4 GPUs and ${env_var} pointing at the checkpoint")
    choice, fast = _generate(model, name, **kwargs)
    assert choice.endswith("." + name)
    general_choice, ref = _generate(model, "off", **kwargs)
    assert general_choice == ""
    # Different kernels round differently: require the greedy prefixes to agree.
    agree = sum(f[:8] == r[:8] for f, r in zip(fast, ref))
    assert agree >= len(PROMPTS) - 1, (fast, ref)
