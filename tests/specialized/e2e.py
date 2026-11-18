# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared end-to-end check: a pinned specialized directory against the general
model on the same checkpoint, TP4."""

import os

import pytest
import torch

from tests.models.utils import check_logprobs_close

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "List the first ten prime numbers:",
    "Question: What is 17 * 23? Answer:",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
]


def _generate(model: str, specialized: str, chat: dict | None, **kwargs):
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=model,
        tensor_parallel_size=4,
        specialized_model=specialized,
        max_model_len=4096,
        **kwargs,
    )
    choice = llm.llm_engine.vllm_config.model_config.specialized_choice
    params = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
    if chat is None:
        outs = llm.generate(PROMPTS, params)
    else:  # chat models degenerate on raw completions
        messages = [[{"role": "user", "content": p}] for p in PROMPTS]
        outs = llm.chat(messages, params, chat_template_kwargs=chat)
    results = [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outs
    ]
    del llm
    cleanup_dist_env_and_memory()
    return choice, results


def check_against_general(
    env_var: str, name: str, chat: dict | None = None, **kwargs
) -> None:
    model = os.environ.get(env_var)
    if not model or torch.accelerator.device_count() < 4:
        pytest.skip(f"needs 4 GPUs and ${env_var} pointing at the checkpoint")
    choice, fast = _generate(model, name, chat, **kwargs)
    assert choice.endswith("." + name)
    general_choice, ref = _generate(model, "off", chat, **kwargs)
    assert general_choice == ""
    # Different kernels round differently; vLLM's usual greedy check: where the
    # tokens diverge, each must be in the other's top logprobs.
    check_logprobs_close(
        outputs_0_lst=fast, outputs_1_lst=ref, name_0=name, name_1="general"
    )
