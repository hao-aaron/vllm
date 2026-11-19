# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Core internals this directory still reaches directly (Rule 1).

Kept exact by tests/specialized/test_rules.py; each entry is a missing piece
of `vllm.specialized`, and each MUTATIONS entry a write to be removed."""

EXCEPTIONS = {
    "vllm._custom_ops": "fused CUDA op in the core extension",
    "vllm.forward_context": "step metadata; needs prepare_inputs",
    "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm": "shared layer ops",
    "vllm.model_executor.layers.fused_moe.moe_output": "MoE output types",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner": "MoE output types",
    "vllm.model_executor.layers.fusion.quant_activation": "shared layer ops",
    "vllm.model_executor.layers.quantization.utils.quant_utils": "shared layer ops",
    "vllm.sequence": "IntermediateTensors type",
    "vllm.third_party.fmha_sm100.api": "vendored attention kernels",
    "vllm.triton_utils": "triton import shim",
    "vllm.v1.worker.gpu.sample.gumbel": "sampler internals (Markov sampler)",
    "vllm.v1.worker.gpu.spec_decode.draft_logits": "speculator internals (Markov sampler)",
}

MUTATIONS = {
    "_cd.CuTeDSL._generate_kernel_attrs = ...": "patches CuTe DSL library internals",
    "hidden_norm.forward = ...": "patches forward of a general module",
    "module.forward = ...": "patches forward of a general module",
    "norm.forward = ...": "patches forward of a general module",
}
