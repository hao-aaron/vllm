# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Core internals this directory still reaches directly (Rule 1).

Kept exact by tests/specialized/test_rules.py; each entry is a missing piece
of `vllm.specialized`, and each MUTATIONS entry a write to be removed."""

EXCEPTIONS = {
    "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16": "CuTe DSL GEMMs; shared-op candidate",
    "vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm": "CuTe DSL GEMMs; shared-op candidate",
    "vllm.model_executor.layers.fused_moe.moe_output": "MoE output types",
    "vllm.model_executor.layers.mamba.mamba_utils": "shared layer ops",
    "vllm.model_executor.layers.quantization.utils.flashinfer_utils": "shared layer ops",
    "vllm.model_executor.layers.sparse_attn_indexer": "shared layer ops",
    "vllm.platforms": "device queries; RouteConfig probe",
    "vllm.utils.deep_gemm": "DeepGEMM flags, streams",
    "vllm.utils.torch_utils": "DeepGEMM flags, streams",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse": "general attention backend internals",
    "vllm.v1.attention.backends.mla.index_group": "general attention backend internals",
    "vllm.v1.attention.backends.mla.indexer": "general attention backend internals",
}

MUTATIONS = {
    "indexer.forward = ...": "patches forward of a general module",
    "module.quant_method = ...": "swaps a general layer's quant_method",
    "w.forward = ...": "patches forward of a general module",
}
