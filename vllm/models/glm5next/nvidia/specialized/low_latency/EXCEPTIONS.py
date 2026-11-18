# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core internals this directory still reaches directly (Rule 1).

Kept exact by tests/specialized/test_rule1_lint.py; each entry is a
missing piece of `vllm.specialized`."""

EXCEPTIONS = {
    "vllm.config": "reads engine settings at build; needs RouteConfig/settings handed to Model",
    "vllm.forward_context": "per-step attention metadata and capture state; needs a prepare_inputs hook",
    "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16": "shared CuTe DSL GEMM kernels; candidate shared ops",
    "vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm": "shared CuTe DSL GEMM kernels; candidate shared ops",
    "vllm.model_executor.layers.fused_moe.moe_output": "MoE output types shared with the general layers",
    "vllm.model_executor.layers.mamba.mamba_utils": "shared layer ops/types the executor reuses",
    "vllm.model_executor.layers.mhc": "shared layer ops/types the executor reuses",
    "vllm.model_executor.layers.quantization.utils.flashinfer_utils": "shared layer ops/types the executor reuses",
    "vllm.model_executor.layers.sparse_attn_indexer": "shared layer ops/types the executor reuses",
    "vllm.platforms": "device queries; belongs in RouteConfig probes",
    "vllm.utils.deep_gemm": "utility helpers (DeepGEMM flags, streams)",
    "vllm.utils.torch_utils": "utility helpers (DeepGEMM flags, streams)",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse": "reads the general attention backend's metadata and impl internals",
    "vllm.v1.attention.backends.mla.index_group": "reads the general attention backend's metadata and impl internals",
    "vllm.v1.attention.backends.mla.indexer": "reads the general attention backend's metadata and impl internals",
}

MUTATIONS = {
    "indexer.forward = ...": "replaces forward on a general-tier module",
    "module.quant_method = ...": "swaps quant_method on a general linear layer",
    "sparse_indexer.SPECIALIZED_KPOOL = ...": "writes a general-tier module global the general indexer reads",
    "w.forward = ...": "replaces forward on a general-tier module",
}
