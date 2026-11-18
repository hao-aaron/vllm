# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 low-latency decode: an executor for FULL-graph decode steps of up
to `MAX_ROWS` rows over the general backbone, at TP4 on SM100/SM103, with a
fused DSpark drafter. Other steps run the general layers.
"""

from vllm.specialized.declare import RouteConfig, Support

from .config import (
    INDEXER_KV_DTYPES,
    KV_CACHE_DTYPES,
    MAX_PAGES,
    MAX_SPECULATIVE_TOKENS,
    PAGE_SIZE,
    STEP_LEVEL_KV_CONNECTORS,
    SUPPORTED_TP_SIZES,
    max_rows,
)


def check(cfg: RouteConfig) -> list[str]:
    """Settings the `Support` fields don't cover."""
    s = cfg.settings
    out = []
    quant = getattr(s.model_config.hf_config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = quant.to_dict()
    if quant.get("moe_quant_algo") != "NVFP4":
        out.append("checkpoint without NVFP4 routed experts")
    if s.kernel_config.moe_backend == "deep_gemm_mega_moe":
        out.append("MegaMoE backend (needs the default TRTLLM-gen MoE)")
    if s.scheduler_config.async_scheduling is False:
        out.append("async scheduling off")
    attention = s.attention_config
    if attention.indexer_kv_dtype not in INDEXER_KV_DTYPES:
        out.append(f"indexer_kv_dtype {attention.indexer_kv_dtype}, needs fp8")
    if attention.indexer_decode_cp or attention.minimax_m3_indexer_cp:
        out.append("indexer context parallelism")
    spec = s.speculative_config
    if spec is not None and spec.enable_adaptive_verification:
        out.append("adaptive verification")
    kv = s.kv_transfer_config
    if kv is not None and kv.kv_connector not in STEP_LEVEL_KV_CONNECTORS:
        out.append(f"KV connector {kv.kv_connector} has per-layer hooks")
    sizes = s.compilation_config.cudagraph_capture_sizes
    cap = max_rows(1 + cfg.num_speculative_tokens)
    if sizes is not None and not any(size <= cap for size in sizes):
        out.append(f"no CUDA graph capture size <= {cap} rows")
    return out


SUPPORTS = Support(
    arch={"sm100", "sm103"},
    libraries={"nvidia-cutlass-dsl": "4.0"},
    checkpoint={"mxfp8", "modelopt_mxfp8"},
    kv_cache_dtype=set(KV_CACHE_DTYPES),
    tp=set(SUPPORTED_TP_SIZES),
    pp={1},
    dp={1},
    ep={1},
    dcp={1},
    pcp={1},
    micro_batching=False,
    graph_mode={"FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"},
    compile=False,
    speculation={None, "dspark"},
    num_speculative_tokens=set(range(MAX_SPECULATIVE_TOKENS + 1)),
    block_size={PAGE_SIZE},
    max_model_len=range(1, MAX_PAGES * PAGE_SIZE + 1),
    role={"colocated"},
    features={"multimodal", "kv_transfer"},
    # Not yet validated for `auto`: runs only when pinned or required.
    default_on=False,
    check=check,
)


def defaults(cfg: RouteConfig) -> dict:
    return {
        "compilation_config.cudagraph_mode": "FULL_AND_PIECEWISE",
        "cache_config.block_size": PAGE_SIZE,
        "attention_config.indexer_kv_dtype": "fp8",
    }


# The VL wrapper stays general; its language model is the specialized class.
MODEL = {
    "MiniMaxM3SparseForConditionalGeneration": "language_model",
    "MiniMaxM3SparseForCausalLM": "model:LowLatencyMiniMaxM3",
}
DRAFT_MODEL = {"dspark": "draft:LowLatencyMiniMaxM3DSpark"}
