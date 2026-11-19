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
)


def check(cfg: RouteConfig) -> list[str]:
    """Settings the `Support` fields don't cover."""
    s = cfg.settings
    out = []
    if s.scheduler_config.async_scheduling is False:
        out.append("async scheduling off")
    attention = s.attention_config
    if attention.indexer_kv_dtype not in INDEXER_KV_DTYPES:
        out.append(
            f"indexer_kv_dtype {attention.indexer_kv_dtype}, needs fp8 "
            "(set -ac.indexer_kv_dtype fp8)"
        )
    if attention.indexer_decode_cp or attention.minimax_m3_indexer_cp:
        out.append("indexer context parallelism")
    spec = s.speculative_config
    if spec is not None and spec.enable_adaptive_verification:
        out.append("adaptive verification")
    kv = s.kv_transfer_config
    if kv is not None and kv.kv_connector not in STEP_LEVEL_KV_CONNECTORS:
        out.append(f"KV connector {kv.kv_connector} has per-layer hooks")
    return out


SUPPORTS = Support(
    arch={"sm100", "sm103"},
    libraries={"nvidia-cutlass-dsl": "4.0"},
    checkpoint={"modelopt_mxfp8"},
    expert_quantization={"nvfp4"},
    kv_cache_dtype=set(KV_CACHE_DTYPES),
    # MegaMoE replaces the TRTLLM-gen experts the executor reads.
    moe_backend={"auto", "flashinfer_trtllm"},
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


# The VL wrapper stays general and builds this text model through the registry.
MODEL = {"MiniMaxM3SparseForCausalLM": "model:LowLatencyMiniMaxM3"}
DRAFT_MODEL = {"dspark": "draft:LowLatencyMiniMaxM3DSpark"}
