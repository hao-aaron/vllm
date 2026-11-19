# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash low-latency decode: an executor for FULL-graph decode steps over
the general backbone, at TP4 on SM100/SM103. Other steps run the general layers.
"""

from dataclasses import dataclass, field

from vllm.specialized.declare import RouteConfig, Support


@dataclass(frozen=True)
class OPTIONS:
    """--specialized-model-options for this directory."""

    # Overrides of the per-token-count dispatch by stage name (see `policy.py`),
    # e.g. {"moe": "1-4:per_route,8-16:grouped"}.
    policy: dict[str, str] = field(default_factory=dict)


# The model geometry the kernels are specialized for: hidden size, KDA heads and
# head dim, conv width, MLA heads / latent / NoPE dims, indexer, mHC streams, MoE.
GEOMETRY = {
    "hidden_size": 4096,
    "linear_num_heads": 64,
    "linear_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "index_topk": 2048,
    "index_kpool": 4,
    "hc_mult": 4,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "moe_intermediate_size": 2048,
    "mhc": True,
}


def check(cfg: RouteConfig) -> list[str]:
    """Settings the `Support` fields don't cover."""
    s = cfg.settings
    out = []
    quant = getattr(s.model_config.hf_config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = quant.to_dict()
    if quant.get("weight_block_size") != [128, 128]:
        out.append("checkpoint is not native block-FP8 (128x128)")
    backend = s.attention_config.backend_per_kind.get("mla_attention")
    if backend is not None and backend.name != "FLASHINFER_MLA_SPARSE":
        out.append(f"MLA backend {backend.name}, needs FLASHINFER_MLA_SPARSE")
    if s.parallel_config.use_sequence_parallel_moe:
        out.append("sequence-parallel MoE")
    if s.cache_config.kv_cache_dtype_skip_layers:
        out.append("mixed KV cache dtypes")
    if s.attention_config.hisparse_config is not None:
        out.append("HiSparse")
    # The KDA decode keeps a BF16 conv state and an FP32 recurrent state in place.
    if s.cache_config.mamba_cache_dtype not in ("auto", "bfloat16") or (
        s.cache_config.mamba_ssm_cache_dtype not in ("auto", "float32")
    ):
        out.append("KDA cache dtypes other than BF16 conv / FP32 recurrent")
    if s.cache_config.use_replayssm or s.cache_config.use_kda_recoverssm:
        out.append("ReplaySSM / RecoverSSM")
    return out


SUPPORTS = Support(
    arch={"sm100", "sm103"},
    libraries={"nvidia-cutlass-dsl": "4.0"},
    checkpoint={"fp8"},
    model_shape=GEOMETRY,
    dtype={"bfloat16"},
    kv_cache_dtype={"fp8", "fp8_e4m3"},
    attention_backend={None, "FLASHINFER_MLA_SPARSE"},
    tp={4},
    pp={1},
    dp={1},
    ep={1},
    dcp={1},
    pcp={1},
    micro_batching=False,
    # Eager and PIECEWISE steps run the general layers; FULL decode graphs run this.
    graph_mode={"FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "FULL"},
    compile=False,
    speculation={None, "mtp"},
    num_speculative_tokens=set(range(8)),
    role={"colocated"},
    # Multimodal prefill runs the general layers; decode is text only.
    features={"multimodal"},
    default_on=True,
    check=check,
)


# The VL wrapper stays general and builds this text model through the registry.
MODEL = {"Glm5NextForCausalLM": "model:LowLatencyGlm5Next"}
DRAFT_MODEL = {"mtp": "draft:LowLatencyGlm5NextMTP"}
