# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rule 2: who runs what. Each family's declarations are evaluated over a grid
of deployments and compared to the stored table below, so a change in routing
shows up in review. Tuning knobs (max_num_seqs) must not change the result."""

import dataclasses
import importlib
from types import SimpleNamespace as NS

import pytest

from vllm.specialized.declare import RouteConfig

GLM = "vllm.models.glm5next"
MM = "vllm.models.minimax_m3"


def _settings(quant: dict, **over):
    s = NS(
        model_config=NS(hf_config=NS(quantization_config=quant)),
        attention_config=NS(
            backend_per_kind={},
            hisparse_config=None,
            indexer_kv_dtype="fp8",
            indexer_decode_cp=False,
            minimax_m3_indexer_cp=False,
        ),
        parallel_config=NS(use_sequence_parallel_moe=False),
        cache_config=NS(
            kv_cache_dtype_skip_layers=[],
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            use_replayssm=False,
            use_kda_recoverssm=False,
        ),
        kernel_config=NS(moe_backend="auto"),
        scheduler_config=NS(async_scheduling=True, max_num_seqs=256),
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=NS(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64]),
    )
    for path, value in over.items():
        obj = s
        *parts, last = path.split(".")
        for p in parts:
            obj = getattr(obj, p)
        setattr(obj, last, value)
    return s


def _base(family: str) -> RouteConfig:
    if family == GLM:
        geometry = importlib.import_module(
            f"{GLM}.nvidia.specialized.low_latency"
        ).GEOMETRY
        hf, ckpt, quant = NS(**geometry), "fp8", {"weight_block_size": [128, 128]}
    else:
        hf, ckpt, quant = NS(), "mxfp8", {"moe_quant_algo": "NVFP4"}
    return RouteConfig(
        platform="cuda",
        arch="sm100",
        compute_units=None,
        machine=frozenset(),
        libraries={"nvidia-cutlass-dsl": "4.3.0"},
        dtype="bfloat16",
        max_model_len=8192,
        checkpoint=ckpt,
        hf_config=hf,
        kv_cache_dtype="fp8",
        attention_backend=None,
        tp=4,
        pp=1,
        dp=1,
        ep=1,
        dcp=1,
        pcp=1,
        micro_batching=False,
        speculation=None,
        num_speculative_tokens=0,
        graph_mode="FULL_AND_PIECEWISE",
        compile=False,
        role="colocated",
        features=frozenset(),
        block_size=128,
        settings=_settings(quant),
    )


def _route(family: str, cfg: RouteConfig, pinned: bool) -> str:
    """The first accepted directory, or the general model."""
    from vllm.specialized._routing import candidates

    for name in candidates(family, cfg.platform):
        support = importlib.import_module(name).SUPPORTS
        if not pinned and not support.default_on:
            continue
        if not support.reasons(cfg):
            return name.rsplit(".", 1)[-1]
    return "general"


# (family, label, RouteConfig overrides, settings overrides, pinned) -> choice
TABLE = [
    (GLM, "base", {}, {}, False, "low_latency"),
    (GLM, "gb300", {"arch": "sm103"}, {}, False, "low_latency"),
    (GLM, "hopper", {"arch": "sm90"}, {}, False, "general"),
    (GLM, "tp8", {"tp": 8}, {}, False, "general"),
    (GLM, "bf16 kv", {"kv_cache_dtype": "auto"}, {}, False, "general"),
    (
        GLM,
        "mtp3",
        {"speculation": "mtp", "num_speculative_tokens": 3},
        {},
        False,
        "low_latency",
    ),
    (
        GLM,
        "eagle3",
        {"speculation": "eagle3", "num_speculative_tokens": 3},
        {},
        False,
        "general",
    ),
    (GLM, "eager", {"graph_mode": "NONE"}, {}, False, "general"),
    (GLM, "lora", {"features": frozenset({"lora"})}, {}, False, "general"),
    (
        GLM,
        "weight reload",
        {"features": frozenset({"weight_reload"})},
        {},
        False,
        "general",
    ),
    (
        GLM,
        "multimodal",
        {"features": frozenset({"multimodal"})},
        {},
        False,
        "low_latency",
    ),
    (GLM, "other geometry", {"hf_config": NS(hidden_size=6144)}, {}, False, "general"),
    (GLM, "replayssm", {}, {"cache_config.use_replayssm": True}, False, "general"),
    (
        GLM,
        "old cutedsl",
        {"libraries": {"nvidia-cutlass-dsl": "3.9"}},
        {},
        False,
        "general",
    ),
    (MM, "base auto", {}, {}, False, "general"),
    (MM, "base pinned", {}, {}, True, "low_latency"),
    (
        MM,
        "dspark7",
        {"speculation": "dspark", "num_speculative_tokens": 7},
        {},
        True,
        "low_latency",
    ),
    (
        MM,
        "dspark8",
        {"speculation": "dspark", "num_speculative_tokens": 8},
        {},
        True,
        "general",
    ),
    (MM, "block 64", {"block_size": 64}, {}, True, "general"),
    (MM, "nvfp4 kv", {"kv_cache_dtype": "nvfp4"}, {}, True, "general"),
    (
        MM,
        "bf16 indexer",
        {},
        {"attention_config.indexer_kv_dtype": "auto"},
        True,
        "general",
    ),
    (
        MM,
        "cpu offload",
        {"features": frozenset({"kv_transfer"})},
        {"kv_transfer_config": NS(kv_connector="SimpleCPUOffloadConnector")},
        True,
        "low_latency",
    ),
    (
        MM,
        "nixl",
        {"features": frozenset({"kv_transfer"})},
        {"kv_transfer_config": NS(kv_connector="NixlConnector")},
        True,
        "general",
    ),
    (MM, "sleep mode", {"features": frozenset({"sleep_mode"})}, {}, True, "general"),
    (
        MM,
        "megamoe",
        {},
        {"kernel_config.moe_backend": "deep_gemm_mega_moe"},
        True,
        "general",
    ),
    (
        MM,
        "no small graphs",
        {},
        {"compilation_config.cudagraph_capture_sizes": [128]},
        True,
        "general",
    ),
]


@pytest.mark.parametrize("max_num_seqs", [1, 256])
@pytest.mark.parametrize(
    "family,label,over,settings_over,pinned,expected",
    TABLE,
    ids=[f"{f.rsplit('.', 1)[-1]}-{label}" for f, label, *_ in TABLE],
)
def test_routing_table(
    family, label, over, settings_over, pinned, expected, max_num_seqs
):
    base = _base(family)
    quant = base.settings.model_config.hf_config.quantization_config
    settings = _settings(
        quant, **settings_over, **{"scheduler_config.max_num_seqs": max_num_seqs}
    )
    cfg = dataclasses.replace(base, settings=settings, **over)
    assert _route(family, cfg, pinned) == expected
