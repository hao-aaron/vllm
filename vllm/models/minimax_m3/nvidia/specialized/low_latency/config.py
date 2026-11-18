# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Limits and settings of the MiniMax-M3.1 low-latency decode executor.

It runs one precision setup: fp8 (E4M3) KV and index-K caches, MXFP8 attention
and dense linears, NVFP4 routed experts. Each stage declares the cache dtypes it
runs, so more setups can be added stage by stage.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Tensor parallel sizes the executor runs at. TP8 and TP16 change the per-rank head
# counts (and the MoE intermediate size) every stage is written for.
SUPPORTED_TP_SIZES = (4,)
# Most DSpark draft tokens per request: a verify step has K + 1 rows a request.
MAX_SPECULATIVE_TOKENS = 7
# Most requests of an executor step.
MAX_REQUESTS = 8
# Rows of the largest executor step: MAX_REQUESTS x (MAX_SPECULATIVE_TOKENS + 1). Larger
# FULL decode graphs run the stock layers.
MAX_ROWS = MAX_REQUESTS * (MAX_SPECULATIVE_TOKENS + 1)
# The KV / index page size the kernels are written for, and the most pages a request may
# span (the page top-k's column limit, kernels/page_topk.MAX_COLS).
PAGE_SIZE = 128
MAX_PAGES = 8192
# --kv-cache-dtype values the executor accepts, by the dtype it reads.
KV_CACHE_DTYPES: dict[str, "CacheDtype"] = {"fp8": "fp8", "fp8_e4m3": "fp8"}
# attention_config.indexer_kv_dtype values the executor accepts (the MiniMax-M3 default,
# "auto", is BF16).
INDEXER_KV_DTYPES: dict[str, "CacheDtype"] = {"fp8": "fp8", "fp8_e4m3": "fp8"}

# KV connectors the executor runs with. A connector's per-layer hooks
# (`wait_for_layer_load`, `save_kv_layer`) wrap the stock attention layers,
# which the executor's steps skip; these connectors' per-layer hooks are no-ops
# and they move KV around the forward instead (`start_load_kv`, `wait_for_save`).
STEP_LEVEL_KV_CONNECTORS = ("SimpleCPUOffloadConnector",)

# The dtypes a cache can have in a DecodeConfig. Only fp8 is accepted today; BF16 is
# named so that stages can already declare it.
CacheDtype = Literal["bf16", "fp8"]

# The stage slots of ``stages.StepStages``, in order (checked there).
STAGE_NAMES = (
    "front",
    "index_score",
    "page_topk",
    "sparse_decode",
    "dense_decode",
    "o_proj",
    "allreduce_norm",
    "moe",
    "dense_mlp",
)


class CacheDtypes(NamedTuple):
    """The dtypes of the caches the attention stages read and write."""

    # The main K/V cache.
    kv: CacheDtype
    # The indexer's index-K cache.
    index: CacheDtype  # type: ignore[assignment]


def cache_dtypes(vllm_config: "VllmConfig") -> CacheDtypes | None:
    """The deployment's cache dtypes, or None if the executor refuses them
    (NVFP4, BF16, ...)."""
    kv = KV_CACHE_DTYPES.get(str(vllm_config.cache_config.cache_dtype))
    index = INDEXER_KV_DTYPES.get(str(vllm_config.attention_config.indexer_kv_dtype))
    if kv is None or index is None:
        return None
    return CacheDtypes(kv=kv, index=index)


@dataclass(frozen=True)
class DecodeConfig:
    # The KV and index-K cache dtypes; stage implementations declare which they run
    # (``Stage.cache_dtypes``).
    cache_dtypes: CacheDtypes
    # Rows a request occupies in a decode step: 1 + the DSpark draft tokens.
    query_len: int
    # The FULL decode CUDA graph sizes the executor runs (every step's rows).
    step_rows: tuple[int, ...] = ()


def max_rows(query_len: int) -> int:
    """Rows of the largest executor step at `query_len` rows a request: whole
    requests of up to `MAX_REQUESTS`, at most `MAX_ROWS`."""
    if query_len < 1:
        raise ValueError(f"query_len must be positive, got {query_len}")
    return min(MAX_ROWS, MAX_REQUESTS * query_len)


def decode_config(vllm_config: "VllmConfig") -> DecodeConfig:
    """The executor settings for a deployment routing already accepted."""
    spec = vllm_config.speculative_config
    query_len = 1 + (0 if spec is None else int(spec.num_speculative_tokens or 0))
    dtypes = cache_dtypes(vllm_config)
    assert dtypes is not None
    cap = max_rows(query_len)
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    step_rows = tuple(sorted({size for size in sizes if 0 < size <= cap}))
    logger.info_once(
        "MiniMax-M3.1 low-latency decode for FULL-graph decode steps of up to %d "
        "rows (%d rows a request; %s KV cache, %s index cache).",
        cap,
        query_len,
        dtypes.kv,
        dtypes.index,
    )
    return DecodeConfig(cache_dtypes=dtypes, query_len=query_len, step_rows=step_rows)
