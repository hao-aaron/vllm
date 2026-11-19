# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 low_latency against the general model (TP4, SM100/SM103)."""

from tests.specialized.e2e import check_against_general


def test_matches_general():
    # Pinned: the directory is not on by default. Both arms get the settings it
    # requires, so the comparison is on the same deployment.
    check_against_general(
        "MINIMAX_M3_1_FLASH",
        "low_latency",
        chat={"reasoning_effort": "low"},
        kv_cache_dtype="fp8",
        block_size=128,
        attention_config={"indexer_kv_dtype": "fp8"},
        additional_config={"minimax_m3_fused_finalize_max_tokens": 64},
    )
