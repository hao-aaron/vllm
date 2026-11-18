# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 low_latency against the general model (TP4, SM100/SM103)."""

from tests.specialized.e2e import check_against_general


def test_matches_general():
    # Pinned: the directory is not on by default; its defaults fill the block
    # size and index cache dtype.
    check_against_general(
        "MINIMAX_M3_1_FLASH",
        "low_latency",
        chat={"reasoning_effort": "low"},
        kv_cache_dtype="fp8",
    )
