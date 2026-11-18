# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash low_latency against the general model (TP4, SM100/SM103)."""

from tests.specialized.e2e import check_against_general


def test_matches_general():
    check_against_general(
        "GLM_5_3_FLASH",
        "low_latency",
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
