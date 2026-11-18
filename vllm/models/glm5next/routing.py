# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized implementations per platform, most specific first."""

SPECIALIZED = {
    "cuda": ["nvidia.specialized.low_latency"],
}
