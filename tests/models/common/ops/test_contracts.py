# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every shared op matches its reference on every declared cell."""

import pytest
import torch

import vllm.models.common.ops.mhc  # noqa: F401  (registers its ops)
from vllm.models.common.ops.contract import REGISTRY

CASES = [(op, cell) for op in REGISTRY for cell in op.CELLS]


@pytest.mark.parametrize(
    "op,cell", CASES, ids=[f"{type(o).__name__}-{c.reason}" for o, c in CASES]
)
def test_matches_reference(op, cell):
    device = torch.device("cuda" if "cpu" not in op.ARCHS else "cpu")
    args = cell.args(device)
    torch.testing.assert_close(op(*args), op.reference(*args))
