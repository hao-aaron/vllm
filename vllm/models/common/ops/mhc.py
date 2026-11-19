# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC stream expand and contract, shared by the general mHC layers and
specialized decodes."""

import torch

from .contract import Cell, SharedOp, shared_op


def _streams(dtype: torch.dtype, n: int, hidden: int):
    def make(device: torch.device):
        return (torch.randn(5, n, hidden, device=device).to(dtype), n)

    return make


def _rows(dtype: torch.dtype, n: int, hidden: int):
    def make(device: torch.device):
        return (torch.randn(5, hidden, device=device).to(dtype), n)

    return make


@shared_op
class HcExpand(SharedOp):
    """[s, hidden] -> [s, n, hidden] by replication."""

    SUPPORTS = {"dtypes": ("bfloat16", "float32")}
    CELLS = (
        Cell("GLM-5.3 mHC, 4 streams", _rows(torch.bfloat16, 4, 64)),
        Cell("fp32 residual", _rows(torch.float32, 2, 32)),
    )
    ARCHS = frozenset({"cpu"})

    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x.unsqueeze(1).expand(-1, n, -1).contiguous()

    def reference(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return torch.stack([x] * n, dim=1)


@shared_op
class HcContract(SharedOp):
    """[s, n, hidden] -> [s, hidden] by averaging the streams."""

    SUPPORTS = {"dtypes": ("bfloat16", "float32")}
    CELLS = (
        Cell("GLM-5.3 mHC, 4 streams", _streams(torch.bfloat16, 4, 64)),
        Cell("fp32 residual", _streams(torch.float32, 2, 32)),
    )
    ARCHS = frozenset({"cpu"})

    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x.mean(dim=1)

    def reference(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return (x.float().sum(dim=1) / n).to(x.dtype)


hc_expand = HcExpand
hc_contract = HcContract
