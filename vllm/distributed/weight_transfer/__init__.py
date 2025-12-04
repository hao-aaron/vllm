# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight transfer engines for syncing model weights from trainers to inference workers."""

from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine


WEIGHT_TRANSFER_ENGINE_REGISTRY = {
    "nccl": NCCLWeightTransferEngine,
}

def register_weight_transfer_engine(name: str, engine: type[WeightTransferEngine]) -> None:
    if name in WEIGHT_TRANSFER_ENGINE_REGISTRY:
        raise ValueError(f"Weight transfer engine {name} already registered")
    WEIGHT_TRANSFER_ENGINE_REGISTRY[name] = engine

__all__ = [
    "WeightTransferEngine",
    "NCCLWeightTransferEngine",
    "register_weight_transfer_engine",
]
