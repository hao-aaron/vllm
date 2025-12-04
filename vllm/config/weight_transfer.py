from typing import Literal

from dataclasses import dataclass
from vllm.config.utils import config


@config
@dataclass
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "rdma"] = "nccl"
    """The backend to use for weight transfer."""