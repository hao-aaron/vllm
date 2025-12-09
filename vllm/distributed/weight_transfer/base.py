# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

from abc import ABC, abstractmethod
from typing import Any, Type

from dataclasses import dataclass
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.config.parallel import ParallelConfig


@dataclass
class WeightUpdateRequest(ABC):
    names: list[str]
    dtype_names: list[str]
    shapes: list[tuple]

@dataclass
class WeightTransferInitInfo(ABC):
    pass
class WeightTransferEngine(ABC):
    """
    Base class for weight transfer engines that handle transport of model weights
    from a trainer to inference workers.

    This abstraction separates weight transfer transport logic from the worker
    implementation, allowing different backends (NCCL, CUDA IPC, RDMA) to be
    plugged in.
    """

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        self.config = config
        self.parallel_config = parallel_config
    
    @property
    def init_info_cls(self) -> Type[WeightTransferInitInfo]:
        """
        Get the class for the weight transfer init info.
        """
        raise NotImplementedError
    
    @property
    def update_request_cls(self) -> Type[WeightUpdateRequest]:
        """
        Get the class for the weight update request.
        """
        raise NotImplementedError

    @abstractmethod
    def init_transfer(self, init_info: WeightTransferInitInfo) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.

        Args:
            init_info: WeightTransferInitInfo
        """
        raise NotImplementedError

    @abstractmethod
    def receive_weights(
        self, request: WeightUpdateRequest
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from the trainer.

        Args:
            request: WeightUpdateRequest

        Returns:
            List of (name, weight_tensor) tuples ready to be loaded into the model
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the weight transfer engine.
        This should be called when the worker is shutting down.
        """
        raise NotImplementedError