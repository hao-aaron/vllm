# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.config.parallel import ParallelConfig


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

    @abstractmethod
    def init_transfer(self, **kwargs: Any) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.

        Args:
            **kwargs: Backend-specific initialization arguments
                For NCCL: master_address, master_port, rank_offset, world_size
                For IPC: (no args needed)
                For RDMA: rdma_specific_args
        """
        raise NotImplementedError

    @abstractmethod
    def receive_weights(
        self, names: list[str], dtype_names: list[str], shapes: list[tuple], **kwargs: Any
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from the trainer.

        Args:
            names: List of weight parameter names
            dtype_names: List of dtype names (e.g., ['float32', 'float16'])
            shapes: List of weight shapes
            **kwargs: Backend-specific arguments
                For NCCL: (no additional args)
                For IPC: ipc_handles
                For RDMA: rdma_handles

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