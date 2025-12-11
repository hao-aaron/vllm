# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig


@dataclass
class WeightUpdateRequest:
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    extras: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self):
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`"
                f" got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`"
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        if not isinstance(self.extras, dict):
            raise ValueError("`extras` must be a dictionary")

        for key in self.extras:
            if len(self.extras[key]) != num_params:
                raise ValueError(
                    f"`extras[{key}]` should be of the same size as `names`"
                    f"got {len(self.extras[key])} and {len(self.names)}"
                )


@dataclass
class WeightTransferInitInfo:
    init_kwargs: dict[str, Any] = field(default_factory=dict)


class WeightTransferEngine(ABC):
    """
    Base class for weight transfer engines that handle transport of model weights
    from a trainer to inference workers.

    This abstraction separates weight transfer transport logic from the worker
    implementation, allowing different backends (NCCL, CUDA IPC, RDMA) to be
    plugged in.
    """

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        self.config = config
        self.parallel_config = parallel_config

    @abstractmethod
    def init_transfer(self, **kwargs) -> None:  # noqa: B027
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.
        """
        raise NotImplementedError

    @abstractmethod
    def receive_weights(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        **kwargs,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from the trainer.

        Args:
            names: List of parameter names
            dtype_names: List of dtype names (e.g., "float32", "bfloat16")
            shapes: List of parameter shapes
            **kwargs: Backend-specific parameters for weight transfer

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
