# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based weight transfer engine."""

from dataclasses import dataclass

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    BackendInitInfo,
    BackendUpdateInfo,
    WeightTransferEngine,
)


@dataclass
class NCCLInitInfo(BackendInitInfo):
    """Initialization info for NCCL weight transfer backend."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


@dataclass
class NCCLUpdateInfo(BackendUpdateInfo):
    """Update info for NCCL weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

    def __post_init__(self):
        """Validate that all lists have the same length."""
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )


class NCCLWeightTransferEngine(WeightTransferEngine[NCCLInitInfo, NCCLUpdateInfo]):
    """
    Weight transfer engine using NCCL for communication between trainer and workers.

    This implementation uses NCCL broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group.
    """

    # Define backend-specific dataclass types
    init_info_cls = NCCLInitInfo
    update_info_cls = NCCLUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        """
        Initialize the NCCL weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        super().__init__(config, parallel_config)
        self.model_update_group = None

    def init_transfer(self, init_info: NCCLInitInfo) -> None:
        """
        Initialize NCCL process group with the trainer.

        Args:
            init_info: NCCL initialization info containing master address, port,
                      rank offset, and world size
        """
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Calculate the global rank in the trainer-worker process group
        worker_rank = self.parallel_config.rank
        rank = worker_rank + init_info.rank_offset

        # Create stateless process group
        pg = StatelessProcessGroup.create(
            host=init_info.master_address,
            port=init_info.master_port,
            rank=rank,
            world_size=init_info.world_size,
        )

        # Initialize NCCL communicator
        self.model_update_group = PyNcclCommunicator(
            pg, device=torch.cuda.current_device()
        )

    def receive_weights(
        self, update_info: NCCLUpdateInfo
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from trainer via NCCL broadcast.

        Args:
            update_info: NCCL update info containing parameter names, dtypes, and shapes

        Returns:
            List of (name, weight_tensor) tuples
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. Call init_transfer() first."
            )

        weights = []
        for name, dtype_name, shape in zip(
            update_info.names, update_info.dtype_names, update_info.shapes
        ):
            # Get the torch dtype
            dtype = getattr(torch, dtype_name)

            # Allocate buffer for receiving weight
            weight = torch.empty(shape, dtype=dtype, device="cuda")

            # Broadcast from rank 0 (trainer)
            self.model_update_group.broadcast(
                weight, src=0, stream=torch.cuda.current_stream()
            )

            weights.append((name, weight))

        return weights

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.destroy()
