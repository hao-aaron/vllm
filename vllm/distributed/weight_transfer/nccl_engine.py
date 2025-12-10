# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based weight transfer engine."""

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
)


class NCCLWeightTransferEngine(WeightTransferEngine):
    """
    Weight transfer engine using NCCL for communication between trainer and workers.

    This implementation uses NCCL broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group.
    """

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

    def init_transfer(  # type: ignore[override]
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        **kwargs,
    ) -> None:
        """
        Initialize NCCL process group with the trainer.

        Args:
            master_address: Address of the master process
            master_port: Port of the master process
            rank_offset: Offset to add to worker rank to get global rank
            world_size: Total number of processes in the process group
            **kwargs: Additional unused parameters
        """
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Calculate the global rank in the trainer-worker process group
        worker_rank = self.parallel_config.rank
        rank = worker_rank + rank_offset

        # Create stateless process group
        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=rank,
            world_size=world_size,
        )

        # Initialize NCCL communicator
        self.model_update_group = PyNcclCommunicator(
            pg, device=torch.cuda.current_device()
        )

    def receive_weights(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        **kwargs,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from trainer via NCCL broadcast.

        Args:
            names: List of parameter names
            dtype_names: List of dtype names (e.g., "float32", "bfloat16")
            shapes: List of parameter shapes
            **kwargs: Additional parameters

        Returns:
            List of (name, weight_tensor) tuples
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. Call init_transfer() first."
            )

        weights = []
        for name, dtype_name, shape in zip(names, dtype_names, shapes):
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
