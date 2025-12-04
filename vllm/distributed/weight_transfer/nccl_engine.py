# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based weight transfer engine."""

from typing import Any

import torch

from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.config.parallel import ParallelConfig

class NCCLWeightTransferEngine(WeightTransferEngine):
    """
    Weight transfer engine using NCCL for communication between trainer and workers.

    This implementation uses NCCL broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group.
    """

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        """
        Initialize the NCCL weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        super().__init__(config, parallel_config)
        self.model_update_group = None

    def init_transfer(self, **kwargs: Any) -> None:
        """
        Initialize NCCL process group with the trainer.

        Required kwargs:
            master_address (str): IP address of the trainer (rank 0)
            master_port (int): Port for the trainer
            rank_offset (int): Rank offset for this worker in the process group
            world_size (int): Total world size including trainer and all workers
            worker_rank (int): The rank of this worker in the inference world group
        """
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        master_address = kwargs["master_address"]
        master_port = kwargs["master_port"]
        rank_offset = kwargs["rank_offset"]
        world_size = kwargs["world_size"]

        # Calculate the global rank in the trainer-worker process group
        worker_rank = self.parallel_config.rank
        rank = worker_rank + rank_offset

        # Create stateless process group
        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )

        # Initialize NCCL communicator
        self.model_update_group = PyNcclCommunicator(pg, device=torch.cuda.current_device())

    def receive_weights(
        self, names: list[str], dtype_names: list[str], shapes: list[tuple], **kwargs: Any
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from trainer via NCCL broadcast.

        Args:
            names: List of weight parameter names
            dtype_names: List of dtype names (e.g., ['float32', 'float16'])
            shapes: List of weight shapes

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