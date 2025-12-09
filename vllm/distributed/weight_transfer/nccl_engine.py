# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based weight transfer engine."""

from typing import Any, Type

import torch

from dataclasses import dataclass

from vllm.distributed.weight_transfer.base import WeightTransferEngine, WeightUpdateRequest, WeightTransferInitInfo
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.config.parallel import ParallelConfig



@dataclass
class NCCLWeightTransferInitInfo(WeightTransferInitInfo):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int

@dataclass
class NCCLWeightUpdateRequest(WeightUpdateRequest):
    pass

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

    @property
    def init_info_cls(self) -> Type[NCCLWeightTransferInitInfo]:
        return NCCLWeightTransferInitInfo

    @property
    def update_request_cls(self) -> Type[NCCLWeightUpdateRequest]:
        return NCCLWeightUpdateRequest

    def init_transfer(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """
        Initialize NCCL process group with the trainer.

        Args: 
            init_info: NCCLWeightTransferInitInfo
        """
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Calculate the global rank in the trainer-worker process group
        worker_rank = self.parallel_config.rank
        rank = worker_rank + init_info.rank_offset

        # Create stateless process group
        pg = StatelessProcessGroup.create(
            host=init_info.master_address, port=init_info.master_port, rank=rank, world_size=init_info.world_size
        )

        # Initialize NCCL communicator
        self.model_update_group = PyNcclCommunicator(pg, device=torch.cuda.current_device())

    def receive_weights(
        self, request: NCCLWeightUpdateRequest
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from trainer via NCCL broadcast.

        Args:
            request: NCCLWeightUpdateRequest

        Returns:
            List of (name, weight_tensor) tuples
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. Call init_transfer() first."
            )

        weights = []
        for name, dtype_name, shape in zip(request.names, request.dtype_names, request.shapes):
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