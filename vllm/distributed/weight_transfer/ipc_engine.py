
from ast import ListComp
from typing import Any, List, Dict, Tuple

import torch

from vllm.distributed.weight_transfer.base import WeightTransferEngine

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.config.parallel import ParallelConfig


class IPCWeightTransferEngine(WeightTransferEngine):
    """
    Weight transfer engine using CUDA IPC for communication between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group.
    """

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        super().__init__(config, parallel_config)

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
        pass
        

    def receive_weights(
        self, names: List[str], dtype_names: List[str], shapes: List[Tuple], ipc_handles: List[Dict[str, str]]
    ) -> List[Tuple[str, torch.Tensor]]:
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
        weights = []
        for name, dtype_name, shape, ipc_handle in zip(names, dtype_names, shapes, ipc_handles):
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties()
            physical_gpu_id = str(props.uuid)

            handle = ipc_handle[physical_gpu_id]

            device_id = device.index
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            weight = func(*list_args)
            weights.append((name, weight))

        return weights