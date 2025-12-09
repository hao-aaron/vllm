
from ast import ListComp
from typing import Any, List, Dict, Tuple, Type

import torch
from dataclasses import dataclass

from vllm.distributed.weight_transfer.base import WeightTransferEngine, WeightUpdateRequest, WeightTransferInitInfo

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.config.parallel import ParallelConfig



@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    pass

@dataclass
class IPCWeightUpdateRequest(WeightUpdateRequest):
    ipc_handles: list[dict[str, str]]


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
    
    @property
    def init_info_cls(self) -> Type[IPCWeightTransferInitInfo]:
        return IPCWeightTransferInitInfo

    @property
    def update_request_cls(self) -> Type[IPCWeightUpdateRequest]:
        return IPCWeightUpdateRequest

    def init_transfer(self, init_info: IPCWeightTransferInitInfo) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.

        Args:
            init_info: IPCWeightTransferInitInfo
        """
        pass
        

    def receive_weights(
        self, request: IPCWeightUpdateRequest
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Receive weights from the trainer.

        Args:
            request: IPCWeightUpdateRequest

        Returns:
            List of (name, weight_tensor) tuples ready to be loaded into the model
        """
        weights = [] 
        for name, dtype_name, shape, ipc_handle in zip(request.names, request.dtype_names, request.shapes, request.ipc_handles):
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties()
            physical_gpu_id = str(props.uuid)

            handle = ipc_handle[physical_gpu_id]

            func, args = handle
            list_args = list(args)
            list_args[6] = device_index
            weight = func(*list_args)
            weights.append((name, weight))

        return weights
    
    def shutdown(self) -> None:
        """
        Shutdown the weight transfer engine.
        """
        pass