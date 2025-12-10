# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
)


class IPCWeightTransferEngine(WeightTransferEngine):
    """
    Weight transfer engine using CUDA IPC for communication between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group.
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
        super().__init__(config, parallel_config)

    def init_transfer(self) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.
        """
        pass

    def receive_weights(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        **kwargs: dict[str, Any],
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Receive weights from the trainer via CUDA IPC handles.

        Args:
            names: List of parameter names
            dtype_names: List of dtype names (e.g., "float32", "bfloat16")
            shapes: List of parameter shapes
            kwargs: Additional kwargs
                ipc_handles (List[Dict[str, Any]]):
                    A list of CUDA IPC metadata dictionaries, one per parameter.

                    Each dictionary is a mapping between physical GPU UUID
                    and the IPC handle.

        Returns:
            List of (name, weight_tensor) tuples ready to be loaded into the model
        """
        ipc_handles = kwargs.get("ipc_handles")
        if ipc_handles is None:
            raise ValueError(
                "Expected `ipc_handles` to be present in weight update request"
            )
        elif not isinstance(ipc_handles, list) or not isinstance(ipc_handles[0], dict):
            raise ValueError(
                f"Expected ipc_handles to be a list of CUDA IPC metadata"
                f" dictionaries, got {ipc_handles}"
            )
        elif len(ipc_handles) != len(names):
            raise ValueError(
                f"Expected `ipc_handles` to be of the same size as `names`,"
                f"got {len(ipc_handles)} and {len(names)} "
            )

        weights = []
        for name, dtype_name, shape, ipc_handle in zip(
            names, dtype_names, shapes, ipc_handles
        ):
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties()
            physical_gpu_id = str(props.uuid)

            handle = ipc_handle[physical_gpu_id]

            func, args = handle
            list_args = list(args)  # type: ignore
            list_args[6] = device_index
            weight = func(*list_args)  # type: ignore
            weights.append((name, weight))

        return weights

    def shutdown(self) -> None:
        """
        Shutdown the weight transfer engine.
        """
        pass
