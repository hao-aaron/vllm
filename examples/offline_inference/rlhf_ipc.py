# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies GPU 0 for training, whereas a
tensor-parallel vLLM inference engine occupies GPU 1–2.

The example performs the following steps:

* Load the training model on GPU 0.
* Split the inference model across GPUs 1–2 using vLLM's tensor parallelism
  and Ray placement groups.
* Generate text from a list of prompts using the inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group. Note that
  for demonstration purposes we simply zero out the weights.

For a production-ready implementation that supports multiple training and
inference replicas, see the OpenRLHF framework:
https://github.com/OpenRLHF/OpenRLHF

This example assumes a single-node cluster with three GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


# Load the OPT-125M model onto GPU 0 for the training workload.


@ray.remote
class TrainModel:
    def __init__(self):
        self.train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.train_model.to("cuda:0")

    def init_weight_transfer(self):
        # Set up the communication channel between the training process and the
        # inference engine.
        pass

    def broadcast_weights(self, llm_handle: ray.ObjectRef):
        self.llm_handle = llm_handle
        names, dtypes, shapes, ipc_handles = [], [], [], []

        for name, p in self.train_model.named_parameters():
            names.append(name)
            dtypes.append(str(p.dtype).split(".")[-1])
            shapes.append(p.shape)

            from torch.multiprocessing.reductions import reduce_tensor

            weight = p.detach().contiguous()
            ipc_handle = reduce_tensor(weight)
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handles.append(ipc_handle)

        ray.get(
            self.llm_handle.collective_rpc.remote(
                "update_weights", args=(names, dtypes, shapes, ipc_handles)
            )
        )


ray.init(runtime_env={"excludes": [".git/objects/pack/"]})

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/placement-groups.html

pg_colocate = placement_group([{"GPU": 1, "CPU": 0}])
ray.get(pg_colocate.ready())


llm = ray.remote(
    num_cpus=0,
    num_gpus=0.4,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg_colocate,
        placement_group_capture_child_tasks=True,
    ),
)(MyLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=True,
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
    gpu_memory_utilization=0.7,
    weight_transfer_config=WeightTransferConfig(backend="ipc"),
)

train_model = TrainModel.options(
    num_gpus=0.1,
    num_cpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg_colocate, placement_group_capture_child_tasks=True
    ),
).remote(llm)


# Generate text from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

ray.get(train_model.init_weight_transfer.remote())

# Simulate a training step by zeroing out all model weights.
# In a real RLHF training loop the weights would be updated using the gradient
# from an RL objective such as PPO on a reward model.
for name, p in train_model.named_parameters():
    p.data.zero_()

# Synchronize the updated weights to the inference engine using batched API.
ray.get(train_model.broadcast_weights.remote(llm))

# Finalize the weight update (processes weights for quantization/kernel format)
ray.get(llm.collective_rpc.remote("finalize_weight_update"))

# Generate text with the updated model. The output is expected to be nonsense
# because the weights are zero.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
