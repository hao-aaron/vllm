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
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        # print(f"VLLM: INIT WEIGHT UPDATE: GPU ID {ray.get_gpu_ids()[0]}")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


MODEL_NAME = "unsloth/gpt-oss-20b-BF16"
TP_SIZE = 2
# MODEL_NAME = "facebook/opt-125m"

# Load the model onto GPU 0 for the training workload.
train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
device = "cuda:3"
train_model.to(device)

# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
ray.init()

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/placement-groups.html
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * TP_SIZE)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch the vLLM inference engine. The `enforce_eager` flag reduces
# start-up latency.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=TP_SIZE,
    distributed_executor_backend="ray",
    # enable_expert_parallel=True,
)

# Generate text from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0, max_tokens=100)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

# Set up the communication channel between the training process and the
# inference engine.
master_address = get_ip()
master_port = get_open_port()

handle = llm.collective_rpc.remote(
    "init_weight_update_group", args=(master_address, master_port, 1, 1+TP_SIZE)
)

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 1+TP_SIZE, torch.device(device)
)
ray.get(handle)

# Synchronize the updated weights to the inference engine.
# synchronize all weights by default
i = 0
for name, p in train_model.named_parameters():
    # If you uncomment this, you'll no longer get garbage responses
    # if (name.endswith("experts.gate_up_proj_bias") or name.endswith("experts.down_proj_bias")):
    #     continue
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight", args=(name, dtype_name, p.shape)
    )
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)
    i += 1

# Generate text with the updated model. The output is expected to be nonsense due to issues with bias terms in the experts, 
# but weight loading is successful now
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50, flush=True)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}", flush=True)
    print("-" * 50, flush=True)