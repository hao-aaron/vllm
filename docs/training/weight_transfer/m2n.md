# NCCL M2N Engine

The `nccl_m2n` weight transfer engine uses [NCCL M2N](https://github.com/NVIDIA/nccl-extensions) to **reshard** weights between two disjoint meshes of GPUs — the trainer and the inference workers — in a single collective per parameter. Unlike the broadcast-based [NCCL engine](nccl.md), the trainer sends its local shard as-is: no full tensor is ever materialized on the training side, whatever its parallelism layout.

## When to Use NCCL M2N

- The trainer and the inference engine use **different parallelism layouts** (e.g. trainer FSDP/EP, inference TP) and gathering full tensors on the trainer is expensive
- Training and inference are on **separate GPUs**, possibly across nodes
- You want weight sync off the critical path of an async RL loop

## Requirements

This backend depends on the `nccl-extensions` Python package (and its `nccl4py` dependency), which vLLM does not install. Install both against a compatible NCCL 2.30.5 or newer runtime. The backend creates its data-plane communicator through `nccl4py` and passes that communicator directly to `nccl.m2n`.

The backend imports the runtime lazily, so vLLM is unaffected unless you select it.

## How It Works

1. A `StatelessProcessGroup` exchanges metadata and the NCCL unique IDs. Each parameter belongs to a **source group**, its `ParamMeta.owners` (every trainer rank by default). Every distinct source group gets one `nccl.core.Communicator`: the group's trainer ranks, in order, then all inference workers. A pipeline-parallel trainer declares one group per stage.
2. At the init handshake the trainer ships each parameter's name, dtype, shape, owners, source mesh and source placements. Each worker builds a **load plan** (`plan_loads`, [RFC #64418](https://github.com/vllm-project/vllm/issues/64418)) of its model: which boxes of each checkpoint tensor it needs, and where they land in live memory.
3. The workers agree on one **destination layout** per parameter: they test a fixed list of candidates (every 2-D factorization of the workers, one axis `Shard(d)`, the other `Replicate`, most shards first, then fully replicated) against their own boxes, AND the results over the metadata group, and take the first layout that gives every worker everything it needs. A parameter no worker needs is not sent.
4. Each round, both sides issue one `reshard` per parameter in the same order. A worker reshards straight into the live tensor when its destination is one contiguous box, otherwise into a staging buffer whose box it copies into the plan's destinations. A parameter the plan cannot write directly (a cast, a computed loader) arrives in full and goes through `load_weights`.

The engine requires `VLLM_RELOAD_MODE=direct`, and logs, per worker, how many parameters take each path and how many bytes it receives per update.

## Configuration

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl_m2n"),
)
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "nccl_m2n"}'
```

## Trainer Side

The trainer uses the stateful trainer engine. m2n plans each reshard from the parameter's source layout, which a [`WeightSource`](base.md#parammeta-ownership-and-layout) declares on its `ParamMeta` (`owners`, `mesh_shape`, `placements`), and sends what `iter_local()` yields: each rank's **local shard**, never a `full_tensor()`. `ModuleSource` covers the common case: it reads each parameter's device mesh and placements.

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

# Called on every trainer rank; rank 0 is the sender.
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=M2NTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=num_trainer_ranks + num_inference_workers,
        num_trainer_ranks=num_trainer_ranks,
        rank=my_rank,
    ),
    client=RayVLLMWeightSyncClient(llm_handle),
    source=ModuleSource(model),
)

# Each round, on every trainer rank:
engine.send_weights()
```

`send_weights()` drives the whole round — `start_weight_update`, `update_weights`, `finish_weight_update`. Every rank in the communicator must be inside the same reshard, so `update_weights` has to be in flight while the trainer sends; the engine runs it on a helper thread so the trainer loop does not have to open-code that.

Trainers with a custom producer (a Megatron export, MoE re-fusing) subclass
`WeightSource`, declare each layout on `ParamMeta` and implement `iter_local()`.
A 1-D mesh is described over m2n's 2-D mesh with a size-1 second axis. Each
yielded tensor must match its declared dtype and local shape, and is kept alive
in stream order until its reshard completes.

See [`examples/rl/rlhf_m2n.py`](../../../examples/rl/rlhf_m2n.py) for a runnable FSDP → TP example.

`M2NTrainerInitInfo.forbid_full_fallback` names parameters that must never arrive in full for `load_weights`; init fails if a worker's plan cannot write one of them directly.

## Limitations

- vLLM validates tensor-rank and dtype support while agreeing on the plan; the
  installed M2N runtime validates them again for each reshard.
- Source groups run one after another, in metadata order; their transfers do not overlap yet.
- No pipeline-parallel inference workers, no draft-model updates.
- `Partial` placements cannot be sent as shards. `ModuleSource` declares such a parameter replicated and gathers it in `iter_local()`.
- A reshard cannot be issued from inside a CUDA graph capture (weight updates run outside capture, so this only matters for custom callers).
- Destination layouts over conservative 16-source and 64-target capacities
  are not candidates. The installed M2N
  runtime remains authoritative and validates each reshard again.
- Each communicator holds a cached staging pool (~256 MiB by default: 4 channels x 64 MiB).
