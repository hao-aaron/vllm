# Weight Updates (SupportsReload)

RL frameworks update a running engine's weights in place
(`start_weight_update` → `update_weights` → `finish_weight_update`, or
`reload_weights`). The model's own `load_weights` and weight loaders run on
the new tensors; each module's post-load processing runs again, and its
results land in the live tensors, which captured CUDA graphs keep reading.
Then every weight-derived value is recomputed in place.

Each module takes one of two paths, chosen at the start of every update from
what was recorded at model init:

- **direct**: loaders write live memory, either the live tensor itself (its
  shape, stride and dtype are the checkpoint's) or a checkpoint-layout view of
  its bytes. Scales whose live form is smaller than their checkpoint form are
  staged in small side buffers. No extra memory; the live bytes hold the
  update from the first write.
- **modulewise**: loaders write a fresh checkpoint-format scratch tensor (e.g.
  online quantization, where the BF16 checkpoint doesn't fit the FP8 live
  bytes); the live bytes are untouched until the module lands. Tensors from
  the CPU (a reload from disk) are buffered in host memory and loaded when
  the module's last one arrived. The cold load of online-quantized models
  takes this path too.

`VLLM_RELOAD_DIRECT=0` forces modulewise everywhere. A module completes when
every unit it expects arrived (see below): its post-load processing runs and
its results land. Models that run post-load work inside `load_weights`, or
whose loads were never observed, use the legacy layerwise path
(`VLLM_RELOAD_LEGACY=1` selects it for every model).

Cold start and reload run the same phases, in the same order:

1. every quant method's `process_weights_after_loading` (PWAL);
2. kernel `refresh()` of every reload-safe quant method;
3. attention post-load processing;
4. layer refreshes declared with `register_refresh`;
5. `SupportsReload.refresh_model()`, children first;
6. cold start only: `SupportsReload.post_load_setup()`.

A model that implements
[SupportsReload][vllm.model_executor.models.interfaces.SupportsReload] puts
every kind of post-load work in one place, so cold start and every reload
run the same code:

| Kind of work | Where it goes |
| --- | --- |
| Per-shard transform (TP narrowing, permutation, layout scatter) | the param's `weight_loader` |
| Checkpoint → kernel conversion | the quant method's `process_weights_after_loading` |
| Weight-derived values held by kernels | kernel `refresh()` (framework-called) |
| Weight-derived values spanning modules | `SupportsReload.refresh_model()` |
| Non-weight setup (compile, device placement) | `SupportsReload.post_load_setup()` |

A model that implements `SupportsReload` does not define a model-level
`process_weights_after_loading()`: reload never calls it.

## Rules

Loaders:

- **L0.** No post-load work inside `load_weights` (no `is_reload_active()`
  checks, no writes outside loaders): with direct loading it would run on live
  bytes mid-update. Such models take the legacy path.
- **L1.** Every write to a loadable param goes through that param's
  `weight_loader`. `load_weights` only routes names. During a reload the
  params are meta placeholders until a loader runs, so `param.data.copy_(...)`
  or `default_weight_loader(param, w)` in `load_weights` silently does nothing.
- **L2.** A transform where each output byte depends on one checkpoint tensor
  belongs in the loader (wrap it, e.g. `composed_weight_loader`).
- **L3.** State pairing tensors across loader calls lives on `self`: a
  streamed reload calls `load_weights` once per batch.
- **L4.** Apply `WeightsMapper` rules once per name.

Post-load processing (PWAL):

- **P1.** Always transform; guard only construction, on the object itself
  (`if self.moe_kernel is None`), never with a global flag.
- **P2.** Kernels reference registered params; any other tensor they hold is
  declared (`register_derived`, `add_refresh(fn, *outputs)`,
  `declare_derived`) and written by `refresh()`.
- **P3.** No transform inside kernel construction.
- **P4.** Config side effects are computed from tensors, so a rerun writes
  the same value.
- **P5.** Don't set params to `None` or delete submodules reload needs.
- **P6.** A MoE method whose kernel holds undeclared derived state keeps
  `reload_safe = False`; under CUDA graphs reload refuses it.
- **P7.** A quant method may declare `reload_post_step`: `"pwal"` (default,
  runs when the module's units arrived), `"pwal_at_finish"` (runs at finish,
  after every other module landed: required when the PWAL runs collectives or
  reads another module's weights), or `"none"` (only when the live layout is
  the checkpoint layout, or the transform happens in the loader; each
  declaration needs a refit-equals-fresh case on the hardware it's declared
  for).

Refresh (`refresh_model`, kernel `refresh`, `register_refresh`):

- **R1.** Read only live, landed tensors.
- **R2.** Write only buffers allocated by the first call (cold start, before
  graph capture), with `copy_` / `out=`.
- **R3.** Idempotent, and safe when only some modules were updated.
- **R4.** Collectives only if every rank calls them.

## Changing a generic layer's processing

Use `wrap_quant_method(layer, before_pwal=..., after_pwal=...)` at
construction instead of monkeypatching `quant_method.process_weights_after_loading`
or rewriting the layer's tensors from a model hook. `before_pwal` sees the
checkpoint-format tensors, `after_pwal` the PWAL results; both run on every
load and their results land once.

## Arrival units

Every loader call reports what it delivered: a whole tensor, one shard of a
fused linear (`loaded_shard_id`), or one expert's shard (`shard_id`,
`expert_id`); the framework infers these from the loader's arguments. A loader
that declines data belonging to another rank calls `report_declined()`; one
that writes a tensor over several calls whose arguments don't say which part
calls `report_units(...)`.

An update's expected units come, in order of preference, from a dry run of
`load_weights` over the trainer's metadata (shipped at
`init_weight_transfer_engine`), over a reloaded checkpoint's safetensors
headers, or from the units the cold load delivered. At finish, a module that
received only some of its units fails the update: its live bytes would be
mixed. A unit whose tensor was created with a meaningful value (ones, scale
sentinels) is optional unless the sender's own metadata lists it.

## Integrity

No forward runs between `start_weight_update` and `finish_weight_update`: the
engine pauses a running scheduler at start and resumes it after a successful
finish, and the worker refuses any step mid-update (DP engines are not paused
automatically; pause them before updating). All ranks agree before any write
(the reload policy, a declared module set) and on the outcome before any of
them serves.

A failed update leaves the model dirty: the engine keeps generation paused,
rejects new requests and refuses to resume, while weight-update RPCs keep
working. An update clears it when it succeeds and re-completes every module
the failed one wrote (a module counts when every tensor the cold load wrote
received a loader call). A sparse update never clears it.

`start_weight_update(partial=False)` requires the update to reach every module
the cold load wrote (finish raises otherwise); `partial=True` allows a
subset, with `modules=[...]` checked to be the same on every rank.

## Checking a model

```python
llm = LLM(model, weight_transfer_config={"backend": "nccl"})
llm.check_reload_support()  # declarations, ownership, structure, configuration
```

The same report is served at `GET /check_reload_support`. Each item is SAFE,
UNVERIFIED (not checkable from structure) or UNSAFE (an update would serve
stale or wrong weights). An update is refused, before any write, only for
cases known to produce wrong weights: configurations (EPLB, weight
offloading, sparse patches into kernel layouts; `VLLM_RELOAD_ALLOW_UNSAFE=1`
overrides) and, under CUDA graphs, an undeclared MoE method that built a
kernel or a kernel object or live tensor replaced outside reload. The verifier proves structure, not values:
every reload-safe model or backend also needs a refit-equals-fresh test
(`tests/model_executor/model_loader/test_reload_refit.py`).
