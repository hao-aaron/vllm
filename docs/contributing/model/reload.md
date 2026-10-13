# Weight Updates (SupportsReload)

RL frameworks update a running engine's weights in place
(`start_weight_update` → `update_weights` → `finish_weight_update`, or
`reload_weights`). Reload restores each module's checkpoint-format params,
runs the model's own `load_weights` and weight loaders on the new tensors,
re-runs each module's post-load processing, and lands the results in the
live tensors, which captured CUDA graphs keep reading. Then every
weight-derived value is recomputed in place.

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

## Checking a model

```python
llm = LLM(model, weight_transfer_config={"backend": "nccl"})
llm.check_reload_support()  # declarations, ownership, structure, configuration
```

The same report is served at `GET /check_reload_support`. Each item is SAFE,
UNVERIFIED (not checkable from structure) or UNSAFE (an update would serve
stale or wrong weights). The report never refuses an update; reload itself
refuses, before any write, the cases it knows it cannot apply under CUDA
graphs (an undeclared MoE method that built a kernel, a kernel object or live
tensor replaced outside reload). The verifier proves structure, not values:
every reload-safe model or backend also needs a refit-equals-fresh test
(`tests/model_executor/model_loader/test_reload_refit.py`).
