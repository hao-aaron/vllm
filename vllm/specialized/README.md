# Specialized model directories (prototype)

A specialized implementation is one directory,
`vllm/models/<family>/<platform>/specialized/<name>/`, that talks to core only
through `vllm.specialized`. Examples:
`glm5next/nvidia/specialized/low_latency` (on by default) and
`minimax_m3/nvidia/specialized/low_latency` (pinned only).

Each platform folder holds its general model (`model.py` and its own files)
plus `specialized/`. `common/` is shared helpers only, except GLM-5-Next, whose
single general model lives in `common/`.

| File | Contents |
| --- | --- |
| `__init__.py` | Import-light declarations: `SUPPORTS` (`Support`), optional `resources(cfg)`, `MODEL` (architecture -> class), optional `DRAFT_MODEL`, optional `OPTIONS` (a dataclass of options only this directory reads, set with `--specialized-model-options`) |
| `EXCEPTIONS.py` | Core internals still imported (`EXCEPTIONS`) and shared objects written (`MUTATIONS`): the backlog, kept exact by `tests/specialized/test_rules.py` |
| `model.py` | The `Model` subclass of the general model; free-form otherwise |

The family lists its directories per platform in `<family>/routing.py`, and
its `__init__.py` loads model classes lazily.

## Core side

- `_routing.py`: `--specialized-model auto|off|require|<name>`. Runs once, last
  in `VllmConfig.__post_init__`, on the resolved config: tests `SUPPORTS`,
  records the choice in `model_config.specialized_choice` and prints why,
  naming the flag to add. It never changes a core setting. `MODEL` applies at
  every class resolution (top level, `init_vllm_registered_model`, draft);
  workers first re-check arch and libraries and agree by a MIN all-reduce.
  `vllm.specialized.options(vllm_config)` returns the directory's `OPTIONS`.
- `model.py`: `Model`, the one model interface. Hooks keep `ModelState`'s names
  and signatures (`initialize_kv_cache`, `capture_inner_cudagraphs`,
  `add_request(req_index, new_req_data)`, `prepare_inputs(input_batch)`,
  `prepare_attn`, ...); each default is core's automatic wrapper for the
  model's kind, and an override may call `super()` for it. Accessors
  (`get_embedding_layer`, `get_lm_head`, `get_language_model`, optional
  `get_topk_indices_buffer`) are what speculators read. Draft hooks
  (`set_prefill_rows`, `sample_sequential`, `sample_markov_chain`,
  `combine_hidden_states`) have no base defaults: a draft defines the ones it
  has. A specialized model mixes `Model` into the general class
  (`class X(Model, GeneralX)`) and overrides only hooks, accessors and the
  general class's `EXTENSION_POINTS` (class attributes or overridable build
  methods), never `__init__`.
- `vllm/v1/worker/gpu/model_states/__init__.py`: wraps every model in its
  kind's state; for a `Model`, or a general wrapper whose language model is
  one, the state dispatches each overridden hook to it, except hooks the
  wrapper implements itself. Models with their own `get_model_state_cls` take
  a deprecated path.
- `services.py`: `groups`; `comm` (collectives, `peer_arena` / `peer_buffer`
  for memory shared with peer ranks); `step`, the read-only metadata of the
  step or capture being run (`attn_metadata[layer]`, `slot_mapping[layer]`,
  `is_capturing`, `graph_mode`).
- `_lint.py`: import, mutation and override lints
  (`python -m vllm.specialized._lint -v`). They read the source, so attributes
  set at run time are review items.

Order: `load_weights`; `process_weights_after_loading()` (weight-only copies
and re-layouts); `target.attach_draft(draft)`; `target.warmup()` (JIT,
autotune, and executors that read anything set up after loading; a target
whose draft shares its state builds the draft's executor here);
`draft.warmup()`; memory profiling; `initialize_kv_cache`; capture, after
which core logs `fast_path_sizes()`, the captured sizes that take the fast
path. A directory that builds in `warmup()` can't list `weight_reload`.
