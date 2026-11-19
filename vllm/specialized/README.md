# Specialized model directories (prototype)

A specialized implementation is one directory,
`vllm/models/<family>/<platform>/specialized/<name>/`, that talks to core only
through `vllm.specialized`. Examples:
`glm5next/nvidia/specialized/low_latency` (on by default) and
`minimax_m3/nvidia/specialized/low_latency` (pinned only).

| File | Contents |
| --- | --- |
| `__init__.py` | Import-light declarations: `SUPPORTS` (`Support`), optional `resources(cfg)`, `MODEL` (architecture -> class), optional `DRAFT_MODEL` |
| `EXCEPTIONS.py` | Core internals still imported (`EXCEPTIONS`) and shared objects written (`MUTATIONS`): the backlog, kept exact by `tests/specialized/test_rules.py` |
| `model.py` | The `Model` subclass of the general model; free-form otherwise |

The family lists its directories per platform in `<family>/routing.py`, and
its `__init__.py` loads model classes lazily.

## Core side

- `_routing.py`: `--specialized-model auto|off|require|<name>`. Runs once, last
  in `VllmConfig.__post_init__`, on the resolved config: tests `SUPPORTS`,
  records the choice in `model_config.specialized_choice` and prints why,
  naming the flag to add. It never changes a setting. `MODEL` applies at every
  class resolution (top level, `init_vllm_registered_model`, draft); workers
  first re-check arch and libraries and agree by a MIN all-reduce.
- `model.py`: `Model`, the one model interface. Its hooks keep `ModelState`'s
  names and signatures; each default is core's automatic wrapper for the
  model's kind, and an override may call `super()` for it. Accessors
  (`get_input_embeddings`, `get_lm_head`, `get_language_model`) are what
  speculators read. A specialized model mixes it into the general class
  (`class X(Model, GeneralX)`) and overrides only hooks, accessors and the
  general class's `EXTENSION_POINTS`, never `__init__`.
- `vllm/v1/worker/gpu/model_states/__init__.py`: wraps every model in its
  kind's state; for a `Model` (or a general wrapper whose language model is
  one) the state dispatches each overridden hook to it. Models with their own
  `get_model_state_cls` take a deprecated path.
- `services.py`: `groups`, and `comm` (collectives, `peer_arena` /
  `peer_buffer` for memory shared with peer ranks).
- `_lint.py`: import, mutation and override lints
  (`python -m vllm.specialized._lint -v`).

Load order: `load_weights`, then `process_weights_after_loading()` (derived
weights and executors), then `attach_draft(draft)` and `warmup()` (JIT,
autotune), all before memory profiling.
