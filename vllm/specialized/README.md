# Specialized model directories (prototype)

A specialized implementation is one directory,
`vllm/models/<family>/<platform>/specialized/<name>/`, that talks to core only
through `vllm.specialized`. Examples:
`glm5next/nvidia/specialized/low_latency` (on by default) and
`minimax_m3/nvidia/specialized/low_latency` (pinned only).

| File | Contents |
| --- | --- |
| `__init__.py` | Import-light declarations: `SUPPORTS` (`Support`), optional `defaults(cfg)` and `resources(cfg)`, `MODEL`, optional `DRAFT_MODEL` |
| `EXCEPTIONS.py` | Core internals still imported (`EXCEPTIONS`) and shared objects patched (`MUTATIONS`), kept exact by `tests/specialized/test_rules.py` |
| `model.py` | The `Model` subclass; free-form otherwise |

The family lists its directories per platform in `<family>/routing.py`, and
its `__init__.py` loads model classes lazily.

## Core side

- `_routing.py`: `--specialized-model auto|off|require|<name>`. Runs once in
  `VllmConfig.__post_init__`: applies each candidate's `defaults` to unset
  settings, tests `SUPPORTS`, records the choice in
  `model_config.specialized_choice`, prints why, and re-checks it once the
  config is resolved. Workers re-check per-rank facts and agree before
  `initialize_model` builds `MODEL` / `DRAFT_MODEL`.
- `model.py`: `Model`, an `nn.Module` mixin placed before the general class
  (`class X(Model, GeneralX)`). Core reaches into the general structure
  (`lm_head`, `model.embed_tokens`, ...), so the class extends the general one
  rather than wrapping it.
- `vllm/v1/worker/gpu/model_states/specialized.py`: the pass-through
  ModelState: the general model's state class, then the `Model` hooks
  (`warmup` after load, `init_kv_cache`, `sample_logits`, ...).
- `services.py`: `groups` (rank, size, device and CPU groups) and `comm`
  (collectives) over core's process groups.
- `_lint.py`: the Rule 1 import and mutation lints
  (`python -m vllm.specialized._lint -v`).

`MODEL` may map architectures to classes; mapping a wrapper architecture to
`"language_model"` keeps the wrapper general and specializes its language
model.
