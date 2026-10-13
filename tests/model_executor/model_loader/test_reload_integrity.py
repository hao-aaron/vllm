# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-update integrity on layerwise reload: a failed update leaves the
model dirty until an update re-completes every module the failed one wrote,
a full update must reach every module the cold load wrote, and nothing may
serve mid-update."""

import pytest
import torch

from vllm.model_executor.model_loader.reload import (
    abort_reload,
    begin_update,
    check_can_serve,
    end_update,
    finalize_layerwise_reload,
    get_reload_session,
    initialize_layerwise_reload,
    is_model_dirty,
    mark_dirty,
    record_metadata_for_reloading,
    record_written,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    ReloadDirtyError,
    ReloadIncompleteError,
    record_cold_load,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def _layer(n_params: int = 2) -> torch.nn.Module:
    layer = torch.nn.Module()
    for i in range(n_params):
        p = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        p.weight_loader = default_weight_loader
        layer.register_parameter(f"w{i}", p)
    return layer


def _model(n: int = 3) -> torch.nn.Sequential:
    model = torch.nn.Sequential(*(_layer() for _ in range(n)))
    record_metadata_for_reloading(model)
    record_cold_load(model, [f"{i}.w{j}" for i in range(n) for j in range(2)])
    return model


def _load(model, i: int, value: float, params=(0, 1)) -> None:
    layer = model[i]
    for j in params:
        p = getattr(layer, f"w{j}")
        p.weight_loader(p, torch.full((4, 4), value))


def _update(model, values: dict[int, float], **kwargs) -> None:
    initialize_layerwise_reload(model, **kwargs)
    for i, v in values.items():
        _load(model, i, v)
    finalize_layerwise_reload(model, model_config=None)


def test_failed_update_is_dirty_until_its_modules_are_recompleted():
    model = _model()
    live = [model[i].w0 for i in range(3)]
    initialize_layerwise_reload(model)
    _load(model, 0, 1.0)  # completes and lands layer 0
    _load(model, 1, 2.0, params=(0,))  # layer 1 incomplete
    abort_reload(model)  # e.g. the transport failed
    # the original tensor objects are back; layer 1 kept its old values
    assert [model[i].w0 for i in range(3)] == live
    assert torch.equal(model[0].w0, torch.ones(4, 4))
    assert torch.equal(model[1].w0, torch.zeros(4, 4))
    assert is_model_dirty(model)
    assert get_reload_session(model).dirty_modules == {"0"}
    with pytest.raises(RuntimeError, match="refuses to serve"):
        check_can_serve(model)

    # an update that doesn't re-complete layer 0 leaves it dirty
    with pytest.raises(ReloadDirtyError, match=r"\['0'\]"):
        _update(model, {1: 3.0, 2: 3.0}, partial=True)
    assert is_model_dirty(model)
    # touching layer 0 without completing it doesn't count either
    initialize_layerwise_reload(model, partial=True)
    _load(model, 0, 4.0, params=(0,))
    with pytest.raises(ReloadDirtyError):
        finalize_layerwise_reload(model, model_config=None)
    assert is_model_dirty(model)

    _update(model, {0: 5.0}, partial=True)  # re-completes layer 0
    assert not is_model_dirty(model)
    check_can_serve(model)


def test_full_update_must_reach_every_module_the_cold_load_wrote():
    model = _model()
    with pytest.raises(ReloadIncompleteError, match=r"\['2'\]"):
        _update(model, {0: 1.0, 1: 1.0}, partial=False)
    abort_reload(model)
    assert is_model_dirty(model)  # layers 0 and 1 landed
    _update(model, {0: 1.0, 1: 1.0, 2: 1.0}, partial=False)
    assert not is_model_dirty(model)


def test_modules_the_cold_load_did_not_write_are_not_required():
    """e.g. an attention layer whose k/v scales a checkpoint doesn't carry."""
    model = torch.nn.Sequential(_layer(), _layer())
    record_metadata_for_reloading(model)
    record_cold_load(model, ["0.w0", "0.w1"])
    _update(model, {0: 1.0}, partial=False)
    assert not is_model_dirty(model)


def test_shared_param_counts_for_its_first_owner():
    """A tied lm_head is loaded through embed_tokens' loader."""
    model = torch.nn.Sequential(_layer(1), torch.nn.Module())
    model[1].register_parameter("w0", model[0].w0)
    record_metadata_for_reloading(model)
    record_cold_load(model, ["0.w0", "1.w0"])
    assert model.__dict__["_reload_required_modules"] == {"0"}


def test_parameters_tied_after_the_cold_load_count_for_their_first_owner():
    """An `lm_head` the checkpoint shipped loads untied, and is re-tied to the
    embeddings after the cold load when identical: an update then reaches it
    through the embeddings' loader only."""
    model = _model(2)
    model[1].w0, model[1].w1 = model[0].w0, model[0].w1  # re-tied
    _update(model, {0: 2.0}, partial=False)
    assert model[1].w0 is model[0].w0 and model[1].w0[0, 0] == 2.0


def test_unspecified_update_logs_untouched_modules(caplog_vllm):
    model = _model()
    _update(model, {0: 1.0})
    assert "received no weights" in caplog_vllm.text
    assert not is_model_dirty(model)


def test_no_forward_mid_update():
    model = _model()
    initialize_layerwise_reload(model)
    with pytest.raises(RuntimeError, match="mid weight update"):
        check_can_serve(model)
    _load(model, 0, 1.0)
    _load(model, 1, 1.0)
    _load(model, 2, 1.0)
    finalize_layerwise_reload(model, model_config=None)
    check_can_serve(model)


def test_failed_start_is_not_dirty():
    model = _model()
    begin_update(model)
    abort_reload(model, dirty=False)
    assert not is_model_dirty(model)


def test_peer_failure_marks_this_ranks_completed_modules():
    model = _model()
    _update(model, {0: 1.0, 2: 1.0})
    mark_dirty(model)  # the update failed on another rank
    assert get_reload_session(model).dirty_modules == {"0", "2"}
    _update(model, {0: 2.0, 2: 2.0}, partial=True)
    assert not is_model_dirty(model)


def test_writes_outside_loaders_join_the_dirty_set():
    model = _model()
    begin_update(model)
    record_written(model, ["1"])  # e.g. a planned direct write
    abort_reload(model)
    assert get_reload_session(model).dirty_modules == {"1"}


def test_sparse_update_never_clears_dirty():
    model = _model()
    begin_update(model)
    record_written(model, ["0"])
    abort_reload(model)
    begin_update(model, sparse=True)
    record_written(model, ["0", "1", "2"])
    with pytest.raises(ReloadDirtyError, match="sparse"):
        end_update(model)
    assert is_model_dirty(model)
    # a clean model takes sparse updates normally
    clean = _model()
    begin_update(clean, sparse=True)
    record_written(clean, ["0"])
    end_update(clean)
    assert not is_model_dirty(clean)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_engine_refuses_after_failed_update_and_recovers(vllm_runner):
    """A full update that misses modules fails at finish: the engine stays up
    but refuses to serve (no weight version is set) until an update succeeds
    that re-sends what the failed one wrote."""
    from vllm import SamplingParams

    from .test_reload_refit import HARNESS

    model_b = "inference-optimization/Qwen3-0.6B-debug-multiply"
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        enable_prefix_caching=False,
        weight_transfer_config={"backend": "ipc"},
        worker_extension_cls=HARNESS,
    ) as runner:
        llm = runner.llm
        sp = SamplingParams(max_tokens=4, temperature=0)

        def generate():
            return llm.generate(["The capital of France is"], sp, use_tqdm=False)[0]

        assert generate().outputs[0].text
        llm.start_weight_update(partial=False)
        llm.collective_rpc(
            "mw_load_only", kwargs=dict(path=model_b, skip_regex=r"layers\.1\.mlp")
        )
        with pytest.raises(Exception, match="layers.1.mlp"):
            llm.finish_weight_update(weight_version="v1")
        assert llm.get_weight_version() != "v1"
        try:
            refused = generate().outputs[0].finish_reason == "error"
        except Exception:
            refused = True
        assert refused

        llm.start_weight_update(partial=False)
        llm.collective_rpc("mw_load_only", kwargs=dict(path=model_b))
        llm.finish_weight_update(weight_version="v2")
        assert llm.get_weight_version() == "v2"
        assert generate().outputs[0].text
