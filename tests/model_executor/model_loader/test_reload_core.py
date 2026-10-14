# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reload core: per-module DIRECT / MODULEWISE paths chosen from metadata,
explicit arrival units, coverage at finish, and routing to the legacy path."""

import pytest
import torch

import vllm.model_executor.model_loader.reload.core as core
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    abort_reload,
    finish_reload,
    get_reload_session,
    is_model_dirty,
    record_metadata_for_reloading,
    report_declined,
    start_reload,
    track_names,
)
from vllm.model_executor.model_loader.reload.layerwise import ReloadIncompleteError
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter

# ---------------------------------------------------------------------------
# Toy layers
# ---------------------------------------------------------------------------


class _Method(QuantizeMethodBase):
    """PWAL: `transform(layer)`; post step as declared."""

    def __init__(self, transform=None, post="pwal"):
        self.transform = transform
        self.reload_post_step = post
        self.calls = 0

    def create_weights(self, layer, *a, **k):
        pass

    def apply(self, layer, *a, **k):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.calls += 1
        if self.transform is not None:
            self.transform(layer)


def _param(t: torch.Tensor, loader=default_weight_loader) -> torch.nn.Parameter:
    p = torch.nn.Parameter(t, requires_grad=False)
    p.weight_loader = loader
    return p


class _Fused(torch.nn.Module):
    """A q|k|v-like fused weight: loaders write one shard per call
    (`loaded_shard_id`), or the whole tensor for a pre-fused checkpoint."""

    def __init__(self, rows=2, cols=3, method=None):
        super().__init__()
        self.rows = rows
        if method is not None:
            self.quant_method = method
        self.weight = _param(torch.zeros(3 * rows, cols), self.weight_loader)

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id is None:
            param.data.copy_(loaded_weight)
            return
        i = "qkv".index(loaded_shard_id)
        param.data[i * self.rows : (i + 1) * self.rows].copy_(loaded_weight)


class _Experts(torch.nn.Module):
    """Experts 0..3, of which this rank holds 1 and 2."""

    def __init__(self):
        super().__init__()
        self.local = {1: 0, 2: 1}
        self.w = _param(torch.zeros(2, 4), self.weight_loader)

    def weight_loader(self, param, loaded_weight, shard_id, expert_id):
        if expert_id not in self.local:
            report_declined()
            return
        param.data[self.local[expert_id]].copy_(loaded_weight)


class _Model(torch.nn.Module):
    def __init__(self, layers: dict[str, torch.nn.Module]):
        super().__init__()
        for name, layer in layers.items():
            self.add_module(name, layer)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, w in weights:
            if "." in name and name.split(".")[1] in "qkv" and "fused" in name:
                module, shard = name.rsplit(".", 1)
                param = params[f"{module}.weight"]
                param.weight_loader(param, w, shard)
            elif name.startswith("experts."):
                _, expert = name.split(".")
                param = params["experts.w"]
                param.weight_loader(param, w, "w1", int(expert))
            else:
                param = params[name]
                param.weight_loader(param, w)


def _cold(model: _Model, weights: list) -> None:
    """A cold load: record metadata, observe the load, post-process."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    record_metadata_for_reloading(model)
    with core.observe_cold_load(model):
        model.load_weights(track_names(weights))
    process_weights_after_loading(model, _cfg(), torch.device("cpu"))


def _cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        quantization=None,
        dtype=torch.float32,
        word_embeddings_untied_by_checkpoint=False,
    )


def _update(model, weights, **kwargs):
    start_reload(model, **kwargs)
    model.load_weights(track_names(weights))
    finish_reload(model, _cfg())


def _fused_weights(v: float, fused=False):
    if fused:
        return [("fused.weight", torch.full((6, 3), v))]
    return [(f"fused.{s}", torch.full((2, 3), v + i)) for i, s in enumerate("qkv")]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_direct_as_is_writes_live_memory_and_keeps_pointers():
    model = _Model({"fused": _Fused(method=_Method(post="none"))})
    _cold(model, _fused_weights(1.0))
    live = model.fused.weight
    ptr = live.data_ptr()
    start_reload(model)
    plan = core._core(model).modules["fused"]
    assert plan.path == "direct" and plan.targets["weight"].kind == core.AS_IS
    model.load_weights(track_names(_fused_weights(5.0)))
    # written in place as it arrived: no scratch
    assert torch.equal(live[:2], torch.full((2, 3), 5.0))
    finish_reload(model, _cfg())
    assert model.fused.weight is live and live.data_ptr() == ptr
    assert torch.equal(live[4:], torch.full((2, 3), 7.0))
    assert "weight_loader" in live.__dict__  # its own loader, unwrapped
    assert live.weight_loader.__name__ == "weight_loader"


def test_direct_pwal_completes_when_its_units_arrived():
    def double(layer):
        replace_parameter(layer, "weight", layer.weight * 2)

    method = _Method(double)
    model = _Model({"fused": _Fused(method=method)})
    _cold(model, _fused_weights(1.0))
    live = model.fused.weight
    start_reload(model)
    state = core._core(model).modules["fused"]
    model.load_weights(track_names(_fused_weights(5.0)[:2]))
    assert not state.completed  # v is missing
    model.load_weights(track_names(_fused_weights(5.0)[2:]))
    assert state.completed  # completed while streaming, before finish
    finish_reload(model, _cfg())
    assert model.fused.weight is live
    assert torch.equal(live[:2], torch.full((2, 3), 10.0))
    assert method.calls == 2


def test_view_hosts_a_checkpoint_layout_over_live_bytes():
    """Per-tensor FP8 style: the checkpoint is [N, K], the live weight [K, N]."""

    def transpose(layer):
        replace_parameter(layer, "weight", layer.weight.t().contiguous())

    layer = torch.nn.Module()
    layer.quant_method = _Method(transpose)
    layer.weight = _param(torch.zeros(2, 3))
    model = _Model({"lin": layer})
    w = torch.arange(6.0).reshape(2, 3)
    _cold(model, [("lin.weight", w)])
    live = model.lin.weight
    assert live.shape == (3, 2)
    start_reload(model)
    assert core._core(model).modules["lin"].targets["weight"].kind == core.VIEW
    model.load_weights(track_names([("lin.weight", w + 10)]))
    finish_reload(model, _cfg())
    assert model.lin.weight is live
    assert torch.equal(live, (w + 10).t())


def test_scales_with_a_smaller_live_form_use_side_buffers():
    """[2] checkpoint scales reduced to one live scale: the module stays
    DIRECT, the scale is staged in a side buffer from its create-time value."""

    def reduce(layer):
        replace_parameter(layer, "scale", layer.scale.max().reshape(1))

    layer = torch.nn.Module()
    layer.quant_method = _Method(reduce)
    layer.weight = _param(torch.zeros(4))
    layer.scale = _param(torch.ones(2))
    model = _Model({"lin": layer})
    _cold(
        model, [("lin.weight", torch.ones(4)), ("lin.scale", torch.tensor([2.0, 3.0]))]
    )
    live_scale, live_w = model.lin.scale, model.lin.weight
    start_reload(model)
    state = core._core(model).modules["lin"]
    assert state.path == "direct"
    assert state.targets["scale"].kind == core.SIDE
    assert state.targets["weight"].kind == core.AS_IS
    model.load_weights(
        track_names(
            [
                ("lin.weight", torch.full((4,), 2.0)),
                ("lin.scale", torch.tensor([5.0, 4.0])),
            ]
        )
    )
    finish_reload(model, _cfg())
    assert model.lin.scale is live_scale and torch.equal(
        live_scale, torch.tensor([5.0])
    )
    assert model.lin.weight is live_w and torch.equal(live_w, torch.full((4,), 2.0))


def test_modulewise_scratch_keeps_live_bytes_until_landing(monkeypatch):
    """Online-quant style: the live form can't host the checkpoint (wider
    dtype), so the module loads into scratch and lands when complete."""

    def quantize(layer):
        replace_parameter(layer, "weight", layer.weight.to(torch.int8))

    layer = torch.nn.Module()
    layer.quant_method = _Method(quantize)
    layer.weight = _param(torch.zeros(2, 2))
    model = _Model({"q": layer, "fused": _Fused(method=_Method(post="none"))})
    _cold(model, [("q.weight", torch.ones(2, 2)), *_fused_weights(1.0)])
    live = model.q.weight
    assert live.dtype == torch.int8
    start_reload(model)
    assert core._core(model).modules["q"].path == "modulewise"
    model.load_weights(track_names(_fused_weights(2.0)))
    assert torch.equal(live, torch.ones(2, 2, dtype=torch.int8))  # untouched
    model.load_weights(track_names([("q.weight", torch.full((2, 2), 3.0))]))
    finish_reload(model, _cfg())
    assert model.q.weight is live
    assert torch.equal(live, torch.full((2, 2), 3, dtype=torch.int8))


def test_direct_disabled_forces_modulewise(monkeypatch):
    monkeypatch.setattr(core, "DIRECT_ENABLED", False)
    model = _Model({"fused": _Fused(method=_Method(post="pwal"))})
    _cold(model, _fused_weights(1.0))
    live = model.fused.weight
    start_reload(model)
    state = core._core(model).modules["fused"]
    assert state.path == "modulewise"
    model.load_weights(track_names(_fused_weights(4.0)[:1]))
    assert torch.equal(live[:2], torch.full((2, 3), 1.0))  # live untouched
    model.load_weights(track_names(_fused_weights(4.0)[1:]))
    finish_reload(model, _cfg())
    assert torch.equal(live[:2], torch.full((2, 3), 4.0))


# ---------------------------------------------------------------------------
# Units and coverage
# ---------------------------------------------------------------------------


def test_missing_shard_fails_the_update():
    model = _Model({"fused": _Fused(method=_Method())})
    _cold(model, _fused_weights(1.0))
    start_reload(model)
    model.load_weights(track_names(_fused_weights(2.0)[:2]))  # no v
    with pytest.raises(ReloadIncompleteError, match="'v'"):
        finish_reload(model, _cfg())
    abort_reload(model)
    assert is_model_dirty(model)


def test_untouched_module_fails_only_a_full_update():
    model = _Model(
        {"fused": _Fused(method=_Method()), "plain": _Fused(method=_Method())}
    )
    plain = [("plain.weight", torch.ones(6, 3))]
    _cold(model, _fused_weights(1.0) + plain)
    _update(model, _fused_weights(2.0), partial=True)
    _update(model, _fused_weights(3.0))  # unspecified: logged
    with pytest.raises(ReloadIncompleteError, match="'plain'"):
        _update(model, _fused_weights(4.0), partial=False)
    abort_reload(model)
    _update(model, _fused_weights(5.0) + plain, partial=False)
    assert not is_model_dirty(model)


def test_pre_fused_checkpoint_is_a_format_mismatch_not_a_failure():
    """The cold load sent q/k/v; this sender sends one fused tensor. The
    expected set doesn't describe it: the module completes at finish when
    every tensor received data."""
    model = _Model({"fused": _Fused(method=_Method())})
    _cold(model, _fused_weights(1.0))
    live = model.fused.weight
    _update(model, _fused_weights(9.0, fused=True))
    assert torch.equal(live, torch.full((6, 3), 9.0))
    assert not is_model_dirty(model)


def test_declined_experts_deliver_no_units():
    model = _Model({"experts": _Experts()})
    weights = [(f"experts.{e}", torch.full((4,), float(e))) for e in range(4)]
    _cold(model, weights)
    expected = model.__dict__["_reload_traces"]["cold"].expected.units["experts"]
    assert {k[:2] for k in expected} == {("w", ("w1", 1)), ("w", ("w1", 2))}
    live = model.experts.w
    _update(model, [(n, w + 10) for n, w in weights])
    assert torch.equal(
        live, torch.stack([torch.full((4,), 11.0), torch.full((4,), 12.0)])
    )


def test_duplicate_unit_against_an_exact_contract_raises():
    model = _Model({"fused": _Fused(method=_Method(post="none"))})
    _cold(model, _fused_weights(1.0))
    meta = [(n, tuple(w.shape), w.dtype) for n, w in _fused_weights(1.0)]
    core.run_trace(model, meta, "trainer")
    start_reload(model, source="trainer")
    with pytest.raises(core.ReloadUnitError, match="more times"):
        model.load_weights(track_names(_fused_weights(2.0) + _fused_weights(2.0)[:1]))
    abort_reload(model)


# ---------------------------------------------------------------------------
# Abort, routing
# ---------------------------------------------------------------------------


def test_abort_puts_the_live_objects_back_and_marks_direct_modules_dirty():
    model = _Model(
        {"fused": _Fused(method=_Method(post="none")), "q": torch.nn.Module()}
    )
    _cold(model, _fused_weights(1.0))
    live = model.fused.weight
    start_reload(model)
    model.load_weights(track_names(_fused_weights(2.0)[:1]))
    abort_reload(model)
    assert model.fused.weight is live
    assert live.weight_loader.__name__ == "weight_loader"
    session = get_reload_session(model)
    assert session.dirty and session.dirty_modules == {"fused"}


def test_routing(monkeypatch):
    model = _Model({"fused": _Fused(method=_Method())})
    record_metadata_for_reloading(model)
    assert "never observed" in core.legacy_reason(model)
    with core.observe_cold_load(model):
        model.load_weights(track_names(_fused_weights(1.0)))
    assert core.legacy_reason(model) is None
    monkeypatch.setattr(core, "LEGACY", True)
    assert core.legacy_reason(model) == "VLLM_RELOAD_LEGACY=1"
    monkeypatch.setattr(core, "LEGACY", False)
    model.finalizes_weights_during_load = True
    assert core.legacy_reason(model) == "finalizes_weights_during_load"


def test_trace_detects_post_load_work_in_load_weights():
    from vllm.model_executor.model_loader.reload import is_reload_active

    class _Finalizing(_Model):
        def load_weights(self, weights):
            super().load_weights(weights)
            if is_reload_active(self):
                self.fused.weight.data.mul_(2)

    model = _Finalizing({"fused": _Fused(method=_Method())})
    record_metadata_for_reloading(model)
    meta = [(n, tuple(w.shape), w.dtype) for n, w in _fused_weights(1.0)]
    state = core.run_trace(model, meta, "trainer")
    assert state.reload_active_queried
    assert "is_reload_active" in core.legacy_reason(model, "trainer")

    class _Direct(_Model):
        def load_weights(self, weights):
            super().load_weights(weights)
            self.fused.weight.data.zero_()  # outside any loader

    model = _Direct({"fused": _Fused(method=_Method())})
    record_metadata_for_reloading(model)
    core.run_trace(model, meta, "trainer")
    assert "outside weight loaders" in core.legacy_reason(model, "trainer")


def test_legacy_models_reload_through_layerwise():
    model = _Model({"fused": _Fused(method=_Method())})
    _cold(model, _fused_weights(1.0))
    model.finalizes_weights_during_load = True
    live = model.fused.weight
    start_reload(model)
    assert not core.core_active(model)
    model.load_weights(track_names(_fused_weights(6.0)))
    finish_reload(model, _cfg())
    assert model.fused.weight is live and torch.equal(live[:2], torch.full((2, 3), 6.0))


# ---------------------------------------------------------------------------
# R4: fault injection, and R5 (CPU part): any order, any batching
# ---------------------------------------------------------------------------


def test_dropped_expert_fails_the_update():
    model = _Model({"experts": _Experts()})
    weights = [(f"experts.{e}", torch.full((4,), float(e))) for e in range(4)]
    _cold(model, weights)
    start_reload(model)
    model.load_weights(track_names([w for w in weights if w[0] != "experts.2"]))
    with pytest.raises(ReloadIncompleteError, match=r"\('w1', 2\)"):
        finish_reload(model, _cfg())
    abort_reload(model)


def test_writes_counting_would_miss_still_complete():
    """A loader placing data with index_copy_ (invisible to element counting)
    reports its units like any other: the module completes."""

    class _IndexCopy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = _Method()
            self.w = _param(torch.zeros(3, 2), self.weight_loader)

        def weight_loader(self, param, loaded_weight):
            param.data.index_copy_(0, torch.tensor([2, 0, 1]), loaded_weight)

    model = _Model({"ic": _IndexCopy()})
    _cold(model, [("ic.w", torch.ones(3, 2))])
    live = model.ic.w
    start_reload(model)
    model.load_weights(track_names([("ic.w", torch.arange(6.0).reshape(3, 2))]))
    assert core._core(model).modules["ic"].completed
    finish_reload(model, _cfg())
    assert torch.equal(live[2], torch.tensor([0.0, 1.0]))


def test_stop_mid_update_restores_objects_and_leaves_written_modules_dirty():
    def double(layer):
        replace_parameter(layer, "weight", layer.weight * 2)

    model = _Model({"fused": _Fused(method=_Method(double)), "experts": _Experts()})
    experts = [(f"experts.{e}", torch.full((4,), float(e))) for e in range(4)]
    _cold(model, _fused_weights(1.0) + experts)
    live = {n: t for n, t in model.named_parameters()}
    start_reload(model)
    model.load_weights(track_names(_fused_weights(2.0)))  # fused completes
    model.load_weights(track_names(experts[:2]))  # experts partly written
    abort_reload(model)
    assert {n: t for n, t in model.named_parameters()} == live
    session = get_reload_session(model)
    assert session.dirty_modules == {"fused", "experts"}
    # a later update that re-completes both clears the flag
    _update(model, _fused_weights(3.0) + experts)
    assert not is_model_dirty(model)


@pytest.mark.parametrize("seed", range(20))
def test_any_order_and_batching_equals_a_fresh_load(seed):
    import random

    def double(layer):
        replace_parameter(layer, "weight", layer.weight * 2)

    def build():
        return _Model(
            {
                "fused": _Fused(method=_Method(double)),
                "plain": _Fused(method=_Method(post="none")),
                "experts": _Experts(),
            }
        )

    def weights(v):
        return [
            *_fused_weights(v),  # one shard per call
            ("plain.weight", torch.full((6, 3), v + 20)),  # pre-fused
            *((f"experts.{e}", torch.full((4,), v + e)) for e in range(4)),
        ]

    model = build()
    _cold(model, weights(1.0))
    fresh = build()
    _cold(fresh, weights(5.0))
    rng = random.Random(seed)
    ws = weights(5.0)
    rng.shuffle(ws)
    start_reload(model)
    i = 0
    while i < len(ws):
        n = rng.randint(1, 3)
        model.load_weights(track_names(ws[i : i + n]))
        i += n
    finish_reload(model, _cfg())
    for (name, a), (_, b) in zip(model.named_parameters(), fresh.named_parameters()):
        assert torch.equal(a, b), name


def test_planned_writes_through_the_per_module_api():
    """An engine routing bytes itself (sharded RDT's scatter): materialize the
    module's targets, write at each target's own storage offset, record the
    units, complete the module on its own thread."""

    def double(layer):
        replace_parameter(layer, "weight", layer.weight * 2)

    model = _Model({"fused": _Fused(method=_Method(double))})
    _cold(model, _fused_weights(1.0))
    live = model.fused.weight
    start_reload(model)
    targets = core.ensure_materialized(model.fused)
    w = targets["weight"]
    w.data.as_strided((6, 3), (3, 1), w.storage_offset()).fill_(4.0)
    assert core.record_units(model.fused)  # the plan covers the module
    core.complete_module(model.fused)
    finish_reload(model, _cfg())
    assert model.fused.weight is live and torch.equal(live, torch.full((6, 3), 8.0))


def test_write_observer_flags_writes_outside_loaders(monkeypatch):
    """R2 (CI): on a real update, a write to a model tensor that no wrapped
    loader made fails the update; the framework's own writes (fills, PWAL,
    landing) don't count."""

    def double(layer):
        replace_parameter(layer, "weight", layer.weight * 2)

    monkeypatch.setattr(core, "OBSERVE_WRITES", True)
    model = _Model({"fused": _Fused(method=_Method(double))})
    _cold(model, _fused_weights(1.0))
    _update(model, _fused_weights(2.0))  # clean

    start_reload(model)
    model.load_weights(track_names(_fused_weights(3.0)))
    with torch.no_grad():
        model.fused.weight.add_(1)  # e.g. post-load work in load_weights
    with pytest.raises(RuntimeError, match="outside weight loaders"):
        finish_reload(model, _cfg())
    abort_reload(model)


def test_none_params_survive_an_update():
    """`register_parameter("bias", None)` (a linear without bias) stays
    registered as None: forwards read `self.bias`."""
    layer = _Fused(method=_Method(post="none"))
    layer.register_parameter("bias", None)
    model = _Model({"fused": layer})
    _cold(model, _fused_weights(1.0))
    _update(model, _fused_weights(2.0))
    assert "bias" in model.fused._parameters and model.fused.bias is None
    core.run_trace(
        model, [(n, tuple(w.shape), w.dtype) for n, w in _fused_weights(1.0)], "t"
    )
    assert "bias" in model.fused._parameters and model.fused.bias is None


class _AttrLoader(torch.nn.Module):
    """A loader that reads a weight attribute (like the MoE scale loaders'
    `quant_method`), and a PWAL that replaces the param with the same layout
    (`replace_parameter` keeps only `weight_loader`)."""

    def __init__(self):
        super().__init__()
        self.quant_method = _Method(
            lambda layer: replace_parameter(layer, "scale", layer.scale.data * 2)
        )
        self.scale = _param(torch.zeros(4), self.weight_loader)
        self.scale.quant_method = "block"

    def weight_loader(self, param, loaded_weight):
        assert param.quant_method == "block"
        param.data.copy_(loaded_weight)


def test_as_is_targets_keep_the_recorded_loader_attributes():
    model = _Model({"layer": _AttrLoader()})
    _cold(model, [("layer.scale", torch.full((4,), 1.0))])
    live = model.layer.scale
    assert "quant_method" not in live.__dict__
    _update(model, [("layer.scale", torch.full((4,), 3.0))])
    assert model.layer.scale is live and live[0] == 6.0
    assert "quant_method" not in live.__dict__  # lent for the update only


def test_forced_modulewise_scratch_keeps_the_recorded_loader_attributes(
    monkeypatch,
):
    monkeypatch.setattr(core, "DIRECT_ENABLED", False)
    model = _Model({"layer": _AttrLoader()})
    _cold(model, [("layer.scale", torch.full((4,), 1.0))])
    live = model.layer.scale
    _update(model, [("layer.scale", torch.full((4,), 3.0))])
    assert model.layer.scale is live and live[0] == 6.0
    assert "quant_method" not in live.__dict__


def _add_scale(layer):
    """PWAL adding a tensor (online quantization's scales)."""
    layer.register_parameter("scale", _param(torch.ones(1)))


def test_shadow_count_ignores_tensors_a_pwal_added(monkeypatch):
    monkeypatch.setattr(core, "SHADOW_COUNT", True)
    model = _Model({"fused": _Fused(method=_Method(_add_scale))})
    _cold(model, _fused_weights(1.0))
    _update(model, _fused_weights(2.0))
    assert not model.__dict__.get("_reload_shadow")


def test_param_re_tied_after_the_cold_load_counts_for_its_first_owner():
    """A checkpoint's own `lm_head` loads untied and is re-tied to the
    embeddings after the cold load: a full update reaches it through them."""
    model = _Model({"embed": torch.nn.Linear(2, 2), "head": torch.nn.Linear(2, 2)})
    for m in (model.embed, model.head):
        m.bias = None
        m.weight.requires_grad_(False).weight_loader = default_weight_loader
    _cold(
        model, [("embed.weight", torch.ones(2, 2)), ("head.weight", torch.ones(2, 2))]
    )
    model.head.weight = model.embed.weight  # re-tied
    _update(model, [("embed.weight", torch.full((2, 2), 2.0))], partial=False)
    assert model.head.weight is model.embed.weight
    assert torch.equal(model.embed.weight, torch.full((2, 2), 2.0))
