# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The post-load contract (SupportsReload): quant-method PWAL always
transforms and builds kernels once, weight-derived state is refreshed in place
after every load (cold start included), landing is exact, and reload refuses
what it cannot apply correctly before any live write."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.model_executor.model_loader.reload.layerwise as reload_layerwise
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _ProcessRecorder(QuantizeMethodBase):
    """Records a clone of every tensor of the layer when processing runs."""

    def __init__(self):
        self.calls: list[dict[str, torch.Tensor]] = []

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.calls.append(
            {n: t.detach().clone() for n, t in get_layer_tensors(layer).items()}
        )


class _ExpertLayer(torch.nn.Module):
    """Two local experts out of `num_global`; non-local calls copy nothing and
    return False, like `RoutedExperts.weight_loader`."""

    def __init__(self, num_global=8, local=(2, 5), device="cuda"):
        super().__init__()
        self.local = {g: i for i, g in enumerate(local)}
        self.quant_method = _ProcessRecorder()
        w = torch.nn.Parameter(torch.randn(len(local), 4, 6, device=device))
        w.weight_loader = self.weight_loader
        self.register_parameter("w", w)

    def weight_loader(self, param, loaded_weight, expert_id):
        if expert_id not in self.local:
            return False
        param.data[self.local[expert_id]].copy_(loaded_weight)
        return True


class _BuildOnceMethod(_ProcessRecorder):
    """A reload-safe method: PWAL always transforms (doubles `w`) and builds
    its kernel object once, keyed on the object's presence (no reload flag);
    weight-derived state lives in refresh()."""

    reload_safe = True

    def __init__(self):
        super().__init__()
        self.kernel: Any = None
        self.builds = 0
        self.refreshed: list[torch.Tensor] = []

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.utils import replace_parameter

        super().process_weights_after_loading(layer)
        replace_parameter(layer, "w", layer.w * 2)
        if self.kernel is None:
            self.builds += 1
            self.kernel = SimpleNamespace(w=layer.w)

    def refresh(self, layer):
        self.refreshed.append(layer.w.detach().clone())


def _cold_config():
    return SimpleNamespace(
        quantization=None,
        dtype=torch.float32,
        word_embeddings_untied_by_checkpoint=False,
    )


class _HookModel(torch.nn.Sequential):
    """A model with a model-level hook; as `SupportsReload`, its derived
    buffer lives in refresh_model()."""

    def __init__(self, *mods, supports_reload=False):
        super().__init__(*mods)
        self.supports_reload = supports_reload
        self.calls: list[str] = []
        self.register_buffer("derived", torch.zeros(4, 6), persistent=False)

    def process_weights_after_loading(self):
        self.calls.append("hook")

    def refresh_model(self):
        # derived from live params, allocated once, written in place
        self.calls.append("refresh_model")
        self.derived.copy_(self[0].w[0] * 2)

    def post_load_setup(self):
        self.calls.append("post_load_setup")


class _NestedReloadable(torch.nn.Module):
    supports_reload = True

    def __init__(self, calls):
        super().__init__()
        self._calls = calls

    def refresh_model(self):
        self._calls.append("nested_refresh_model")

    def post_load_setup(self):
        self._calls.append("nested_post_load_setup")


def _load_layer(layer, value):
    for e in (2, 5):
        layer.w.weight_loader(layer.w, torch.full((4, 6), value), expert_id=e)


class _DerivedAttention(torch.nn.Module, AttentionLayerBase):
    """Attention whose PWAL derives a tensor from another module's live
    weight (like MLA's W_UK_T from kv_b_proj) and re-registers it."""

    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self._source = [source]
        self.register_buffer("derived", torch.zeros(4, 6))

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None

    def process_weights_after_loading(self, act_dtype):
        self.register_buffer("derived", self._source[0].w[0] * 2)


def _stub_moe_method(reload_safe: bool):
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    class _StubMoEMethod(FusedMoEMethodBase):
        def create_weights(self, layer, *a, **k):
            pass

        def get_fused_moe_quant_config(self, layer):
            return None

        def apply(self, layer, *a, **k):
            raise NotImplementedError

        def process_weights_after_loading(self, layer):
            pass

    method = _StubMoEMethod.__new__(_StubMoEMethod)
    method.moe_kernel = SimpleNamespace(refresh=lambda: None)  # built at cold start
    method.reload_safe = reload_safe
    return method


def _one_weight_model(quant_method=None):
    layer = torch.nn.Module()
    if quant_method is not None:
        layer.quant_method = quant_method
    w = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    w.weight_loader = default_weight_loader
    layer.register_parameter("weight", w)
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    return model, layer


def test_strict_landing_flags_broadcast(monkeypatch):
    dst = torch.zeros(3)
    src = torch.tensor(2.0)  # 0-dim: copy_ would silently broadcast
    assert reload_layerwise.check_exact_landing(dst, src) is not None
    assert reload_layerwise.check_exact_landing(dst, torch.ones(3)) is None
    # size-1 dims carry no layout, [1] vs [1] with different strides is exact
    assert (
        reload_layerwise.check_exact_landing(torch.zeros(4, 1), torch.zeros(1, 4).t())
        is None
    )
    assert "stride" in reload_layerwise.check_exact_landing(
        torch.zeros(4, 3), torch.zeros(3, 4).t()
    )
    layer = torch.nn.Module()
    monkeypatch.setattr(reload_layerwise, "STRICT_LANDING_RAISE", True)
    with pytest.raises(reload_layerwise.LandingMismatchError):
        reload_layerwise._land(layer, "w", dst, src)
    monkeypatch.setattr(reload_layerwise, "STRICT_LANDING_RAISE", False)
    reload_layerwise._land(layer, "w", dst, src)  # log-only: copies as before
    assert torch.equal(dst, torch.full((3,), 2.0))


def test_pwal_transforms_always_and_builds_once():
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    layer = _ExpertLayer(device="cpu")
    method = layer.quant_method = _BuildOnceMethod()
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    layer.w.data.fill_(1.0)
    process_weights_after_loading(model, _cold_config(), torch.device("cpu"))
    live = layer.w
    assert method.kernel.w is live and torch.equal(live, torch.full((2, 4, 6), 2.0))
    # the framework refreshes after the cold PWAL too
    assert len(method.refreshed) == 1

    for value in (3.0, 5.0):
        initialize_layerwise_reload(model)
        for e in (2, 5):
            layer.w.weight_loader(layer.w, torch.full((4, 6), value), expert_id=e)
        finalize_layerwise_reload(model, model_config=None)
        assert layer.w is live and method.kernel.w is live
        assert torch.equal(live, torch.full((2, 4, 6), 2 * value))
        # refresh runs after landing: it reads the new live values
        assert torch.equal(method.refreshed[-1], live)
    assert method.builds == 1 and len(method.refreshed) == 3


def test_wrap_quant_method_runs_around_pwal_on_every_load():
    """A model-specific step around a generic layer's PWAL (the replacement
    for monkeypatching an instance's PWAL or rewriting it from the model
    hook): `before_pwal` sees checkpoint-format tensors, `after_pwal` the PWAL
    results, on cold start and reload alike, and its output lands once."""
    from vllm.model_executor.layers.quantization.wrap import wrap_quant_method
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.utils import replace_parameter

    layer = _ExpertLayer(device="cpu")
    method = layer.quant_method = _BuildOnceMethod()
    seen: list[tuple[str, float]] = []

    def before(layer):
        seen.append(("before", layer.w.flatten()[0].item()))

    def after(layer):
        seen.append(("after", layer.w.flatten()[0].item()))
        replace_parameter(layer, "w", layer.w + 1)

    wrap_quant_method(layer, before_pwal=before)
    wrap_quant_method(layer, after_pwal=after)
    assert isinstance(layer.quant_method, _BuildOnceMethod)
    assert "process_weights_after_loading" not in vars(layer.quant_method)
    assert layer.quant_method.reload_safe

    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    layer.w.data.fill_(1.0)
    process_weights_after_loading(model, _cold_config(), torch.device("cpu"))
    live = layer.w
    assert seen == [("before", 1.0), ("after", 2.0)]
    assert torch.equal(live, torch.full((2, 4, 6), 3.0))

    seen.clear()
    initialize_layerwise_reload(model)
    for e in (2, 5):
        layer.w.weight_loader(layer.w, torch.full((4, 6), 4.0), expert_id=e)
    finalize_layerwise_reload(model, model_config=None)
    assert seen == [("before", 4.0), ("after", 8.0)]
    assert layer.w is live and torch.equal(live, torch.full((2, 4, 6), 9.0))
    assert method.builds == 1


def test_model_hook_warns_unless_supports_reload(monkeypatch):
    layer = _ExpertLayer(device="cpu")
    unsafe = _HookModel(layer)
    record_metadata_for_reloading(unsafe)
    warnings = []
    monkeypatch.setattr(
        reload_layerwise.logger,
        "warning",
        lambda fmt, *a: warnings.append(a[0]) if "UNSAFE" in fmt else None,
    )
    initialize_layerwise_reload(unsafe)  # warns loudly by default
    assert warnings and "_HookModel" in warnings[0]
    finalize_layerwise_reload(unsafe, model_config=None)
    monkeypatch.setattr(reload_layerwise, "STRICT_MODEL_HOOK", True)
    with pytest.raises(reload_layerwise.ReloadUnsafeModelError):
        initialize_layerwise_reload(unsafe)
    warnings.clear()
    unsafe.supports_reload = True
    initialize_layerwise_reload(unsafe)
    finalize_layerwise_reload(unsafe, model_config=None)
    assert not warnings


def test_post_load_contract_order_cold_and_reload():
    """Cold start: model hook, then declared layer refreshes, refresh_model()
    (nested SupportsReload modules first) and post_load_setup(). Reload:
    the same refreshes after landing; neither the hook nor the setup."""
    from vllm.model_executor.model_loader.reload import register_refresh
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    layer = _ExpertLayer(device="cpu")
    model = _HookModel(layer, supports_reload=True)
    model.nested = _NestedReloadable(model.calls)
    register_refresh(layer, lambda: model.calls.append("layer_refresh"))
    record_metadata_for_reloading(model)
    process_weights_after_loading(model, _cold_config(), torch.device("cpu"))
    assert model.calls == [
        "hook",
        "layer_refresh",
        "nested_refresh_model",
        "refresh_model",
        "nested_post_load_setup",
        "post_load_setup",
    ]

    model.calls.clear()
    derived_ptr = model.derived.data_ptr()
    initialize_layerwise_reload(model)
    for e in (2, 5):
        layer.w.weight_loader(layer.w, torch.full((4, 6), float(e)), expert_id=e)
    finalize_layerwise_reload(model, model_config=None)
    assert model.calls == ["layer_refresh", "nested_refresh_model", "refresh_model"]
    assert model.derived.data_ptr() == derived_ptr
    assert torch.equal(model.derived, torch.full((4, 6), 4.0))


@requires_cuda
def test_fp8_flashinfer_cutlass_quant_config_refresh():
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        make_fp8_moe_quant_config,
    )

    layer = torch.nn.Module()
    E = 4
    w1s = torch.rand(E, device="cuda") + 0.5
    w2s = torch.rand(E, device="cuda") + 0.5
    a1s = torch.tensor(0.25, device="cuda")
    a2s = torch.tensor(0.5, device="cuda")
    qc = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.FLASHINFER_CUTLASS,
        w1_scale=w1s,
        w2_scale=w2s,
        a1_scale=a1s,
        a2_scale=a2s,
        layer=layer,
    )
    ptrs = (qc.g1_alphas.data_ptr(), qc.a1_gscale.data_ptr(), qc.a2_gscale.data_ptr())
    # a reload lands new values into the same tensors ...
    w1s.mul_(3.0)
    a1s.fill_(0.125)
    a2s.fill_(2.0)
    qc.refresh()
    # ... and refresh() updates derived state in place
    assert torch.allclose(qc.g1_alphas, w1s * a1s)
    assert torch.allclose(qc.a1_gscale, torch.tensor(8.0, device="cuda"))
    assert torch.allclose(qc.a2_gscale, torch.tensor(0.5, device="cuda"))
    assert ptrs == (
        qc.g1_alphas.data_ptr(),
        qc.a1_gscale.data_ptr(),
        qc.a2_gscale.data_ptr(),
    )
    assert qc.g1_alphas is layer.g1_alphas or qc.g1_alphas.data_ptr() == (
        layer.g1_alphas.data_ptr()
    )


def test_replaced_kernel_object_refuses_update_under_graphs(monkeypatch):
    """Built-once kernel objects are recorded with the live tensors after an
    update: one rebuilt outside reload would not be the one reload refreshes
    (and captured graphs read)."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    layer = _ExpertLayer(device="cpu")
    method = layer.quant_method = _BuildOnceMethod()
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    process_weights_after_loading(model, _cold_config(), torch.device("cpu"))
    initialize_layerwise_reload(model)
    _load_layer(layer, 1.0)
    finalize_layerwise_reload(model, model_config=None)

    method.kernel = SimpleNamespace(w=layer.w)  # rebuilt outside reload
    monkeypatch.setattr(reload_layerwise, "_cudagraphs_captured", lambda: True)
    with pytest.raises(reload_layerwise.ReloadUnsupportedError, match="kernel"):
        initialize_layerwise_reload(model)


def test_refresh_derived_state_reruns_attention_phase():
    """Session-less writes (sparse patches) to a weight attention derives from
    reach the attention-derived tensor, in place (review of #59502)."""
    from vllm.model_executor.model_loader.reload import refresh_derived_state

    layer = _ExpertLayer(device="cpu")
    attn = _DerivedAttention(layer)
    model = torch.nn.Sequential(layer, attn)
    attn.process_weights_after_loading(torch.float32)
    live = attn.derived
    with torch.no_grad():
        layer.w.fill_(5.0)  # a sparse patch
    refresh_derived_state(model, SimpleNamespace(dtype=torch.float32))
    assert attn.derived is live
    assert torch.equal(live, torch.full((4, 6), 10.0))


@requires_cuda
def test_trtllm_mxfp4_situ_constants_refresh_in_place():
    """The per-expert gemm1 constants are allocated once and refilled by
    refresh() (a rebuilt kernel used to allocate fresh ones while a captured
    graph kept reading the freed ones)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsBase,
    )

    moe_config = SimpleNamespace(
        routing_method=None,
        experts_per_token=2,
        intermediate_size_per_partition=256,
        hidden_dim=512,
        hidden_dim_unpadded=512,
        num_local_experts=4,
        moe_parallel_config=SimpleNamespace(ep_rank=0),
        activation=MoEActivation.SITU,
        activation_situ_beta=1.5,
        activation_situ_linear_beta=0.75,
    )
    quant_config = SimpleNamespace(
        gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None
    )
    experts = object.__new__(TrtLlmMxfp4ExpertsBase)
    TrtLlmMxfp4ExpertsBase.__init__(experts, moe_config, quant_config)
    ptrs = (experts.gemm1_alpha.data_ptr(), experts.gemm1_beta.data_ptr())
    assert torch.equal(experts.gemm1_alpha, torch.full((4,), 1.5, device="cuda"))
    assert torch.equal(experts.gemm1_beta, torch.full((4,), 0.75, device="cuda"))
    assert experts.gemm1_clamp_limit is None
    experts.gemm1_alpha.zero_()  # e.g. garbage after an old path
    experts.refresh()
    assert torch.equal(experts.gemm1_alpha, torch.full((4,), 1.5, device="cuda"))
    assert ptrs == (experts.gemm1_alpha.data_ptr(), experts.gemm1_beta.data_ptr())


@requires_cuda
def test_fused_router_gate_refresh_in_place():
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = torch.nn.Linear(8, 4, bias=False, device="cuda")
    runner.shared_expert_gate = torch.nn.Linear(8, 1, bias=False, device="cuda")
    runner._combined_gate_weight = None
    runner.refresh()  # not built yet: nothing to do
    assert runner._combined_gate_weight is None
    runner._maybe_fuse_gate_weights()
    fused = runner._combined_gate_weight
    assert fused is not None
    ptr = fused.data_ptr()
    with torch.no_grad():
        runner.gate.weight.fill_(3.0)  # a reload lands new gate weights
    runner.refresh()
    assert runner._combined_gate_weight is fused and fused.data_ptr() == ptr
    assert torch.equal(fused[:4], torch.full((4, 8), 3.0, device="cuda"))
    assert torch.equal(fused[4:], runner.shared_expert_gate.weight)


def test_nvfp4_quant_config_gscales_refresh_in_place():
    """NVFP4 MoE: a1/a2_gscale (= 1 / activation scale) are copies the quant
    config derives; refresh() recomputes them in place from the (landed)
    activation scales, with the same expression as at build time."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        make_nvfp4_moe_quant_config,
    )

    E = 4
    a13, a2 = torch.full((E,), 0.5), torch.full((E,), 0.25)
    qc = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.FLASHINFER_TRTLLM,
        w13_scale=torch.ones(E, 8, 2),
        w2_scale=torch.ones(E, 4, 2),
        w13_scale_2=torch.ones(E),
        w2_scale_2=torch.ones(E),
        a13_scale=a13,
        a2_scale=a2,
    )
    g1, g2 = qc.a1_gscale, qc.a2_gscale
    assert torch.equal(g1, 1.0 / a13) and torch.equal(g2, 1.0 / a2)
    ptrs = (g1.data_ptr(), g2.data_ptr())
    a13.fill_(0.3)  # a reload lands new activation scales in place
    a2.fill_(0.7)
    qc.refresh()
    assert (qc.a1_gscale.data_ptr(), qc.a2_gscale.data_ptr()) == ptrs
    assert torch.equal(qc.a1_gscale, 1.0 / a13)
    assert torch.equal(qc.a2_gscale, 1.0 / a2)


def test_refresh_derived_state_runs_declared_refreshes():
    """Kernel-format writes (sparse patches, is_checkpoint_format=False) have
    no reload session: refresh_derived_state runs the declared refreshes only
    (reload-safe quant methods, layer refreshes, refresh_model()); a
    module's undeclared `refresh()` is not called."""
    from vllm.model_executor.model_loader.reload import (
        refresh_derived_state,
        register_refresh,
    )

    calls: list[str] = []

    class _Method(QuantizeMethodBase):
        def __init__(self, safe: bool):
            self.reload_safe = safe

        def create_weights(self, layer, *a, **k):
            pass

        def apply(self, layer, *a, **k):
            raise NotImplementedError

        def refresh(self, layer):
            calls.append(f"quant:{self.reload_safe}")

    class _Local(torch.nn.Module):
        def refresh(self):
            calls.append("local")

    class _Model(torch.nn.Module):
        supports_reload = True

        def refresh_model(self):
            calls.append("model")

    model = _Model()
    model.safe = torch.nn.Module()
    model.safe.quant_method = _Method(True)
    model.unsafe = torch.nn.Module()
    model.unsafe.quant_method = _Method(False)  # rebuilds; no declared refresh
    model.local = _Local()
    register_refresh(model.safe, lambda: calls.append("layer"))
    refresh_derived_state(model)
    assert calls == ["quant:True", "layer", "model"]


@pytest.mark.parametrize(
    "graphs,reload_safe,allow,raises",
    [
        (True, False, False, True),
        (True, True, False, False),  # declared
        (False, False, False, False),  # eager: a rebuilt kernel is fine
        (True, False, True, False),  # VLLM_RELOAD_ALLOW_UNSAFE_MOE=1
    ],
)
def test_undeclared_moe_method_fails_closed_under_graphs(
    monkeypatch, graphs, reload_safe, allow, raises
):
    monkeypatch.setattr(reload_layerwise, "_cudagraphs_captured", lambda: graphs)
    monkeypatch.setattr(reload_layerwise, "ALLOW_UNSAFE_MOE", allow)
    model, _ = _one_weight_model(_stub_moe_method(reload_safe))
    if raises:
        with pytest.raises(reload_layerwise.ReloadUnsafeModelError, match="_StubMoE"):
            initialize_layerwise_reload(model)
        # refused before any live write: nothing was put on meta
        assert not model[0].weight.is_meta
    else:
        initialize_layerwise_reload(model)
        finalize_layerwise_reload(model, model_config=None)


def test_reload_targets_start_from_create_time_values():
    """Elements no loader writes (padding, scales a checkpoint lacks) hold
    their create-time value after a reload, as after a fresh load: weights
    created zero, scales created as ones or sentinels. They used to hold
    whatever `torch.empty` returned."""
    layer = torch.nn.Module()
    padded = torch.nn.Parameter(torch.zeros(6, 4), requires_grad=False)
    scale = torch.nn.Parameter(torch.full((3,), -1.0), requires_grad=False)
    garbage = torch.nn.Parameter(torch.arange(4.0), requires_grad=False)

    def load_rows(param, loaded_weight):  # writes the first rows only
        param.data[: loaded_weight.shape[0]].copy_(loaded_weight)

    for name, p in (("padded", padded), ("scale", scale), ("garbage", garbage)):
        p.weight_loader = load_rows
        layer.register_parameter(name, p)
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)

    initialize_layerwise_reload(model)
    layer.padded.weight_loader(layer.padded, torch.ones(4, 4))
    layer.scale.weight_loader(layer.scale, torch.full((1,), 2.0))
    layer.garbage.weight_loader(layer.garbage, torch.full((2,), 7.0))
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(padded[4:], torch.zeros(2, 4))  # padding: create-time 0
    assert torch.equal(scale, torch.tensor([2.0, -1.0, -1.0]))  # sentinel kept
    assert torch.equal(garbage, torch.tensor([7.0, 7.0, 0.0, 0.0]))  # zero-filled
