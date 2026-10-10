# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reload verifier reports what an in-place weight update would get wrong,
from structure and configuration alone."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    ReloadStatus,
    merge_reload_reports,
    record_metadata_for_reloading,
    verify_reload,
)
from vllm.model_executor.model_loader.reload.coverage import observe_load
from vllm.model_executor.model_loader.reload.verify import verify_post_update
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

UNSAFE, UNVERIFIED, SAFE = (
    ReloadStatus.UNSAFE,
    ReloadStatus.UNVERIFIED,
    ReloadStatus.SAFE,
)


class _Method(QuantizeMethodBase):
    def create_weights(self, layer, *a, **k):
        pass

    def apply(self, layer, *a, **k):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        pass


def _linear(rows=4, cols=4):
    layer = torch.nn.Module()
    layer.quant_method = _Method()
    w = torch.nn.Parameter(torch.zeros(rows, cols), requires_grad=False)
    w.weight_loader = default_weight_loader
    layer.register_parameter("weight", w)
    return layer


class _Model(torch.nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(_linear() for _ in range(n))


class _Reloadable(_Model):
    supports_reload = True

    def refresh_model(self):
        pass

    def post_load_setup(self):
        pass


class _HookedReloadable(_Reloadable):
    def process_weights_after_loading(self):
        pass


def _items(report, check):
    return [(i.module, i.status) for i in report.items if i.check == check]


def test_declarations():
    assert _items(verify_reload(_Model()), "6.1") == [("", UNVERIFIED)]
    assert _items(verify_reload(_Reloadable()), "6.1") == []
    assert verify_reload(_Reloadable()).status == SAFE
    assert _items(verify_reload(_HookedReloadable()), "6.1") == [("", UNSAFE)]

    class _Hooked(_Model):
        def process_weights_after_loading(self):
            pass

    assert _items(verify_reload(_Hooked()), "6.1") == [("", UNSAFE)]

    vetoed = _Reloadable()
    vetoed.reload_safe = False
    assert _items(verify_reload(vetoed), "6.1") == [("", UNSAFE)]


def test_monkeypatched_pwal_is_unsafe_and_wrapping_is_not():
    from vllm.model_executor.layers.quantization.wrap import wrap_quant_method

    model = _Reloadable()
    model.layers[0].quant_method.process_weights_after_loading = lambda layer: None
    wrap_quant_method(model.layers[1], after_pwal=lambda layer: None)
    assert _items(verify_reload(model), "6.1") == [("layers.0", UNSAFE)]


def test_module_pwal_without_outputs_is_unsafe():
    class _Local(torch.nn.Module):
        def process_weights_after_loading(self):
            pass

    class _Declared(_Local):
        reload_outputs = ("out",)

    model = _Reloadable()
    model.local = _Local()
    model.declared = _Declared()
    assert _items(verify_reload(model), "6.1") == [("local", UNSAFE)]


def test_undeclared_moe_method_with_kernel_is_unsafe():
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

    def moe(reload_safe):
        method = object.__new__(_StubMoEMethod)
        method.moe_kernel = object()
        method.reload_safe = reload_safe
        layer = torch.nn.Module()
        layer.quant_method = method
        return layer

    model = _Reloadable()
    model.safe, model.unsafe = moe(True), moe(False)
    assert _items(verify_reload(model), "6.1") == [("unsafe", UNSAFE)]


def test_structure_snapshot_vs_live():
    model = _Reloadable(n=3)
    record_metadata_for_reloading(model)
    model.layers[0].weight = None  # e.g. a transform-then-drop
    del model.layers[2]
    assert sorted(_items(verify_reload(model), "6.3")) == [
        ("layers.0.weight", UNSAFE),
        ("layers.2", UNSAFE),
    ]


def test_load_coverage_finds_direct_writes():
    model = _Reloadable(n=3)

    def load_weights():
        a, b, _ = model.layers
        a.weight.weight_loader(a.weight, torch.ones(4, 4))  # through the loader
        b.weight.data.copy_(torch.ones(4, 4))  # direct write

    with observe_load(model) as coverage:
        load_weights()
    model.__dict__["_reload_load_coverage"] = coverage
    assert coverage.loaded == {"layers.0.weight"}
    assert coverage.direct_writes == ["layers.1.weight"]
    assert coverage.never_loaded == ["layers.2.weight"]
    assert model.layers[0].weight.weight_loader is default_weight_loader
    assert _items(verify_reload(model), "6.4") == [
        ("layers.1.weight", UNSAFE),
        ("layers.2.weight", UNVERIFIED),
    ]


@pytest.mark.parametrize("backend", ["nccl", "sharded_rdt", "sparse_nccl"])
def test_configuration(backend):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=True),
        weight_transfer_config=SimpleNamespace(backend=backend),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        lora_config=object(),
    )
    report = verify_reload(_Reloadable(), config)
    items = _items(report, "6.5")
    assert ("", UNSAFE) in items and ("", UNVERIFIED) in items  # EPLB, LoRA
    assert len(items) == (3 if backend == "sharded_rdt" else 2)
    assert report.status == UNSAFE


def test_post_update_invariants():
    model = _Model(n=1)
    layer = model.layers[0]
    layer.weight = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
    assert verify_post_update(model) == ["layers.0.weight left on meta"]
    assert verify_post_update(model, {(id(layer), "weight")}) == []


def test_merge_reports():
    a = verify_reload(_Reloadable()).to_dict()
    b = verify_reload(_HookedReloadable()).to_dict()
    merged = merge_reload_reports([a, b])
    assert merged["status"] == "UNSAFE"
    assert [(i["rank"], i["check"]) for i in merged["items"]] == [(1, "6.1")]


class _LoadingModel(_Reloadable):
    """load_weights routes `layers.i.weight` through each param's loader,
    except layer 1, which it writes directly, and drops `ignored.*`."""

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, w in weights:
            if name.startswith("ignored."):
                continue
            param = params[name]
            if name == "layers.1.weight":
                param.data.copy_(w)
            else:
                param.weight_loader(param, w)


def test_load_coverage_dry_run_from_checkpoint_meta():
    from vllm.model_executor.model_loader.reload.verify import verify_load_coverage

    model = _LoadingModel(n=3)
    record_metadata_for_reloading(model)
    before = {n: p for n, p in model.named_parameters()}
    meta = [(f"layers.{i}.weight", (4, 4), torch.float32) for i in range(3)]
    meta.append(("ignored.bias", (4,), torch.float32))
    report = verify_load_coverage(model, meta)
    items = sorted((i.module, i.status) for i in report.items)
    assert items == [("", UNVERIFIED), ("layers.1", UNSAFE)]
    assert "ignored.bias" in report.of(UNVERIFIED)[0].reason
    # a dry run: the model is left as it was
    assert {n: p for n, p in model.named_parameters()} == before
    assert all(not p.is_meta for p in model.parameters())


def test_checkpoint_meta_from_safetensors(tmp_path):
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.reload.verify import (
        checkpoint_meta_from_safetensors,
    )

    save_file(
        {"a": torch.zeros(2, 3, dtype=torch.bfloat16), "b": torch.zeros(4)},
        str(tmp_path / "model.safetensors"),
    )
    assert sorted(checkpoint_meta_from_safetensors(str(tmp_path))) == [
        ("a", (2, 3), torch.bfloat16),
        ("b", (4,), torch.float32),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device tensors")
def test_ownership_flags_undeclared_device_tensors():
    from vllm.model_executor.model_loader.reload.refresh import (
        declare_constant,
        declare_derived,
    )

    class _Kernel:
        pass

    model = _Reloadable(n=2)
    for layer in model.layers:
        kernel = layer.quant_method.kernel = _Kernel()
        kernel.alphas = torch.ones(4, device="cuda")  # undeclared, derived
        kernel.weight_view = layer.weight  # a registered tensor: owned
        kernel.workspace = torch.empty(8, device="cuda")
        kernel.lut = torch.arange(4, device="cuda")
        declare_constant(kernel, kernel.workspace)
    declare_derived(
        model.layers[1].quant_method.kernel, model.layers[1].quant_method.kernel.alphas
    )
    model.layers[0].kv_cache = [torch.empty(2, device="cuda")]  # runtime state
    items = _items(verify_reload(model), "6.2")
    # layer indices folded: one item per attribute path
    assert items == [
        ("layers.*.quant_method.kernel.alphas", UNSAFE),  # layer 0 only
        ("layers.*.quant_method.kernel.lut", UNSAFE),
    ]


def test_enforce_refuses_only_known_wrong(monkeypatch):
    """Policy at the start of an update: items known to produce wrong
    weights refuse it (unless VLLM_RELOAD_ALLOW_UNSAFE=1); everything else,
    UNSAFE included, only warns."""
    import vllm.model_executor.model_loader.reload.verify as verify

    warned = []
    monkeypatch.setattr(verify.logger, "warning_once", lambda *a: warned.append(a))
    report = verify.ReloadReport()
    report.add("6.2", "x", "undeclared tensor", UNSAFE)
    verify.enforce(report)  # ownership is report-only
    assert not warned
    report.add("6.1", "", "root hook", UNSAFE)
    verify.enforce(report)  # warns
    assert len(warned) == 1
    report.add("6.5", "", "EPLB", UNSAFE, refuse=True)
    with pytest.raises(verify.ReloadRefusedError, match="EPLB"):
        verify.enforce(report)
    monkeypatch.setattr(verify, "ALLOW_UNSAFE", True)
    verify.enforce(report)


def test_eplb_refuses_weight_update(monkeypatch):
    """Through the reload start: an EPLB configuration refuses the update
    before anything is written."""
    import vllm.model_executor.model_loader.reload.layerwise as layerwise
    from vllm.model_executor.model_loader.reload import is_model_dirty

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=True),
        weight_transfer_config=None,
        offload_config=None,
        lora_config=None,
        model_config=None,
    )
    monkeypatch.setattr(layerwise, "_current_vllm_config", lambda: config)
    model = _Reloadable()
    record_metadata_for_reloading(model)
    with pytest.raises(Exception, match="EPLB"):
        layerwise.initialize_layerwise_reload(model)
    assert not is_model_dirty(model)
    assert all(not p.is_meta for p in model.parameters())
