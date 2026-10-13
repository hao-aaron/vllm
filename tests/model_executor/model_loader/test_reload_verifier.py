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


def test_monkeypatched_pwal_is_unsafe_and_wrapping_is_not():
    from vllm.model_executor.layers.quantization.wrap import wrap_quant_method

    model = _Reloadable()
    model.layers[0].quant_method.process_weights_after_loading = lambda layer: None
    wrap_quant_method(model.layers[1], after_pwal=lambda layer: None)
    assert _items(verify_reload(model), "6.1") == [("layers.0", UNSAFE)]


def test_module_pwal_without_quant_method_is_unsafe():
    class _Local(torch.nn.Module):
        def process_weights_after_loading(self):
            pass

    model = _Reloadable()
    model.local = _Local()
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


def test_merge_reports():
    a = verify_reload(_Reloadable()).to_dict()
    b = verify_reload(_HookedReloadable()).to_dict()
    merged = merge_reload_reports([a, b])
    assert merged["status"] == "UNSAFE"
    assert [(i["rank"], i["check"]) for i in merged["items"]] == [(1, "6.1")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device tensors")
def test_ownership_flags_undeclared_device_tensors():
    from vllm.model_executor.model_loader.reload.refresh import (
        declare_constant,
        declare_derived,
    )

    class _Kernel:
        alphas: torch.Tensor
        weight_view: torch.Tensor
        workspace: torch.Tensor
        lut: torch.Tensor

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_check_reload_support_on_an_engine(vllm_runner):
    """`LLM.check_reload_support()` runs the verifier on every rank and merges
    the reports; a dense BF16 model has no UNSAFE items."""
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
    ) as runner:
        report = runner.llm.check_reload_support()
    assert report["status"] in ("SAFE", "UNVERIFIED")
    assert not [i for i in report["items"] if i["status"] == "UNSAFE"]
