# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import importlib
import importlib.machinery
import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakKeyDictionary, ref

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

import vllm.model_executor.model_loader.reload.layerwise as reload_layerwise
import vllm.model_executor.model_loader.reload.meta as reload_meta
from vllm.config import ModelConfig
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    initialize_online_processing,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.meta import (
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader,
    default_weight_loader,
)
from vllm.platforms import current_platform


def _fp8_reload_unsupported() -> bool:
    """Whether the FP8 reload/online-quantize tests should be skipped.

    ``supports_fp8()`` returns True on MI250 (gfx90a) because the general
    quantization paths upcast FP8 weights, but gfx90a has no native FP8 and
    cannot run these reload models, so treat it as unsupported here.
    """
    if not current_platform.supports_fp8():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a

        return on_gfx90a()
    return False


class _AliasedBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.weight = torch.nn.Parameter(weight)
        self.register_buffer(
            "weight_view", self.weight.detach().view(-1), persistent=False
        )


class _ParentAliasedChildBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.conv1d = torch.nn.Linear(3, 2, bias=False)
        self.conv1d.weight.data.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "conv_weights", self.conv1d.weight.detach().view(-1), persistent=False
        )


class _ChildAliasOnlyBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Linear(3, 2, bias=False)
        self.conv1d.weight.data.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "conv_weights", self.conv1d.weight.detach().view(-1), persistent=False
        )


class _AliasedBufferWithUninitializedChildLayer(_AliasedBufferLayer):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Module()
        self.child.register_parameter(
            "lazy_weight", UninitializedParameter(requires_grad=False)
        )


class _NonPersistentBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.register_buffer("scale", torch.tensor(0.25), persistent=False)


class _ReloadableMMEncoderAttention(MMEncoderAttention):
    """Minimal stand-in to test reload lifecycle without encoder initialization."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.post_load_called = True


class _ReloadableAttentionLayer(
    torch.nn.Module,
    AttentionLayerBase,
):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.post_load_called = True


def test_move_metatensors():
    tensor = torch.empty((1, 2, 3))
    meta_tensor = to_meta_tensor(tensor)
    materialized_tensor = materialize_meta_tensor(meta_tensor)

    assert meta_tensor.device.type == "meta"
    assert tensor.device == materialized_tensor.device

    assert tensor.dtype == meta_tensor.dtype == materialized_tensor.dtype
    assert tensor.shape == meta_tensor.shape == materialized_tensor.shape
    assert tensor.__class__ == meta_tensor.__class__ == materialized_tensor.__class__
    assert tensor.__dict__ == meta_tensor.__dict__ == materialized_tensor.__dict__


@pytest.mark.parametrize(
    "layer_cls",
    [_ReloadableMMEncoderAttention, _ReloadableAttentionLayer],
)
def test_attention_reload_defers_post_load(default_vllm_config, layer_cls):
    default_vllm_config.model_config = ModelConfig()
    layer = layer_cls()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert not layer.post_load_called

    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.post_load_called
    assert torch.equal(layer.weight, loaded_weight)


@pytest.mark.parametrize(
    "layer_cls",
    [_ReloadableMMEncoderAttention, _ReloadableAttentionLayer],
)
def test_attention_first_load_processes_weights(default_vllm_config, layer_cls):
    default_vllm_config.model_config = ModelConfig()
    layer = layer_cls()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    initialize_online_processing(layer)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.post_load_called
    assert torch.equal(layer.weight, loaded_weight)


def test_reload_lifecycle():
    layer = torch.nn.Linear(2, 3)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    restore_layer_on_meta(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        meta_tensor = getattr(layer, name)
        assert tensor.dtype == meta_tensor.dtype
        assert tensor.shape == meta_tensor.shape
        assert tensor.__class__ == meta_tensor.__class__
        assert tensor.__dict__ == meta_tensor.__dict__

    materialize_layer(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        materialized_tensor = getattr(layer, name)
        assert tensor.dtype == materialized_tensor.dtype
        assert tensor.shape == materialized_tensor.shape
        assert tensor.__class__ == materialized_tensor.__class__
        assert tensor.__dict__ == materialized_tensor.__dict__


def test_restore_layer_replaces_postprocessed_tensor_attribute():
    layer = torch.nn.Linear(2, 3, bias=False)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )
    del layer.weight
    layer.weight = torch.empty(3, 2)

    restore_layer_on_meta(layer, info)

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.is_meta


def test_materialize_layer_preserves_non_meta_tensors():
    """Ensure that materialize_layer does not overwrite non meta tensors."""
    layer = torch.nn.Linear(2, 3, bias=True)

    # Create a non meta bias tensor and meta weight, which can happen with FP8
    bias_values = torch.ones(3)
    layer.bias.data.copy_(bias_values)
    layer.weight = torch.nn.Parameter(layer.weight.data.to("meta"))

    assert layer.weight.is_meta
    assert not layer.bias.is_meta

    # materialize the layer weights after the bias is initialized
    info = LayerReloadingInfo(
        restore_metadata=({}, {}),
        restore_device=torch.device("cpu"),
    )
    materialize_layer(layer, info)

    # Ensure the weight materialized off meta
    assert not layer.weight.is_meta
    assert layer.weight.device.type == "cpu"

    # Ensure that the bias is (still) not meta and values are unchanged
    assert not layer.bias.is_meta
    assert torch.equal(layer.bias.data, bias_values)


_MARLIN_SIZE_K, _MARLIN_SIZE_N, _MARLIN_GROUP_SIZE = 128, 64, 64


def _stub_marlin_ops(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda w, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )


def _make_marlin_kernel():
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    kernel = object.__new__(MarlinLinearKernel)
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        partition_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=_MARLIN_GROUP_SIZE,
        zero_points=False,
    )
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    return kernel


def _load_marlin_checkpoint_format_weights(layer):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    layer.qweight = PackedvLLMParameter(
        data=torch.zeros(_MARLIN_SIZE_K // 8, _MARLIN_SIZE_N, dtype=torch.int32),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=8,
        weight_loader=default_weight_loader,
    )
    layer.scales = GroupQuantScaleParameter(
        data=torch.ones(
            _MARLIN_SIZE_K // _MARLIN_GROUP_SIZE, _MARLIN_SIZE_N, dtype=torch.float16
        ),
        input_dim=0,
        output_dim=1,
        weight_loader=default_weight_loader,
    )


def test_marlin_post_load_does_not_own_workspace(monkeypatch, dist_init):
    """Weight reload must not create layer-owned Marlin lock storage."""
    _stub_marlin_ops(monkeypatch)
    kernel = _make_marlin_kernel()

    layer = torch.nn.Module()
    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")

    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")


@pytest.mark.parametrize("variant", ["fp8", "mxfp8", "nvfp4"])
def test_marlin_prepare_layer_does_not_own_workspace(monkeypatch, variant):
    """Weight preparation must not attach runtime workspace to model layers."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import (
        marlin_utils,
        marlin_utils_fp4,
        marlin_utils_fp8,
    )

    size_k, size_n = 128, 64

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda b_q_weight, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = torch.float16
    layer.params_dtype = torch.float16

    if variant == "fp8":
        prepare = marlin_utils_fp8.prepare_fp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_k, size_n, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )
    elif variant == "mxfp8":
        prepare = marlin_utils_fp8.prepare_mxfp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.full((size_n, size_k // 32), 127, dtype=torch.uint8),
                requires_grad=False,
            )
    else:
        prepare = marlin_utils_fp4.prepare_fp4_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(size_n, size_k // 16, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_global_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    load_checkpoint_format_weights()
    prepare(layer)
    assert not hasattr(layer, "workspace")

    # Reload: fresh checkpoint-format tensors, prepare runs again
    load_checkpoint_format_weights()
    prepare(layer)

    assert not hasattr(layer, "workspace")


def test_marlin_workspace_uses_persistent_workspace_manager(monkeypatch):
    """Calls reuse initialized locks; independent streams get separate storage."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    device = torch.device("cpu")
    manager = workspace_module.WorkspaceManager(device)
    monkeypatch.setattr(workspace_module, "_manager", manager)
    stream = "main"
    monkeypatch.setattr(torch_utils, "current_stream", lambda: stream)

    workspace = marlin_utils.get_marlin_workspace(device)
    assert workspace.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert workspace.dtype == torch.int32
    assert torch.count_nonzero(workspace) == 0

    workspace.fill_(1)
    assert marlin_utils.get_marlin_workspace(device) is workspace
    assert torch.all(workspace == 1)

    stream = "aux"
    aux_workspace = marlin_utils.get_marlin_workspace(device)
    assert aux_workspace.data_ptr() != workspace.data_ptr()
    assert torch.count_nonzero(aux_workspace) == 0

    manager.lock()
    assert marlin_utils.get_marlin_workspace(device) is aux_workspace


def test_marlin_workspace_without_manager(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(workspace_module, "_manager", None)
    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    first = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    second = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert torch.count_nonzero(first) == 0


def test_model_cleanup(dist_init, default_vllm_config):
    layer = QKVParallelLinear(2, 3, 4)
    assert layer.weight.weight_loader.__self__ is layer
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    mock_info_dict: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
        WeakKeyDictionary()
    )
    mock_info_dict[layer] = info
    layer_ref = ref(layer)

    del layer
    gc.collect()

    assert layer_ref() is None
    assert len(mock_info_dict) == 0


@pytest.mark.parametrize("is_gated", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_padded_moe_reload_releases_each_layer(
    monkeypatch, is_gated, has_bias, tp_rank, padded
):
    """Checkpoint-sized copies finish each layer without global finalization."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    hidden, intermediate, experts = 4, 3, 2
    stored_hidden, stored_intermediate = (8, 8) if padded else (hidden, intermediate)
    config = SimpleNamespace(
        hidden_dim_unpadded=hidden,
        intermediate_size_per_partition_unpadded=intermediate,
        is_act_and_mul=is_gated,
        has_bias=has_bias,
        tp_rank=tp_rank,
        tp_shard_with_padding=False,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    model = torch.nn.ModuleList()
    processed: list[torch.nn.Module] = []
    for _ in range(2):
        method = object.__new__(UnquantizedFusedMoEMethod)
        torch.nn.Module.__init__(method)
        method.moe = config
        # The regression concerns streaming reload, not kernel conversion.
        monkeypatch.setattr(method, "process_weights_after_loading", processed.append)
        layer = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = config
        layer.quant_config = None
        layer.quant_method = method
        layer.expert_map_manager = SimpleNamespace(map_global_to_local=lambda i: i)
        layer._loaded_expert_biases = set()
        method.create_weights(
            layer,
            experts,
            stored_hidden,
            stored_intermediate,
            torch.float32,
            weight_loader=layer.weight_loader,
        )
        model.append(layer)

    record_metadata_for_reloading(model)
    original_params = [dict(layer.named_parameters()) for layer in model]
    shards = ["w1", "w3", "w2"] if is_gated else ["w1", "w2"]
    for cycle in range(2):
        initialize_layerwise_reload(model)
        for layer_index, layer in enumerate(model):
            info = reload_layerwise.get_layerwise_info(layer)
            inputs = []
            expected = {
                name: torch.full_like(p, float("nan"))
                for name, p in original_params[layer_index].items()
            }
            params = dict(layer.named_parameters())
            calls = [
                (e, s, b)
                for e in range(experts)
                for s in shards
                for b in ([False, True] if has_bias else [False])
            ]
            for call_index, (expert, shard, bias) in enumerate(calls):
                name = ("w2" if shard == "w2" else "w13") + (
                    "_bias" if bias else "_weight"
                )
                shape = (
                    ((hidden,) if bias else (hidden, 2 * intermediate))
                    if shard == "w2"
                    else ((2 * intermediate,) if bias else (2 * intermediate, hidden))
                )
                weight = torch.arange(
                    torch.Size(shape).numel(), dtype=torch.float32
                ).reshape(shape)
                weight = weight + 100 * (1 + call_index + cycle)
                inputs.append(ref(weight))
                # Direct checkpoint loading is the reference for deferred reload.
                layer.weight_loader(expected[name], weight, name, shard, expert)
                param = params[name]
                param.weight_loader(param, weight, name, shard, expert)
                del weight
                if call_index != len(calls) - 1:
                    assert info.can_load(), (
                        "Layer processed before its final checkpoint shard"
                    )

            assert not info.can_load(), (
                "Padding must not defer the layer until finalization"
            )
            assert not info.loaded_weights
            assert len(processed) == cycle * len(model) + layer_index + 1
            assert all(source() is None for source in inputs)
            for name, original in original_params[layer_index].items():
                assert getattr(layer, name) is original
                # Kernel-specific tests cover padding, which is not checkpoint data.
                mask = torch.isfinite(expected[name])
                assert torch.equal(original[mask], expected[name][mask])


def test_get_numel_loaded():
    param = torch.empty(10, device="meta")
    loaded_weight = torch.empty(10)

    def complex_weight_loader(param, loaded_weight):
        param[:3] = loaded_weight[:3]
        param[5:8] = loaded_weight[5:8]
        return "value"

    args = inspect.signature(complex_weight_loader).bind(param, loaded_weight)
    num_loaded, ret = get_numel_loaded(complex_weight_loader, args)
    assert num_loaded == 6
    assert ret == "value"


def test_get_numel_loaded_caps_at_param_size():
    # composed_weight_loader copies into the param twice (the load and the
    # in-place post-load transform), but only param.numel() distinct elements
    # are loaded. get_numel_loaded must not double-count, otherwise a layer's
    # loaded-element total can be reached early and trailing params get dropped.
    param = torch.empty(10)
    loaded_weight = torch.ones(10)
    loader = composed_weight_loader(default_weight_loader, lambda x: x + 1)

    args = inspect.signature(loader).bind(param, loaded_weight)
    num_loaded, _ = get_numel_loaded(loader, args)
    assert num_loaded == 10


def test_layerwise_loading_warning_only_checks_new_layers(monkeypatch):
    layers = [torch.nn.Linear(16, 1, bias=False) for _ in range(2)]

    def partial_weight_loader(param, loaded_weight):
        param.view(-1)[: loaded_weight.numel()].copy_(loaded_weight)

    for layer in layers:
        layer.weight.requires_grad_(False)
        layer.weight.weight_loader = partial_weight_loader
        reload_layerwise.initialize_online_processing(layer)

    monkeypatch.setattr(reload_layerwise, "_has_device_incoming", lambda _: True)
    get_info_size = Mock(return_value=0)
    warning_once = Mock()
    monkeypatch.setattr(reload_layerwise, "get_info_size", get_info_size)
    monkeypatch.setattr(reload_layerwise.logger, "warning_once", warning_once)

    reload_layerwise.LOADING_LAYERS.clear()
    try:
        for layer in layers:
            for _ in range(3):
                layer.weight.weight_loader(layer.weight, torch.ones(1))
    finally:
        reload_layerwise.LOADING_LAYERS.clear()

    assert get_info_size.call_count == 2
    warning_once.assert_called_once()


class _ComposedLoaderLayer(torch.nn.Module):
    """Mimics a Mamba2 mixer's equal-numel direct params (A, D, dt_bias).

    ``A`` uses ``composed_weight_loader`` (an extra in-place transform copy),
    matching ``MambaMixer2`` where ``A`` is loaded as ``-exp(A_log)``.
    """

    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(4, dtype=torch.float32))
        self.D = torch.nn.Parameter(torch.ones(4))
        self.dt_bias = torch.nn.Parameter(torch.ones(4))
        self.A.weight_loader = composed_weight_loader(
            default_weight_loader, lambda x: -torch.exp(x.float())
        )
        self.D.weight_loader = default_weight_loader
        self.dt_bias.weight_loader = default_weight_loader


def test_layerwise_reload_composed_loader_does_not_drop_params(monkeypatch):
    # Regression test: a composed_weight_loader param (A) used to double-count
    # its elements, finalizing the layer before the trailing param (D) was
    # loaded and leaving it as uninitialized materialized memory.
    layer = _ComposedLoaderLayer()
    model = torch.nn.Sequential(layer)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(float("nan"))
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    loaded = {
        "A": torch.full((4,), 0.5),
        "dt_bias": torch.full((4,), 3.0),
        "D": torch.full((4,), 7.0),
    }

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    # Mimic real load_weights: resolve params once, then load in checkpoint
    # order with D last (the param that was dropped).
    params = dict(layer.named_parameters())
    for name in ("A", "dt_bias", "D"):
        param = params[name]
        param.weight_loader(param, loaded[name])
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.A, -torch.exp(loaded["A"]))
    assert torch.equal(layer.dt_bias, loaded["dt_bias"])
    assert torch.equal(layer.D, loaded["D"])


class _RecordingQuantMethod(QuantizeMethodBase):
    """Records the layer's bias at the moment processing runs."""

    uses_meta_device = True

    def __init__(self):
        self.bias_at_process = None

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.bias_at_process = layer.bias.detach().clone()


class _LateBiasLayer(torch.nn.Module):
    """Mimics an online-quantized linear: `weight` is created on meta by
    `create_weights()`, which wraps the loaders, and the linear base registers
    `bias` afterwards."""

    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        weight = torch.nn.Parameter(torch.empty(4, 2, device="meta"))
        weight.weight_loader = default_weight_loader
        self.register_parameter("weight", weight)
        initialize_online_processing(self)
        bias = torch.nn.Parameter(torch.zeros(4))
        bias.weight_loader = default_weight_loader
        self.register_parameter("bias", bias)


def test_online_processing_waits_for_late_registered_bias():
    # Regression test: `bias` is skipped by the meta device paths, but it is
    # still loaded by a weight loader. Excluding it from the processing trigger
    # finalized the layer one load early, so the trailing bias was written into
    # an already-processed layer (e.g. over FP8 Marlin's permuted bias).
    quant_method = _RecordingQuantMethod()
    layer = _LateBiasLayer(quant_method)
    loaded_bias = torch.full((4,), 3.0)

    layer.weight.weight_loader(layer.weight, torch.full((4, 2), 2.0))
    assert quant_method.bias_at_process is None

    layer.bias.weight_loader(layer.bias, loaded_bias)
    assert quant_method.bias_at_process is not None
    assert torch.equal(quant_method.bias_at_process, loaded_bias)


def test_layerwise_reload_skips_non_persistent_parameter_alias_buffers(monkeypatch):
    layer = _AliasedBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert layer.weight_view.untyped_storage().data_ptr() == (
        layer.weight.untyped_storage().data_ptr()
    )
    assert "weight_view" in layer._non_persistent_buffers_set
    assert "0.weight_view" not in model.state_dict()


def test_capture_layer_to_meta_skips_uninitialized_parameter_storage_ptrs():
    layer = _AliasedBufferWithUninitializedChildLayer()

    _, buffers = capture_layer_to_meta(layer)

    assert "weight_view" not in buffers


def test_layerwise_reload_skips_child_parameter_alias_buffers(monkeypatch):
    layer = _ParentAliasedChildBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_conv = torch.full_like(layer.conv1d.weight, 7.0)
    loaded_scale = torch.full_like(layer.scale, 3.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.conv1d.weight, loaded_conv)
    assert torch.equal(layer.conv_weights, loaded_conv.view(-1))
    assert layer.conv_weights.untyped_storage().data_ptr() == (
        layer.conv1d.weight.untyped_storage().data_ptr()
    )
    assert "conv_weights" in layer._non_persistent_buffers_set
    assert "0.conv_weights" not in model.state_dict()


def test_layerwise_reload_restores_alias_buffer_on_zero_size_layer(monkeypatch):
    layer = _ChildAliasOnlyBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_conv = torch.full_like(layer.conv1d.weight, 7.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.conv_weights, loaded_conv.view(-1))
    assert layer.conv_weights.untyped_storage().data_ptr() == (
        layer.conv1d.weight.untyped_storage().data_ptr()
    )
    assert "conv_weights" in layer._non_persistent_buffers_set
    assert "0.conv_weights" not in model.state_dict()


def test_layerwise_reload_preserves_unloaded_non_persistent_buffers(monkeypatch):
    layer = _NonPersistentBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)
    original_scale = layer.scale.clone()

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, original_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


def test_layerwise_reload_updates_loaded_non_persistent_buffers(monkeypatch):
    layer = _NonPersistentBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)
    loaded_scale = torch.full_like(layer.scale, 0.5)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, loaded_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


@pytest.fixture
def hpc_rope_norm(monkeypatch, default_vllm_config):
    """Import HpcRopeNorm with the external ``hpc`` package stubbed out."""
    if "hpc" not in sys.modules:
        stub = types.ModuleType("hpc")
        stub.__spec__ = importlib.machinery.ModuleSpec("hpc", loader=None)
        stub.QuantType = types.SimpleNamespace(  # type: ignore[attr-defined]
            QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR=types.SimpleNamespace(value=0)
        )
        monkeypatch.setitem(sys.modules, "hpc", stub)
    from vllm.model_executor.layers.hpc import rope_norm

    monkeypatch.setattr(rope_norm, "_hpc_rope_norm_instances", {})
    return rope_norm


def test_hpc_rope_norm_kernel_sees_refit_norm_weights(monkeypatch, hpc_rope_norm):
    """The fused HPC kernel is handed the live QK-norm weights after a refit.

    Drives the production ``_forward_impl`` with a recording ``hpc`` stub. The
    Q/K norm weights it receives must be the model's own float32 parameters,
    so a layerwise reload that rewrites them in place (same storage) is what
    the kernel sees. Previously the kernel read separate mirrors that no
    reload path refreshed.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm

    head_dim, num_heads, num_kv_heads, block_size = 128, 8, 1, 4
    layer = torch.nn.Module()
    layer.q_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.k_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.hpc_rope_norm = hpc_rope_norm.HpcRopeNorm(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cos_sin_cache=torch.ones(16, head_dim),
        use_qk_norm=True,
        fallback_qnorm=layer.q_norm,
        fallback_knorm=layer.k_norm,
        kv_cache_dtype="auto",
        layer_name="hpc_test_layer",
    )
    model = torch.nn.Sequential(layer)
    rnorm = layer.hpc_rope_norm

    calls: list[dict] = []

    def record(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sys.modules["hpc"], "rope_norm_store_kv", record, raising=False)

    def kernel_norm_weights():
        q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
        qkv = torch.zeros(1, q_size + 2 * kv_size, dtype=torch.bfloat16)
        kv_cache = torch.zeros(
            2, num_kv_heads, block_size, 2 * head_dim, dtype=torch.bfloat16
        )
        attn_layer = types.SimpleNamespace(
            _k_scale=torch.ones(1), _v_scale=torch.ones(1)
        )
        attn_metadata = types.SimpleNamespace(
            num_actual_tokens=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            max_query_len=1,
            decode_query_len=1,
            qo_indptr=None,
            qo_indptr_decode=None,
            slot_mapping=torch.tensor([4]),
            seq_lens=torch.tensor([1]),
            block_table_tensor=torch.tensor([[1]]),
            hpc_kv_written=False,
        )
        output = torch.zeros(1, q_size, dtype=torch.bfloat16)
        rnorm._forward_impl(qkv, kv_cache, attn_metadata, attn_layer, output)
        return calls[-1]["q_norm_weight"], calls[-1]["k_norm_weight"]

    def loaded(value):
        return torch.full((head_dim,), value, dtype=torch.bfloat16)

    default_weight_loader(layer.q_norm.weight, loaded(0.5))
    default_weight_loader(layer.k_norm.weight, loaded(0.25))
    q, k = kernel_norm_weights()
    assert torch.equal(q, loaded(0.5).float())
    assert torch.equal(k, loaded(0.25).float())
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.q_norm.weight.weight_loader(layer.q_norm.weight, loaded(2.0))
    layer.k_norm.weight.weight_loader(layer.k_norm.weight, loaded(3.0))
    finalize_layerwise_reload(model, model_config=None)

    q, k = kernel_norm_weights()
    assert q.dtype == k.dtype == torch.float32
    assert torch.equal(q, loaded(2.0).float())
    assert torch.equal(k, loaded(3.0).float())
    assert (q.data_ptr(), k.data_ptr()) == (q_ptr, k_ptr)
    assert not hasattr(rnorm, "qnorm_weight")


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-add-FP8_BLOCK",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-multiply-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-add-W4A16-G128",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-add-FP8_DYNAMIC",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-multiply-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-add-NVFP4A16",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_reload_weights(base_model, mul_model, add_model, tp_size, vllm_runner):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if "FP8" in base_model and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


def test_kv_scale_reload(vllm_runner):
    """Test reloading a checkpoint that contains k_scale/v_scale weights."""
    if _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    model = "nm-testing/Llama-3.2-1B-Instruct-FP8-KV"

    # Load dummy weights, then reload real checkpoint
    with vllm_runner(
        model_name=model,
        load_format="dummy",
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc(
            "update_config",
            kwargs={"overrides": {"load_config": {"load_format": "auto"}}},
        )
        llm.collective_rpc("reload_weights", kwargs={"weights_path": model})
        reloaded_perp = llm.generate_prompt_perplexity(
            ["The capital of France is the city of Paris"],
            mask=["The capital of France is"],
        )[0]

    assert reloaded_perp < 10


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model,quantization",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "fp8_per_tensor",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "fp8_per_tensor",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "mxfp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "mxfp8",
            marks=[
                pytest.mark.slow_test,
                pytest.mark.xfail(reason="mxfp4 & mla is not supported yet"),
            ],
        ),
    ],
)
def test_online_quantize_reload(
    base_model, mul_model, add_model, quantization, tp_size, vllm_runner
):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough GPU devices")

    if quantization == "fp8_per_tensor" and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


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


@requires_cuda
def test_first_touch_does_not_retain_nonlocal_experts():
    layer = _ExpertLayer()
    model = torch.nn.Sequential(layer)
    live = layer.w
    live_ptr = live.data_ptr()
    with torch.device("cuda"):
        record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    info = reload_layerwise.get_layerwise_info(layer)

    refs = []
    for e in range(8):
        incoming = torch.full((4, 6), float(e), device="cuda")
        refs.append(ref(incoming))
        layer.w.weight_loader(layer.w, incoming, expert_id=e)
        # nothing is buffered on the first-touch path
        assert info.loaded_weights == [] or not info.can_load()
        del incoming
    gc.collect()
    assert all(r() is None for r in refs), "incoming tensors were retained"
    # the module completed inside the last local call (expert 5)
    assert len(layer.quant_method.calls) == 1
    finalize_layerwise_reload(model, model_config=None)
    assert layer.w is live and layer.w.data_ptr() == live_ptr
    assert torch.equal(layer.w[0], torch.full((4, 6), 2.0, device="cuda"))
    assert torch.equal(layer.w[1], torch.full((4, 6), 5.0, device="cuda"))


@requires_cuda
def test_cpu_incoming_keeps_buffering():
    layer = _ExpertLayer()
    model = torch.nn.Sequential(layer)
    with torch.device("cuda"):
        record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    info = reload_layerwise.get_layerwise_info(layer)
    layer.w.weight_loader(layer.w, torch.full((4, 6), 1.0), expert_id=2)
    assert len(info.loaded_weights) == 1
    assert layer.w.is_meta  # nothing materialized for CPU incoming
    layer.w.weight_loader(layer.w, torch.full((4, 6), 3.0), expert_id=5)
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(layer.w[0].cpu(), torch.full((4, 6), 1.0))
    assert torch.equal(layer.w[1].cpu(), torch.full((4, 6), 3.0))


@requires_cuda
def test_first_touch_interleaved_modules_complete():
    layers = [_ExpertLayer(local=(0, 1)) for _ in range(3)]
    model = torch.nn.Sequential(*layers)
    with torch.device("cuda"):
        record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    # interleave: expert 0 of every layer, then expert 1 of every layer
    for e in (0, 1):
        for i, layer in enumerate(layers):
            layer.w.weight_loader(
                layer.w, torch.full((4, 6), 10.0 * i + e, device="cuda"), expert_id=e
            )
            if e == 0:
                # open modules hold partition scratch, nothing else
                info = reload_layerwise.get_layerwise_info(layer)
                assert info.scratch_bytes == layer.w.nbytes
    assert reload_layerwise.scratch_bytes_in_flight() == 0
    finalize_layerwise_reload(model, model_config=None)
    for i, layer in enumerate(layers):
        assert len(layer.quant_method.calls) == 1
        assert torch.equal(layer.w[1], torch.full((4, 6), 10.0 * i + 1, device="cuda"))


class _PaddedLayer(torch.nn.Module):
    """Live weight [4, 8]; the loader writes only [:, :6] (padding untouched)."""

    def __init__(self):
        super().__init__()
        self.quant_method = _ProcessRecorder()
        w = torch.nn.Parameter(torch.full((4, 8), 7.0, device="cuda"))
        w.weight_loader = lambda param, loaded: param.data[:, :6].copy_(loaded)
        w.weight_loader_numel = 24  # padding is not loadable
        self.register_parameter("weight", w)


@requires_cuda
def test_first_touch_padding_reads_zero():
    layer = _PaddedLayer()
    model = torch.nn.Sequential(layer)
    with torch.device("cuda"):
        record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, torch.ones(4, 6, device="cuda"))
    finalize_layerwise_reload(model, model_config=None)
    seen = layer.quant_method.calls[0]["weight"]
    assert torch.equal(seen[:, 6:], torch.zeros(4, 2, device="cuda"))
    assert torch.equal(layer.weight[:, 6:], torch.zeros(4, 2, device="cuda"))


@requires_cuda
def test_first_touch_late_registered_bias():
    quant_method = _RecordingQuantMethod()
    with torch.device("cuda"):
        layer = _LateBiasLayer(quant_method)
    loaded_bias = torch.full((4,), 3.0, device="cuda")
    layer.weight.weight_loader(layer.weight, torch.full((4, 2), 2.0, device="cuda"))
    assert quant_method.bias_at_process is None
    layer.bias.weight_loader(layer.bias, loaded_bias)
    assert torch.equal(quant_method.bias_at_process, loaded_bias)


class _LoaderlessLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.ones(3), requires_grad=False)


def _qkv_model():
    with torch.device("cuda"):
        qkv = QKVParallelLinear(4, 2, 2, bias=True)
        experts = _ExpertLayer()
        plain = _LoaderlessLayer()
        model = torch.nn.Sequential(qkv, experts, plain)
        record_metadata_for_reloading(model)
    return model, qkv, experts, plain


def _state(model):
    return {
        n: (t.data_ptr(), t.detach().clone(), getattr(t, "weight_loader", None))
        for n, t in list(model.named_parameters()) + list(model.named_buffers())
    }


@requires_cuda
def test_trace_loads_leaves_model_identical_and_attributes(dist_init):
    from vllm.model_executor.model_loader.reload import current_load, trace_loads

    model, qkv, experts, plain = _qkv_model()
    before = _state(model)
    seen = []

    class _Spy(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.copy_.default:
                t = current_load()
                seen.append((type(t.module).__name__, t.param_name) if t else None)
            return func(*args, **(kwargs or {}))

    with trace_loads(model) as trace, _Spy():
        w = qkv.weight
        for shard, rows in (("q", 4), ("k", 4), ("v", 4)):
            w.weight_loader(w, torch.randn(rows, 4, device="cuda"), shard)
            # `bias` stays live during reload; the trace must not write it
            b = qkv.bias
            b.weight_loader(b, torch.randn(rows, device="cuda"), shard)
        for e in range(8):
            experts.w.weight_loader(
                experts.w, torch.randn(4, 6, device="cuda"), expert_id=e
            )
        plain.gate.weight_loader(plain.gate, torch.randn(3, device="cuda"))
        assert qkv.weight.is_meta  # nothing materialized
    assert ("QKVParallelLinear", "weight") in seen
    assert ("_ExpertLayer", "w") in seen
    assert ("_LoaderlessLayer", "gate") in seen
    assert None not in seen
    assert set(trace.complete_modules()) == {qkv, experts, plain}
    after = _state(model)
    assert before.keys() == after.keys()
    for k in before:
        assert before[k][0] == after[k][0], k
        assert torch.equal(before[k][1], after[k][1]), k
        assert before[k][2] is after[k][2], k  # loaders unwrapped / unchanged
    # nothing processed
    assert experts.quant_method.calls == []


@requires_cuda
def test_complete_module_matches_counter_path(dist_init):
    from vllm.model_executor.model_loader.reload import (
        complete_module,
        ensure_materialized,
        finish_reload,
        start_reload,
    )

    model, qkv, experts, plain = _qkv_model()
    values = {e: torch.full((4, 6), float(e), device="cuda") for e in (2, 5)}
    # engine-driven: ensure_materialized + direct writes + complete_module
    start_reload(model)
    targets = ensure_materialized(experts)
    targets["w"][0].copy_(values[2])
    targets["w"][1].copy_(values[5])
    complete_module(experts)
    finish_reload(model, model_config=None)
    engine_result = experts.w.detach().clone()
    # counter-driven through the loader
    start_reload(model)
    for e, v in values.items():
        experts.w.weight_loader(experts.w, v, expert_id=e)
    finish_reload(model, model_config=None)
    assert torch.equal(engine_result, experts.w)
    assert len(experts.quant_method.calls) == 2


@requires_cuda
def test_abort_reload_after_partial_update(dist_init):
    from vllm.model_executor.model_loader.reload import abort_reload, start_reload

    model, qkv, experts, plain = _qkv_model()
    before = _state(model)
    start_reload(model)
    experts.w.weight_loader(
        experts.w, torch.zeros(4, 6, device="cuda"), expert_id=2
    )  # partial: module open with scratch
    qkv.weight.weight_loader(qkv.weight, torch.zeros(4, 4, device="cuda"), "q")
    abort_reload(model)
    after = _state(model)

    for k in before:
        assert before[k][0] == after[k][0], k
        # values intact: nothing was written to live storage yet
        assert torch.equal(before[k][1], after[k][1]), k
    from vllm.model_executor.model_loader.reload import is_model_dirty

    assert not is_model_dirty(model)
    assert reload_layerwise.scratch_bytes_in_flight() == 0
    # the model can be reloaded again afterwards
    start_reload(model)
    abort_reload(model)


@requires_cuda
def test_bake_offsets_relative_to_target():
    """A scatter baked against meta (offset 0) must land correctly in a target
    that sits at a nonzero storage offset (e.g. a view into live storage)."""
    from vllm.distributed.weight_transfer.sharded_rdt_fake import BakeSink
    from vllm.model_executor.model_loader.reload.layerwise import (
        _CURRENT_LOAD,
        LoadTarget,
    )

    layer = torch.nn.Module()
    base = torch.zeros(20, device="cuda")
    layer.register_parameter(
        "weight", torch.nn.Parameter(base[4:16].view(3, 4), requires_grad=False)
    )
    sink = BakeSink()

    class _Src:
        _name = "w"
        shape = (1, 4)
        dtype = torch.float32

        def numel(self):
            return 4

        def _key(self):
            return ("w", ())

    token = _CURRENT_LOAD.set(LoadTarget(layer, "weight"))
    try:
        dest = layer.weight.data[1:2]
        sink.copies_by_layer.clear()
        rec = sink.copies_by_layer[layer]
        # record without firing the meta copy (dest is real here)
        sink.accept_copy(dest, _Src())
    finally:
        _CURRENT_LOAD.reset(token)
    sc = rec[0]
    assert sc.offset == 4  # relative to the target, not to the storage
    t = layer.weight.data
    t.as_strided(sc.shape, sc.stride, t.storage_offset() + sc.offset).fill_(1.0)
    assert torch.equal(layer.weight[1], torch.ones(4, device="cuda"))
    assert base[:8].sum() == 0 and base[12:].sum() == 4 * 0 + 0


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


class _ModeRecorder(_ProcessRecorder):
    def __init__(self):
        super().__init__()
        self.modes: list[bool] = []

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.utils import is_reloading

        self.modes.append(is_reloading())
        super().process_weights_after_loading(layer)


def test_reload_mode_only_during_reload():
    from vllm.model_executor.utils import is_reloading, reload_mode, replace_parameter

    layer = _ExpertLayer(device="cpu")
    layer.quant_method = _ModeRecorder()
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    for e in (2, 5):
        layer.w.weight_loader(layer.w, torch.ones(4, 6), expert_id=e)
    finalize_layerwise_reload(model, model_config=None)
    assert layer.quant_method.modes == [True]
    assert not is_reloading()
    # replace_parameter(prefer_copy=True) still copies in place in reload
    # mode: MLA keeps W_UK_T / W_UV addresses this way during attention PWAL
    holder = torch.nn.Module()
    holder.p = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    old = holder.p
    with reload_mode():
        replace_parameter(holder, "p", torch.ones(2), prefer_copy=True)
    assert holder.p is old and torch.equal(old, torch.ones(2))


class _HookModel(torch.nn.Sequential):
    def __init__(self, *mods, reload_safe=False):
        super().__init__(*mods)
        self.reload_safe = reload_safe
        self.hook_calls = 0
        self.register_buffer("derived", torch.zeros(4, 6), persistent=False)
        self.refresh_calls = 0

    def process_weights_after_loading(self):
        self.hook_calls += 1

    def refresh(self):
        # kind A: derived from live params, allocated once, written in place
        self.refresh_calls += 1
        self.derived.copy_(self[0].w[0] * 2)


def test_model_hook_fail_closed_and_model_phase_refresh(monkeypatch):
    layer = _ExpertLayer(device="cpu")
    unsafe = _HookModel(layer)
    record_metadata_for_reloading(unsafe)
    warnings = []
    monkeypatch.setattr(
        reload_layerwise.logger, "warning", lambda *a: warnings.append(a[1])
    )
    initialize_layerwise_reload(unsafe)  # warns loudly by default
    assert warnings and "_HookModel" in warnings[0]
    reload_layerwise.abort_reload(unsafe)
    monkeypatch.setattr(reload_layerwise, "STRICT_MODEL_HOOK", True)
    with pytest.raises(reload_layerwise.ReloadUnsafeModelError):
        initialize_layerwise_reload(unsafe)
    monkeypatch.setattr(reload_layerwise, "STRICT_MODEL_HOOK", False)

    layer = _ExpertLayer(device="cpu")
    model = _HookModel(layer, reload_safe=True)
    record_metadata_for_reloading(model)
    derived_ptr = model.derived.data_ptr()
    initialize_layerwise_reload(model)
    for e in (2, 5):
        layer.w.weight_loader(layer.w, torch.full((4, 6), float(e)), expert_id=e)
    finalize_layerwise_reload(model, model_config=None)
    assert model.hook_calls == 0  # the free-form hook is cold-start only
    assert model.refresh_calls == 1
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


class _WqB(torch.nn.Module):
    """Stands in for DeepSeek-V4.1's MXFP8 `wq_b` linear (bytes + row scales)."""

    def __init__(self, rows, cols):
        super().__init__()
        self.quant_method = _ProcessRecorder()  # PWAL: identity layout
        for name, shape in (("weight", (rows, cols)), ("weight_scale", (rows, 2))):
            p = torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.uint8), requires_grad=False
            )
            p.weight_loader = default_weight_loader
            self.register_parameter(name, p)


class _MegaAttnLike(torch.nn.Module):
    """Parent that permutes its child's rows in place once, behind a guard,
    exactly like `DeepseekV4MegaAttnAttention.finalize_loaded_weights`."""

    def __init__(self, register_transform):
        super().__init__()
        from vllm.model_executor.model_loader.reload import attach_processing_plan

        self.heads = 2
        self.wq_b = _WqB(self.heads * 32, 8)
        self._fused_layouts_ready = False
        if register_transform:
            attach_processing_plan(self.wq_b, self._permute)

    def _permute(self, wq_b):
        from vllm.models.deepseek_v41.common.ops.fused_layout import permute_wq_b_

        permute_wq_b_(wq_b.weight.data, wq_b.weight_scale.data, self.heads)

    def finalize_loaded_weights(self):
        from vllm.model_executor.utils import is_reloading

        if self._fused_layouts_ready:
            return
        self._permute(self.wq_b)
        if not is_reloading():
            self._fused_layouts_ready = True


@pytest.mark.parametrize("register_transform", [False, True])
def test_attached_processing_plan_permutes_before_landing(register_transform):
    from vllm.models.deepseek_v41.common.ops.fused_layout import q_fused_permutation

    parent = _MegaAttnLike(register_transform)
    model = torch.nn.Sequential(parent)
    record_metadata_for_reloading(model)
    g = torch.Generator().manual_seed(0)
    A = {
        n: torch.randint(0, 255, p.shape, generator=g, dtype=torch.uint8)
        for n, p in parent.wq_b.named_parameters()
    }
    B = {
        n: torch.randint(0, 255, p.shape, generator=g, dtype=torch.uint8)
        for n, p in parent.wq_b.named_parameters()
    }
    for n, v in A.items():
        getattr(parent.wq_b, n).data.copy_(v)
    parent.finalize_loaded_weights()  # cold start: model hook
    ptr = parent.wq_b.weight.data_ptr()
    perm = q_fused_permutation(parent.heads, 32)
    assert torch.equal(parent.wq_b.weight, A["weight"][perm])

    initialize_layerwise_reload(model)
    for n, v in B.items():
        p = getattr(parent.wq_b, n)
        p.weight_loader(p, v)
    finalize_layerwise_reload(model, model_config=None)
    parent.finalize_loaded_weights()  # even if something re-ran it: guarded

    assert parent.wq_b.weight.data_ptr() == ptr
    if register_transform:
        # final (permuted) layout landed exactly once
        assert torch.equal(parent.wq_b.weight, B["weight"][perm])
        assert torch.equal(parent.wq_b.weight_scale, B["weight_scale"][perm])
    else:
        # today's behavior: the checkpoint layout lands and the guard blocks
        # the permute, so the kernel would read a mismatched layout
        assert torch.equal(parent.wq_b.weight, B["weight"])
        assert not torch.equal(parent.wq_b.weight, B["weight"][perm])


def _two_expert_layers():
    layers = [_ExpertLayer(device="cpu") for _ in range(2)]
    model = torch.nn.Sequential(*layers)
    record_metadata_for_reloading(model)
    return model, layers


def _load_layer(layer, value):
    for e in (2, 5):
        layer.w.weight_loader(layer.w, torch.full((4, 6), value), expert_id=e)


def test_failure_after_live_write_leaves_model_dirty():
    from vllm.model_executor.model_loader.reload import (
        abort_reload,
        finish_reload,
        is_model_dirty,
        start_reload,
    )
    from vllm.model_executor.model_loader.reload.layerwise import check_can_serve

    model, (a, b) = _two_expert_layers()
    start_reload(model)
    with pytest.raises(RuntimeError, match="mid weight update"):
        check_can_serve(model)
    _load_layer(a, 1.0)  # completes and lands: live storage written
    b.w.weight_loader(b.w, torch.ones(4, 6), expert_id=2)  # partial
    abort_reload(model)  # e.g. the transport failed here
    assert is_model_dirty(model)
    with pytest.raises(RuntimeError, match="refuses to serve"):
        check_can_serve(model)
    # a new update starts dirty and only a full success clears it
    start_reload(model)
    _load_layer(a, 2.0)
    _load_layer(b, 2.0)
    finish_reload(model, model_config=None)
    assert not is_model_dirty(model)
    check_can_serve(model)


def test_failure_before_any_live_write_is_clean():
    from vllm.model_executor.model_loader.reload import (
        abort_reload,
        is_model_dirty,
        start_reload,
    )

    model, (a, b) = _two_expert_layers()
    before = a.w.detach().clone()
    start_reload(model)
    a.w.weight_loader(a.w, torch.ones(4, 6), expert_id=2)  # partial, scratch only
    abort_reload(model)
    assert not is_model_dirty(model)
    assert torch.equal(a.w, before)


def test_incomplete_module_reported_or_raised(monkeypatch):
    from vllm.model_executor.model_loader.reload import (
        abort_reload,
        finish_reload,
        get_reload_session,
        start_reload,
    )

    model, (a, b) = _two_expert_layers()
    start_reload(model)
    _load_layer(a, 1.0)
    b.w.weight_loader(b.w, torch.ones(4, 6), expert_id=2)  # expert 5 never sent
    finish_reload(model, model_config=None)
    assert len(get_reload_session(model).incomplete) == 1

    monkeypatch.setattr(reload_layerwise, "REQUIRE_COMPLETE", True)
    start_reload(model)
    _load_layer(a, 1.0)
    b.w.weight_loader(b.w, torch.ones(4, 6), expert_id=2)
    with pytest.raises(reload_layerwise.ReloadIncompleteError):
        finish_reload(model, model_config=None)
    abort_reload(model)


def test_unloaded_scale_keeps_create_time_sentinel():
    """A shard scale the checkpoint lacks must read as its create-time sentinel
    after reload (as at cold start), not as zero."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        FP8_SCALE_SENTINEL,
    )

    layer = torch.nn.Module()
    layer.quant_method = _ProcessRecorder()
    scale = torch.nn.Parameter(
        torch.full((3,), FP8_SCALE_SENTINEL), requires_grad=False
    )
    scale.weight_loader = lambda param, w, i: param.data[i].copy_(w)
    layer.register_parameter("weight_scale", scale)
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    scale.data.fill_(1.0)  # "loaded" at cold start
    initialize_layerwise_reload(model)
    layer.weight_scale.weight_loader(layer.weight_scale, torch.tensor(0.5), 0)
    finalize_layerwise_reload(model, model_config=None)
    seen = layer.quant_method.calls[0]["weight_scale"]
    assert seen[0] == 0.5
    assert (seen[1:] == FP8_SCALE_SENTINEL).all()


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


@requires_cuda
def test_flashinfer_bmm_scales_refresh_in_place():
    """FlashInfer's trtllm-gen decode reads bmm1/bmm2 scales from device
    tensors allocated once and refilled by refresh(), so FULL CUDA graphs see
    q/k/v scales changed by a reload. Values match the host-float path: the
    launcher computes float(double(bmm1) * log2(e))."""
    import math

    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as fi

    impl = object.__new__(fi.FlashInferImpl)
    impl.scale = 0.125
    impl.kv_cache_dtype = "fp8"
    impl.bmm1_scale = impl.bmm2_scale = impl.o_sf_scale = None
    impl._bmm_scale_tensors = None
    impl._xqa_bmm1_tensors = None
    impl.float_scales_in_decode = False
    layer = SimpleNamespace(
        _q_scale_float=1.0,
        _k_scale_float=0.3,
        _v_scale_float=0.7,
        _o_scale_float=0.5,
        _k_scale=torch.ones(1, device="cuda"),
    )

    impl.refresh(layer)
    assert impl._bmm_scale_tensors is not None
    bmm1_log2, bmm2 = impl._bmm_scale_tensors
    ptrs = (bmm1_log2.data_ptr(), bmm2.data_ptr())
    assert layer._o_scale_float is None  # re-read at the next eager forward
    assert impl._trtllm_decode_bmm_scales(None) == (bmm1_log2, bmm2)
    assert not impl.float_scales_in_decode

    layer._k_scale_float, layer._v_scale_float = 0.9, 0.2
    impl.refresh(layer)
    assert (bmm1_log2.data_ptr(), bmm2.data_ptr()) == ptrs
    assert impl.bmm1_scale == 0.125 * 0.9
    expected_log2 = torch.tensor([0.125 * 0.9 * math.log2(math.e)], dtype=torch.float32)
    assert torch.equal(bmm1_log2.cpu(), expected_log2)
    assert torch.equal(bmm2.cpu(), torch.tensor([0.2], dtype=torch.float32))

    # XQA (SM90) reads bmm1 per query dtype (q_scale only for an FP8 query)
    xqa = impl._xqa_bmm1_tensors
    assert xqa is not None
    assert xqa[True].item() == pytest.approx(0.125 * 1.0 * 0.9)
    assert xqa[False].item() == pytest.approx(0.125 * 0.9)

    # output-quant fusion folds the o-scale into bmm2: host floats (flagged)
    assert impl._trtllm_decode_bmm_scales(torch.ones(1)) == (None, impl.bmm2_scale)
    assert impl.float_scales_in_decode


class _WeightOnly(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter | None = None):
        super().__init__()
        self.weight = (
            weight if weight is not None else torch.nn.Parameter(torch.zeros(2, 2))
        )
        self.weight.weight_loader = default_weight_loader


def _full_partial_model():
    """Embed / lm_head (tied) / proj / a rotary-like module with only a
    non-persistent buffer."""
    model = torch.nn.Module()
    model.embed = _WeightOnly()
    model.lm_head = _WeightOnly(model.embed.weight)  # tied
    model.proj = _WeightOnly()
    model.rotary = torch.nn.Module()
    model.rotary.register_buffer("cos_sin_cache", torch.ones(2), persistent=False)
    record_metadata_for_reloading(model)
    return model


@pytest.mark.parametrize("partial", [None, False, True])
@pytest.mark.parametrize("send_proj", [True, False])
def test_full_vs_partial_update(partial, send_proj):
    """A full update (partial=False) must send every module that owns a
    checkpoint tensor; a tied lm_head and a module with only non-persistent
    buffers never need their own weights. Unspecified reports, partial
    accepts."""
    model = _full_partial_model()
    initialize_layerwise_reload(model, partial=partial)
    session = reload_layerwise.get_reload_session(model)
    assert session is not None
    assert set(session.required_modules.values()) == {"embed", "proj"}
    sends = [model.embed] + ([model.proj] if send_proj else [])
    for module in sends:
        module.weight.weight_loader(module.weight, torch.full((2, 2), 3.0))
    if not send_proj and partial is False:
        with pytest.raises(reload_layerwise.ReloadIncompleteError, match="proj"):
            finalize_layerwise_reload(model, model_config=None)
        reload_layerwise.abort_reload(model)
        return
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(model.embed.weight, torch.full((2, 2), 3.0))
    assert model.lm_head.weight is model.embed.weight
    expected = 3.0 if send_proj else 0.0  # untouched modules keep old weights
    assert torch.equal(model.proj.weight, torch.full((2, 2), expected))


@requires_cuda
@pytest.mark.parametrize("keep", [True, False])
def test_dots3_vision_moe_fused_fp8_refresh(monkeypatch, keep):
    """dots3 vision MoE (kind C by default: fused FP8 buffers built from the
    experts, which are then deleted). With weight updates configured the
    experts are kept, so the fused buffers are derived state that refresh()
    recomputes in place; otherwise the block is not reload-safe."""
    import vllm.models.dots3_note.nvidia.vision as vision

    monkeypatch.setattr(vision, "_keep_experts_for_reload", lambda: keep)
    config = SimpleNamespace(
        embed_dim=128,
        pyramid_num_routed=[2],
        capacity_factor=1,
        router_scoring_func="sigmoid",
        router_scale=1.0,
        moe_intermediate_size=128,
        use_bias=False,
    )
    with torch.device("cuda"):
        mlp = vision.MoESwiGLUFFNFP8(config, layer_number=0)
    mlp.process_weights_after_loading()
    assert mlp.reload_safe is keep
    if not keep:
        assert not hasattr(mlp, "experts")
        return
    live = {
        n: (getattr(mlp, n), getattr(mlp, n).data_ptr())
        for n in vision._FUSED_FP8_BUFFERS
    }
    with torch.no_grad():
        for p in mlp.experts.parameters():
            p.mul_(2.0)  # a reload lands new expert weights
    expected = mlp._fused_fp8()
    mlp.refresh()
    for (name, (t, ptr)), exp in zip(live.items(), expected):
        now = getattr(mlp, name)
        assert now is t and now.data_ptr() == ptr
        assert torch.equal(now, exp)


@pytest.mark.parametrize(
    "module,cls_name",
    [
        ("vllm.models.kimi_k3.nvidia.model", "KimiK3ForConditionalGeneration"),
        (
            "vllm.models.deepseek_v4.common.vl_model",
            "DeepseekV4ForConditionalGeneration",
        ),
        ("vllm.models.deepseek_v41.nvidia.vl_model", "DeepseekV41ForCausalLM"),
    ],
)
@pytest.mark.parametrize("inner_safe", [True, False, None])
def test_wrapper_models_delegate_reload_safe(module, cls_name, inner_safe):
    """Wrapper models whose hook only runs the language model's hook are as
    reload-safe as that language model (undeclared: not safe)."""
    cls = getattr(importlib.import_module(module), cls_name)
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    inner = torch.nn.Module()
    if inner_safe is not None:
        inner.reload_safe = inner_safe
    model.language_model = inner
    assert model.reload_safe is bool(inner_safe)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reload_attention_scales_use_model_dtype(monkeypatch, dtype):
    """KV-cache scale params are 0-dim tensors of the default dtype, which is
    the model dtype at model init. Reload recreates them under the same dtype,
    so an fp32 checkpoint scale is rounded the same way as on a fresh load."""
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    layer = torch.nn.Module()
    method = object.__new__(BaseKVCacheMethod)
    layer.quant_method = method
    info = SimpleNamespace(loaded_weights=[], restore_device=torch.device("cpu"))
    seen = {}

    def create_weights(layer):
        BaseKVCacheMethod.create_weights(method, layer)
        seen["dtype"] = layer.k_scale.dtype

    method.create_weights = create_weights
    method.process_weights_after_loading = lambda layer: None
    monkeypatch.setattr(
        reload_layerwise, "_copy_and_restore_kernel_tensors", lambda layer, info: None
    )
    reload_layerwise._reload_attention_scales(layer, info, SimpleNamespace(dtype=dtype))
    assert seen["dtype"] == dtype


def _stub_attn(**tensors):
    attn = torch.nn.Module()
    for name, value in tensors.items():
        if isinstance(value, torch.nn.Module):
            attn.add_module(name, value)
        else:
            setattr(attn, name, value)
    return attn


def _stub_linear(rows, cols, bias=False):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    linear = torch.nn.Linear(cols, rows, bias=bias)
    linear.quant_method = object.__new__(UnquantizedLinearMethod)
    return linear


@pytest.mark.parametrize("model", ["qwen3_dflash", "gemma4_dspark"])
def test_dflash_fused_context_kv_buffers_refresh_in_place(model):
    """DFlash/DSpark drafters stack per-layer KV weights and K-norms into
    fused buffers (kind A). refresh() refills them in place from the live
    per-layer params after a reload landed them."""
    if model == "qwen3_dflash":
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model as cls
    else:
        from vllm.model_executor.models.gemma4_dspark import Gemma4DSparkModel as cls

    m = object.__new__(cls)
    torch.nn.Module.__init__(m)
    m.hidden_norm = torch.nn.LayerNorm(8)
    attns = []
    for _ in range(2):
        attns.append(
            _stub_attn(
                qkv_proj=_stub_linear(12, 8, bias=True),
                k_proj=_stub_linear(4, 8, bias=True),
                k_norm=torch.nn.LayerNorm(4),
                q_size=4,
                head_dim=4,
            )
        )
    m.layers = torch.nn.ModuleList([_stub_attn(self_attn=a) for a in attns])
    m._build_context_kv_buffers(attns, True)
    fused_name = "_fused_kv_weight" if model == "qwen3_dflash" else "_fused_k_weight"
    fused, norms = getattr(m, fused_name), m._k_norm_weights
    ptrs = (fused.data_ptr(), norms.data_ptr())
    with torch.no_grad():
        for a in attns:  # a reload lands new per-layer weights
            for p in a.parameters():
                p.add_(1.0)
    m.refresh()
    assert (getattr(m, fused_name).data_ptr(), m._k_norm_weights.data_ptr()) == ptrs
    rebuilt = object.__new__(cls)
    torch.nn.Module.__init__(rebuilt)
    rebuilt.hidden_norm = m.hidden_norm
    rebuilt._build_context_kv_buffers(attns, True)
    assert torch.equal(fused, getattr(rebuilt, fused_name))
    assert torch.equal(norms, rebuilt._k_norm_weights)


def test_k3_dspark_context_kv_norms_refresh_in_place():
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkModel

    m = object.__new__(K3DSparkModel)
    torch.nn.Module.__init__(m)
    layers = []
    for _ in range(3):
        norm = torch.nn.LayerNorm(16)
        norm.variance_epsilon = 1e-6
        attn = _stub_attn(
            kv_a_layernorm=norm,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            kv_cache_dtype="auto",
        )
        layers.append(_stub_attn(self_attn=attn))
    m.layers = torch.nn.ModuleList(layers)
    m._build_fused_context_kv_metadata()
    norms = m._context_kv_norm_weights
    ptr = norms.data_ptr()
    with torch.no_grad():
        for layer in layers:
            layer.self_attn.kv_a_layernorm.weight.mul_(3.0)
    m.refresh()
    assert m._context_kv_norm_weights is norms and norms.data_ptr() == ptr
    expected = torch.stack([la.self_attn.kv_a_layernorm.weight for la in layers])
    assert torch.equal(norms, expected)


def test_refresh_derived_state_runs_declared_refreshes():
    """Kernel-format writes (sparse patches, is_checkpoint_format=False) have
    no reload session: refresh_derived_state recomputes derived state via the
    declared quant-method and model-local refresh() hooks only."""
    from vllm.model_executor.model_loader.reload import refresh_derived_state

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

    model = torch.nn.Module()
    model.safe = torch.nn.Module()
    model.safe.quant_method = _Method(True)
    model.unsafe = torch.nn.Module()
    model.unsafe.quant_method = _Method(False)  # rebuilds; no declared refresh
    model.local = _Local()
    refresh_derived_state(model)
    assert sorted(calls) == ["local", "quant:True"]


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
    method.moe_kernel = object()  # a kernel was built at cold start
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
    from vllm.model_executor.model_loader.reload import abort_reload, is_model_dirty

    monkeypatch.setattr(reload_layerwise, "_cudagraphs_captured", lambda: graphs)
    monkeypatch.setattr(reload_layerwise, "ALLOW_UNSAFE_MOE", allow)
    model, _ = _one_weight_model(_stub_moe_method(reload_safe))
    if raises:
        with pytest.raises(reload_layerwise.ReloadUnsafeModelError, match="_StubMoE"):
            initialize_layerwise_reload(model)
        assert not is_model_dirty(model)  # refused before any live write
    else:
        initialize_layerwise_reload(model)
        abort_reload(model)


@pytest.mark.parametrize("graphs", [True, False])
def test_live_tensor_replaced_between_updates_is_caught(monkeypatch, graphs):
    from vllm.model_executor.model_loader.reload import abort_reload, is_model_dirty

    monkeypatch.setattr(reload_layerwise, "_cudagraphs_captured", lambda: graphs)
    model, layer = _one_weight_model(_ProcessRecorder())

    def update(value):
        initialize_layerwise_reload(model)
        layer.weight.weight_loader(layer.weight, torch.full((4, 4), value))
        finalize_layerwise_reload(model, model_config=None)

    update(1.0)
    layer.weight.data.copy_(torch.full((4, 4), 2.0))  # in place: fine
    update(3.0)
    assert torch.equal(layer.weight, torch.full((4, 4), 3.0))

    # replaced outside reload: graphs and built-once kernels hold the old one
    layer.weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    layer.weight.weight_loader = default_weight_loader
    if graphs:
        with pytest.raises(reload_layerwise.ReloadUnsupportedError, match=r"0\.weight"):
            initialize_layerwise_reload(model)
        assert not is_model_dirty(model)
    else:
        initialize_layerwise_reload(model)  # eager: warns
        abort_reload(model)


def test_attn_sink_padding_keeps_neg_inf_on_reload():
    from vllm.models.deepseek_v4.common.weight_loader import make_attn_sink

    layer = torch.nn.Module()
    layer.attn_sink = make_attn_sink(padded_heads=8, num_local_heads=6)
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    sink = layer.attn_sink
    sink.weight_loader(sink, torch.zeros(6))  # cold load
    ptr = sink.data_ptr()

    initialize_layerwise_reload(model)
    new = torch.arange(6.0)
    layer.attn_sink.weight_loader(layer.attn_sink, new)
    finalize_layerwise_reload(model, model_config=None)

    assert layer.attn_sink is sink and sink.data_ptr() == ptr
    assert torch.equal(sink[:6], new)
    assert torch.isneginf(sink[6:]).all()  # padding stays -inf, not zeros


class _SideBufferLayer(torch.nn.Module):
    """Two-shard param whose loader also writes a live side buffer, like KDA's
    conv1d (it copies each shard into `decode_conv1d_weight` as well)."""

    def __init__(self):
        super().__init__()
        self.quant_method = _ProcessRecorder()
        self.side = torch.zeros(2, 4)
        w = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)

        def loader(param, loaded_weight, shard_id):
            param.data[shard_id].copy_(loaded_weight)
            if not param.is_meta:
                self.side[shard_id].copy_(loaded_weight)

        w.weight_loader = loader
        self.register_parameter("weight", w)


def test_side_buffer_copies_do_not_count_toward_completion():
    layer = _SideBufferLayer()
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    orig = reload_layerwise._has_device_incoming
    reload_layerwise._has_device_incoming = lambda _: True
    try:
        layer.weight.weight_loader(layer.weight, torch.full((4,), 1.0), 0)
        # the side-buffer copy must not make shard 0 look like the whole param
        assert not layer.quant_method.calls
        layer.weight.weight_loader(layer.weight, torch.full((4,), 2.0), 1)
    finally:
        reload_layerwise._has_device_incoming = orig
    finalize_layerwise_reload(model, model_config=None)
    assert len(layer.quant_method.calls) == 1
    assert torch.equal(layer.weight, torch.tensor([[1.0] * 4, [2.0] * 4]))


class _TransposePWAL(QuantizeMethodBase):
    """PWAL that changes the layout (like DeepGEMM's MXFP8 BMM repack)."""

    def create_weights(self, layer, *a, **k):
        pass

    def apply(self, layer, *a, **k):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.utils import replace_parameter

        replace_parameter(layer, "weight", layer.weight.t().contiguous())


def test_attached_plan_runs_on_checkpoint_format_before_pwal():
    """A parent step that cold start runs from `load_weights` (before the
    child's PWAL) must see checkpoint-format tensors on reload too."""
    from vllm.model_executor.model_loader.reload import attach_processing_plan

    perm = torch.tensor([2, 0, 3, 1])
    child = torch.nn.Module()
    child.quant_method = _TransposePWAL()
    w = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
    w.weight_loader = default_weight_loader
    child.register_parameter("weight", w)

    def permute_rows(m):  # checkpoint layout: rows are output channels
        m.weight.data.copy_(m.weight.data[perm])

    attach_processing_plan(child, permute_rows)
    model = torch.nn.Sequential(child)
    record_metadata_for_reloading(model)
    a = torch.arange(12.0).view(4, 3)
    child.weight.data.copy_(a)
    permute_rows(child)  # cold start: the owner's hook, before PWAL
    child.quant_method.process_weights_after_loading(child)
    assert torch.equal(child.weight, a[perm].t())

    b = torch.arange(12.0, 24.0).view(4, 3)
    initialize_layerwise_reload(model)
    child.weight.weight_loader(child.weight, b)
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(child.weight, b[perm].t())
