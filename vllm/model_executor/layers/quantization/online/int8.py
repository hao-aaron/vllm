# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    weight_amax,
)
from vllm.model_executor.utils import is_reloading, replace_parameter


class Int8OnlineMoEMethod(OnlineMoEMethodBase):
    """Online per-channel INT8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-row to int8 during loading.
    """

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )
        from vllm.model_executor.layers.fused_moe.oracle.int8 import Int8MoeBackend

        # Triton reads the (landed) layer weights and scales through the quant
        # config: nothing weight-derived lives in the kernel.
        self.reload_safe = self.int8_backend == Int8MoeBackend.TRITON

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        staging = {
            k: torch.zeros(shape, dtype=dtype, device=layer.w13_weight.device)
            for k, (shape, dtype) in self.expert_staging_spec(layer).items()
        }
        w2_amax = weight_amax(layer.w2_weight, dim=-1)
        w2_amax = amax_for_moe_weight_quant(w2_amax, self.moe.tp_size)
        for expert in range(layer.local_num_experts):
            self.quantize_expert(
                layer, "w13_weight", expert, layer.w13_weight[expert], staging
            )
            self.quantize_expert(
                layer,
                "w2_weight",
                expert,
                layer.w2_weight[expert],
                staging,
                amax=w2_amax[expert],
            )
        self.finish_experts(layer, staging)

        layer._already_called_process_weights_after_loading = True

    # ---- per-expert completion (modulewise reload): one code path for the
    # module-level loop above and per-expert quantization during a reload.

    def per_expert_needs_collective(self) -> bool:
        # w2's per-row amax is all-reduced across TP ranks
        return self.moe.tp_size > 1

    def expert_staging_spec(
        self, layer: Module
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        E = layer.w13_weight.shape[0]
        return {
            "w13": (tuple(layer.w13_weight.shape), torch.int8),
            "w2": (tuple(layer.w2_weight.shape), torch.int8),
            "w13_scale": ((E, layer.w13_weight.shape[1]), torch.float32),
            "w2_scale": ((E, layer.w2_weight.shape[1]), torch.float32),
        }

    def quantize_expert(self, layer, name, expert, src, staging, amax=None):
        vmax = torch.iinfo(torch.int8).max
        if name == "w13_weight":
            # per-row quantization over the hidden_size dim
            scales = src.abs().amax(dim=1) / vmax
            key = "w13"
        else:
            # per-row quantization over the intermediate_size dim
            if amax is None:
                amax = weight_amax(src, dim=-1)
            scales = amax / vmax
            key = "w2"
        q = src.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
        staging[key][expert] = q.to(torch.int8)
        staging[f"{key}_scale"][expert] = scales

    def finish_experts(self, layer: Module, staging: dict[str, torch.Tensor]) -> None:
        replace_parameter(layer, "w13_weight", staging["w13"])
        replace_parameter(layer, "w2_weight", staging["w2"])
        replace_parameter(layer, "w13_scale", staging["w13_scale"])
        replace_parameter(layer, "w2_scale", staging["w2_scale"])
        self._setup_kernel(layer)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        w13, w2 = convert_to_int8_moe_kernel_format(
            int8_backend=self.int8_backend,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            layer=layer,
            w13_scale=layer.w13_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        if not (is_reloading() and self.reload_safe and self.moe_kernel is not None):
            # built once; on reload landing + refresh() update it in place
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_int8_moe_kernel(
                int8_backend=self.int8_backend,
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
        # the experts' own weight transform runs on every (re)load
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        return make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=getattr(layer, "w13_scale", None),
            w2_scale=getattr(layer, "w2_scale", None),
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            layer=layer,
        )
