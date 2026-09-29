# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert completion for online-quantized MoE.

Each expert of an open MoE module is quantized as soon as its pieces arrive, so
the module holds one expert's checkpoint-format (BF16) slot instead of its
whole fused BF16 tensor. The quantization is the quant method's own
`quantize_expert`, the same function its module-level PWAL loops over; the
kernel-format conversion (`finish_experts`) still runs at module completion.

Off by default; VLLM_RELOAD_PER_EXPERT=1 enables it.
"""

import inspect
import os
from collections import Counter

import torch

from . import direct as _direct
from .types import LayerReloadingInfo
from .utils import get_tensor_load_numel

PER_EXPERT = os.getenv("VLLM_RELOAD_PER_EXPERT", "0") == "1"
EXPERT_WEIGHTS = ("w13_weight", "w2_weight")
# staging tensor -> the live param whose bytes may host it
_STAGING_HOSTS = {"w13": "w13_weight", "w2": "w2_weight"}

# Peak number of expert slots alive at once, per module class (metric)
SLOTS_IN_FLIGHT: Counter[str] = Counter()


def eligible(
    layer: torch.nn.Module, original_loader, bound_args: inspect.BoundArguments
) -> bool:
    """Chosen per module at its first expert-weight loader call, from facts
    known up front. Anything else takes the module-level path."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    if not PER_EXPERT:
        return False
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None or not hasattr(quant_method, "quantize_expert"):
        return False
    if quant_method.per_expert_needs_collective():
        return False
    if getattr(original_loader, "__func__", None) is not RoutedExperts.weight_loader:
        return False
    loaded = bound_args.arguments.get("loaded_weight")
    # stacked [E, ...] loads complete the whole module in one call anyway
    return isinstance(loaded, torch.Tensor) and loaded.dim() == 2


class ExpertSlots:
    """Per-(param, local expert) checkpoint-format slots of one module."""

    def __init__(self, layer: torch.nn.Module, info: LayerReloadingInfo):
        self.layer = layer
        self.quant_method = layer.quant_method
        metas = {name: getattr(layer, name) for name in EXPERT_WEIGHTS}
        self.num_experts = metas["w13_weight"].shape[0]
        self.meta = metas
        # per-unit expected numel: the param's load numel over local experts
        self.expected = {
            name: get_tensor_load_numel(t) // self.num_experts
            for name, t in metas.items()
        }
        self.slots: dict[tuple[str, int], torch.Tensor] = {}
        self.count: Counter[tuple[str, int]] = Counter()
        self.done: set[tuple[str, int]] = set()
        self.last_unit: tuple[str, int] | None = None
        self.device = info.restore_device
        self.staging = self._alloc_staging(layer, info)
        self.max_in_flight = 0

    def _alloc_staging(
        self, layer: torch.nn.Module, info: LayerReloadingInfo
    ) -> dict[str, torch.Tensor]:
        """Staging holds the quantized experts in the layout `finish_experts`
        takes. With direct loading it is hosted in the live expert weights'
        bytes (same or larger footprint); scales use scratch."""
        # the staging spec reads shapes from the (meta) fused params
        spec = self.quant_method.expert_staging_spec(layer)
        live = info.kernel_tensors[0] if info.kernel_tensors is not None else {}
        hosts = {k: live.get(name) for k, name in _STAGING_HOSTS.items()}
        staging = {}
        for key, (shape, dtype) in spec.items():
            meta = torch.empty(shape, dtype=dtype, device="meta")
            host = hosts.get(key)
            if (
                _direct.DIRECT_LOAD
                and host is not None
                and _direct.plan_input(meta, host, self.device) is None
            ):
                staging[key] = _direct.checkpoint_view(host, meta)
                info.hosted[_STAGING_HOSTS[key]] = (
                    host.untyped_storage().data_ptr(),
                    host.untyped_storage().nbytes(),
                )
                _direct.PLAN_OUTCOMES.setdefault(layer, {})[f"staging:{key}"] = "hosted"
            else:
                staging[key] = torch.empty(shape, dtype=dtype, device=self.device)
                info.scratch_bytes += staging[key].nbytes
        return staging

    def provider(self, name: str):
        meta = self.meta[name]

        def target(param: torch.Tensor, expert: int) -> torch.Tensor | None:
            if param is not meta and not param.is_meta:
                return None
            unit = (name, expert)
            self.last_unit = unit
            slot = self.slots.get(unit)
            if slot is None:
                slot = torch.zeros(meta.shape[1:], dtype=meta.dtype, device=self.device)
                self.slots[unit] = slot
                self.max_in_flight = max(self.max_in_flight, len(self.slots))
                key = type(self.layer).__name__
                SLOTS_IN_FLIGHT[key] = max(SLOTS_IN_FLIGHT[key], len(self.slots))
            return slot

        return target

    def account(self, numel: int) -> None:
        unit, self.last_unit = self.last_unit, None
        if unit is None:
            return  # non-local expert: nothing written, no slot
        self.count[unit] += min(numel, self.expected[unit[0]])
        if self.count[unit] >= self.expected[unit[0]]:
            self._quantize(unit)

    def _quantize(self, unit: tuple[str, int]) -> None:
        name, expert = unit
        src = self.slots.pop(unit, None)
        if src is None:
            src = torch.zeros(
                self.meta[name].shape[1:],
                dtype=self.meta[name].dtype,
                device=self.device,
            )
        with torch.no_grad():
            self.quant_method.quantize_expert(
                self.layer, name, expert, src, self.staging
            )
        self.done.add(unit)

    def flush(self) -> None:
        """Backstop at module completion: quantize every unit that never
        reached its own total (padding miscounts, or never sent: zeros)."""
        for name in EXPERT_WEIGHTS:
            for expert in range(self.num_experts):
                if (name, expert) not in self.done:
                    self._quantize((name, expert))
