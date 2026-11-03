# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""nccl_m2n destinations from the load plan, on tiny CPU models.

Each simulated worker plans its model, the workers agree on one destination
layout per name, and a receive that hands each worker its box of the new
weights (what the reshard delivers) through the agreed step must leave every
worker's model byte for byte as a cold start on those weights.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.model_loader.test_load_plan import (
    _LINEAR_SHAPES,
    _MODEL_CONFIG,
    BF16,
    _checkpoint,
    _cold_start,
    _experts_model,
    _metadata,
    _state,
    _Toy,
)
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    SKIP,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    agree_destination_layouts,
    candidate_layouts,
    layout_box,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    accepted_layouts,
    copy_staged,
    plan_step,
    staged_view,
)
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    plan_loads,
    start_reload,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_CPU = torch.device("cpu")
R = REPLICATE


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")


def _m2n_metas(shapes, allow_full_fallback=True) -> list[M2NParamMeta]:
    """Replicated on one trainer rank: the source layout does not matter here."""
    return [
        M2NParamMeta(
            m.name,
            m.dtype,
            m.shape,
            M2NLayout(M2NMesh((1, 1), 0), None),
            allow_full_fallback,
            owners=(0,),
        )
        for m in _metadata(shapes)
    ]


def _agree(metas, masks_per_worker, errors=None):
    """`agree_destination_layouts` over a fake metadata group: one trainer
    rank, then the workers."""
    errors = errors or [None] * len(masks_per_worker)
    gathered = [{"masks": None, "error": None}] + [
        {"masks": m, "error": e} for m, e in zip(masks_per_worker, errors)
    ]
    group = SimpleNamespace(all_gather_obj=lambda _: gathered)
    return agree_destination_layouts(group, metas, 1, None)


def _sync(makes, shapes):
    """Cold-start one model per worker (`makes[w]`), plan, agree, then receive
    new weights through the agreed steps. Returns the layouts by name and the
    step kinds per worker; asserts every worker equals a cold start."""
    old, new = _checkpoint(shapes, seed=0), _checkpoint(shapes, seed=1)
    models = [_cold_start(make, old) for make in makes]
    ptrs = [{n: s[3] for n, s in _state(m).items()} for m in models]
    plans = [plan_loads(m, _metadata(shapes)) for m in models]
    metas = _m2n_metas(shapes)
    n = len(makes)
    cands = [candidate_layouts(meta, n) for meta in metas]
    masks = [
        [
            accepted_layouts(
                plan.get_boxes(m.name), plan.get_dest(m.name), m.shape, c, w
            )
            for m, c in zip(metas, cands)
        ]
        for w, plan in enumerate(plans)
    ]
    chosen = _agree(metas, masks)
    kinds: list[dict[str, str]] = []
    for w, (model, plan) in enumerate(zip(models, plans)):
        start_reload(model)
        staging = torch.empty(1 << 16, dtype=torch.uint8)
        kinds.append({})
        for meta, c, index in zip(metas, cands, chosen):
            if index == SKIP:
                kinds[w][meta.name] = "skip"
                continue
            step = plan_step(
                c[index],
                plan.get_boxes(meta.name),
                plan.get_dest(meta.name),
                meta.shape,
                meta.dtype,
                w,
            )
            kinds[w][meta.name] = step.kind
            part = new[meta.name][tuple(slice(a, b) for a, b in step.box)]
            if step.kind == "direct":
                assert step.target is not None
                step.target.copy_(part)  # the reshard
            elif step.kind == "staged":
                buffer = staged_view(staging, step, meta.dtype)
                buffer.copy_(part)  # the reshard
                copy_staged(step, buffer)
            else:
                model.load_weights([(meta.name, part.clone())])
        finish_reload(model, _MODEL_CONFIG, _CPU)

    for w, (make, model) in enumerate(zip(makes, models)):
        expected = _state(_cold_start(make, new))
        for name, (_, _, data, ptr) in _state(model).items():
            assert torch.equal(
                data.view(torch.uint8), expected[name][2].view(torch.uint8)
            ), (w, name)
            assert ptr == ptrs[w][name], (w, name)
    layouts = {
        m.name: None if i == SKIP else c[i] for m, c, i in zip(metas, cands, chosen)
    }
    return layouts, kinds


def _tp(monkeypatch, make, tp_rank, tp_size=2):
    """`make` building its parallel layers as TP rank `tp_rank`, which they
    keep for loading."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    def build():
        for module in (linear, parameter):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_rank", lambda: tp_rank
            )
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_world_size", lambda: tp_size
            )
        return make()

    return build


# ------------------------------------------------------------ candidates


def test_candidates_most_sharded_first():
    meta = _m2n_metas({"w": ((8, 6), BF16)})[0]
    layouts = candidate_layouts(meta, 4)
    assert layouts[:2] == [((1, 4), (R, 0)), ((4, 1), (0, R))]
    assert ((2, 2), (R, 0)) in layouts and ((2, 2), (0, R)) in layouts
    assert ((2, 2), (R, 1)) in layouts  # 6 splits in 2 ...
    assert ((1, 4), (R, 1)) not in layouts  # ... but not evenly in 4
    assert layouts[-1] is None  # REPLICATED last


def test_layout_box_follows_row_major_ranks():
    # (R, Shard(0)) on (2, 2): workers 0..3 hold rows [0,4), [4,8), [0,4), [4,8)
    boxes = [layout_box(((2, 2), (R, 0)), (8, 3), w)[0] for w in range(4)]
    assert boxes == [(0, 4), (4, 8), (0, 4), (4, 8)]
    boxes = [layout_box(((2, 2), (0, R)), (8, 3), w)[0] for w in range(4)]
    assert boxes == [(0, 4), (0, 4), (4, 8), (4, 8)]


# ------------------------------------------------------------ agreement


def test_dp2_tp2_linear_layers_shard_and_write_directly(
    default_vllm_config, monkeypatch
):
    from tests.model_executor.model_loader.test_load_plan import _linear_model

    makes = [_tp(monkeypatch, _linear_model(), w % 2) for w in range(4)]
    layouts, kinds = _sync(makes, _LINEAR_SHAPES)
    # DP2 x TP2: only the (2, 2) mesh with TP on the inner axis matches all
    assert layouts["q_proj.weight"] == ((2, 2), (R, 0))
    assert layouts["down_proj.weight"] == ((2, 2), (R, 1))
    assert all(k == "direct" for worker in kinds for k in worker.values())


def test_kv_replication_with_dp_falls_back_to_replicated_staging(
    default_vllm_config, monkeypatch
):
    """TP4 with 2 kv heads under DP2: kv shards repeat as 0,0,1,1,0,0,1,1,
    which no 2-D layout produces, so every worker receives k/v in full and
    copies its head out (RFC §9's known case)."""
    from vllm.model_executor.layers.linear import QKVParallelLinear

    shapes = {
        "q_proj.weight": ((8, 8), BF16),
        "k_proj.weight": ((4, 8), BF16),
        "v_proj.weight": ((4, 8), BF16),
    }

    def make():
        model = _Toy()
        model.qkv_proj = QKVParallelLinear(8, 2, 4, 2, bias=False, params_dtype=BF16)
        model.stacked = {
            f"{x}_proj.weight": ("qkv_proj.weight", x) for x in ("q", "k", "v")
        }
        return model

    makes = [_tp(monkeypatch, make, w % 4, tp_size=4) for w in range(8)]
    layouts, kinds = _sync(makes, shapes)
    assert layouts["q_proj.weight"] == ((2, 4), (R, 0))
    assert layouts["k_proj.weight"] is None
    assert {k["k_proj.weight"] for k in kinds} == {"staged"}


def test_fused_experts_under_ep_and_expert_tp():
    from tests.model_executor.model_loader.test_load_plan import _EXPERT_SHAPES

    layouts, kinds = _sync(
        [_experts_model(w, 2, 0, 1) for w in range(2)], _EXPERT_SHAPES
    )
    assert layouts["experts.gate_up_proj"] == ((1, 2), (R, 0))
    assert {k["experts.gate_up_proj"] for k in kinds} == {"direct"}

    # expert TP: two boxes per worker (gate and up halves); covered only in full
    layouts, kinds = _sync(
        [_experts_model(0, 1, w, 2) for w in range(2)], _EXPERT_SHAPES
    )
    assert layouts["experts.gate_up_proj"] is None
    assert {k["experts.gate_up_proj"] for k in kinds} == {"staged"}
    assert layouts["experts.down_proj"] == ((1, 2), (R, 2))


def _replicated_model(skipped=frozenset(), loader=default_weight_loader):
    def make():
        model = _Toy()
        model.norm = torch.nn.Parameter(torch.zeros(4, dtype=BF16), False)
        model.norm.weight_loader = loader
        model.skipped = skipped
        return model

    return make


def _row_shard_model(worker, dtype=BF16):
    """A [2, 3] parameter loading rows [2w, 2w+2) of a [4, 3] checkpoint
    tensor, as a TP2 row shard does."""

    def make():
        model = _Toy()
        model.w = torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype), False)
        model.w.weight_loader = lambda p, t: p.copy_(t[2 * worker : 2 * worker + 2])
        return model

    return make


def test_casting_copy_is_staged_and_sharded():
    # an FP32 checkpoint tensor into BF16 row shards: each worker receives its
    # rows in FP32 and the staging copy casts, as the loader's copy_ did
    shapes = {"w": ((4, 3), torch.float32)}
    layouts, kinds = _sync([_row_shard_model(w) for w in range(2)], shapes)
    assert layouts["w"] == ((1, 2), (R, 0))
    assert {k["w"] for k in kinds} == {"staged"}
    # the same dtype is direct
    layouts, kinds = _sync(
        [_row_shard_model(w, torch.float32) for w in range(2)], shapes
    )
    assert {k["w"] for k in kinds} == {"direct"}


def test_fallback_and_skip():
    shapes = {"norm": ((4,), BF16)}
    # a loader that computes: load_weights on every worker, in full
    layouts, kinds = _sync(
        [_replicated_model(loader=lambda p, t: p.copy_(t * 2))] * 2, shapes
    )
    assert layouts["norm"] is None
    assert {k["norm"] for k in kinds} == {"fallback"}
    # a name every worker skips is not sent
    layouts, kinds = _sync([_replicated_model(frozenset({"norm"}))] * 2, shapes)
    assert {k["norm"] for k in kinds} == {"skip"}


def test_a_worker_that_needs_nothing_takes_any_layout():
    meta = _m2n_metas({"w": ((4, 2), BF16)})[0]
    cands = candidate_layouts(meta, 2)
    full = ((0, 4), (0, 2))
    needs_all = accepted_layouts((full,), [(full, torch.empty(4, 2))], (4, 2), cands, 0)
    needs_nothing = accepted_layouts(None, [], (4, 2), cands, 1)
    (index,) = _agree([meta], [[needs_all], [needs_nothing]])
    assert cands[index] is None
    # a fallback worker accepts only the full tensor
    assert accepted_layouts((full,), None, (4, 2), cands, 0) == 1 << cands.index(None)


def test_a_local_error_fails_every_rank():
    meta = _m2n_metas({"w": ((4,), BF16)})[0]
    with pytest.raises(RuntimeError, match="rank 2: ValueError: nope"):
        _agree([meta], [[1], None], errors=[None, "ValueError: nope"])
