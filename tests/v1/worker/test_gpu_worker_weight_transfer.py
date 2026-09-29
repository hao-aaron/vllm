# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.v1.worker.gpu_model_runner import _get_parameter_for_reload
from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    model: nn.Module  # set by tests that drive a reload session

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.started = False
        self.finished = False
        self.reset_count = 0
        self.supports_draft_weight_update = False
        self.update_calls: list[dict] = []
        self.seen_configs: list[VllmConfig] = []

    def _record_config(self) -> None:
        self.seen_configs.append(get_current_vllm_config())

    def start_weight_update(self) -> None:
        self._record_config()
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self._record_config()
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self._record_config()
        self.finished = True

    def reset_weight_update_target(self) -> None:
        self.reset_count += 1


class _RecordingModelRunner:
    def __init__(self) -> None:
        self.seen_config: VllmConfig | None = None
        self.reset_lora_calls = 0

    def reload_weights(self) -> None:
        self.seen_config = get_current_vllm_config()

    def reset_lora_state(self) -> None:
        self.reset_lora_calls += 1


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker._weight_update_failed = False
    worker.model_runner = _RecordingModelRunner()
    return worker


def test_reload_weights_sets_current_config():
    worker = _make_worker(None)
    model_runner = _RecordingModelRunner()
    worker.model_runner = model_runner  # type: ignore[assignment]

    Worker.reload_weights(worker)

    assert model_runner.seen_config is worker.vllm_config


def test_reload_parameter_lookup_preserves_lora_module_names():
    base_layer = nn.Module()
    qweight = nn.Parameter(torch.ones(1))
    base_layer.register_parameter("qweight", qweight)
    wrapper = BaseLayerWithLoRA()
    wrapper.base_layer = base_layer
    model = nn.Module()
    model.proj = wrapper

    named_parameters = dict(model.named_parameters())
    assert set(named_parameters) == {"proj.base_layer.qweight"}
    assert named_parameters["proj.base_layer.qweight"] is qweight
    assert model.get_parameter("proj.base_layer.qweight") is qweight
    assert _get_parameter_for_reload(model, "proj.qweight") is qweight


def test_start_update_finish_delegates_to_engine():
    engine = _RecordingEngine()
    worker = _make_worker(engine)

    Worker.start_weight_update(worker)
    assert engine.started is True
    assert worker._weight_update_active is True

    Worker.update_weights(worker, {"names": ["w"]})
    assert engine.update_calls == [{"names": ["w"]}]
    assert worker._weight_update_active is True

    Worker.finish_weight_update(worker)
    assert engine.finished is True
    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    assert engine.seen_configs == [worker.vllm_config] * 3
    assert worker.model_runner.reset_lora_calls == 1


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, {"names": ["rank-1"]}), (2, {"names": []})],
)
def test_rank_local_update_selects_worker_payload(rank, expected):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = rank
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker, [{"names": ["rank-0"]}, {"names": ["rank-1"]}, {"names": []}]
    )

    assert engine.update_calls == [expected]
    assert worker._weight_update_active is True


def test_rank_local_update_uses_data_parallel_index_after_reconfigure():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = 0
    parallel_config = ParallelConfig(
        data_parallel_size=4,
        data_parallel_rank=2,
    )
    assert parallel_config.data_parallel_rank == 2
    assert parallel_config.data_parallel_index == 2

    parallel_config.reconfigure_for_independent_dp_rank()
    assert parallel_config.data_parallel_rank == 0
    assert parallel_config.data_parallel_index == 2

    worker.vllm_config.parallel_config = parallel_config
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker,
        [
            {"names": ["dp-0"]},
            {"names": ["dp-1"]},
            {"names": ["dp-2"]},
            {"names": ["dp-3"]},
        ],
    )

    assert engine.update_calls == [{"names": ["dp-2"]}]
    assert worker._weight_update_active is True


def test_finish_draft_session_keeps_lora_state():
    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    worker._set_draft_weight_update_target = lambda: None

    Worker.start_draft_weight_update(worker)
    Worker.finish_weight_update(worker)

    assert worker.model_runner.reset_lora_calls == 0


def test_double_start_raises():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="already"):
        Worker.start_weight_update(worker)


def test_update_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        Worker.update_weights(worker, {"names": ["w"]})


def test_finish_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="without a matching"):
        Worker.finish_weight_update(worker)


def test_update_resets_active_on_error():
    engine = _RecordingEngine(raise_on_update=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    # A failed update ends the session so the next start is clean.
    assert engine.reset_count == 1
    assert worker._weight_update_active is False


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)


def test_failed_finish_ends_session_and_aborts_reload():
    """A failed finish used to leave `_weight_update_active` set, so the next
    start raised "already active"; it now aborts the reload and ends the
    session."""
    from vllm.model_executor.model_loader.reload import (
        record_metadata_for_reloading,
    )

    class _FailingFinishEngine(_RecordingEngine):
        def start_weight_update(self) -> None:
            super().start_weight_update()
            from vllm.model_executor.model_loader.reload import start_reload

            start_reload(self.model)

        def finish_weight_update(self) -> None:
            raise RuntimeError("finish failed")

    engine = _FailingFinishEngine()
    engine.model = nn.Sequential(nn.Linear(2, 2, bias=False))
    live = engine.model[0].weight
    record_metadata_for_reloading(engine.model)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)
    assert engine.model[0].weight.is_meta
    with pytest.raises(RuntimeError, match="finish failed"):
        Worker.finish_weight_update(worker)
    assert worker._weight_update_active is False
    assert engine.model[0].weight is live  # abort_reload restored the tensors
    Worker.start_weight_update(worker)  # no "already active"


class _ReloadingEngine(_RecordingEngine):
    def __init__(self, raise_on_update: bool = False):
        super().__init__(raise_on_update)
        self.model = nn.Sequential(nn.Linear(2, 2, bias=False))

    def start_weight_update(self) -> None:
        super().start_weight_update()
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        super().finish_weight_update()
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, None)


def test_finish_fails_when_another_rank_failed(monkeypatch):
    """A finish that succeeded here but failed on a peer rank leaves this rank
    dirty too: the ranks run every forward together."""
    from vllm.model_executor.model_loader.reload import (
        is_model_dirty,
        record_metadata_for_reloading,
    )

    engine = _ReloadingEngine()
    record_metadata_for_reloading(engine.model)
    worker = _make_worker(engine)
    votes: list[bool] = []

    def peer_failed(ok: bool) -> bool:
        votes.append(ok)
        return False

    monkeypatch.setattr(worker, "_weight_update_ok_on_all_ranks", peer_failed)
    Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="failed on another rank"):
        Worker.finish_weight_update(worker)
    assert votes == [True]
    assert is_model_dirty(engine.model)
    assert worker._weight_update_active is False
    Worker.start_weight_update(worker)  # a new full update can start


def test_finish_after_failed_update_joins_agreement(monkeypatch):
    """A rank whose update_weights failed still takes part in finish's
    agreement (voting failure), so its peers don't wait on it forever."""
    from vllm.model_executor.model_loader.reload import (
        record_metadata_for_reloading,
    )

    engine = _ReloadingEngine(raise_on_update=True)
    record_metadata_for_reloading(engine.model)
    worker = _make_worker(engine)
    votes: list[bool] = []

    def vote(ok: bool) -> bool:
        votes.append(ok)
        return ok

    monkeypatch.setattr(worker, "_weight_update_ok_on_all_ranks", vote)
    Worker.start_weight_update(worker)
    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"x": 1})
    with pytest.raises(RuntimeError, match="already failed on this rank"):
        Worker.finish_weight_update(worker)
    assert votes == [False]
    assert not engine.finished  # the engine's finish is not run after an abort
    with pytest.raises(RuntimeError, match="without a matching start"):
        Worker.finish_weight_update(worker)  # a second finish is a misuse
