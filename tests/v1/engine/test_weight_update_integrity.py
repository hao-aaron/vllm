# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore weight-update integrity: nothing runs between start and finish,
and a failed update leaves the engine refusing to serve (not dead) until a
full update succeeds."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCore


class _Scheduler:
    def __init__(self) -> None:
        self.pause_state = PauseState.UNPAUSED

    def set_pause_state(self, state: PauseState) -> None:
        self.pause_state = state


class _Executor:
    def __init__(self) -> None:
        self.fail: set[str] = set()
        self.calls: list[str] = []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append(method)
        if method in self.fail:
            raise RuntimeError(f"{method} failed")
        return [None]


def _engine(dp: int = 1) -> EngineCore:
    engine = object.__new__(EngineCore)
    engine.scheduler = _Scheduler()
    engine.model_executor = _Executor()
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=dp)
    )
    engine._weights_dirty = False
    engine._paused_for_weight_update = False
    return engine


def _update(engine: EngineCore) -> None:
    engine.collective_rpc("start_weight_update")
    engine.collective_rpc("update_weights", kwargs={"update_info": {}})
    engine.collective_rpc("finish_weight_update")


def test_running_scheduler_paused_during_update_and_resumed():
    engine = _engine()
    engine.collective_rpc("start_weight_update")
    assert engine.scheduler.pause_state == PauseState.PAUSED_ALL
    engine.collective_rpc("update_weights", kwargs={"update_info": {}})
    engine.collective_rpc("finish_weight_update")
    assert engine.scheduler.pause_state == PauseState.UNPAUSED
    assert not engine._weights_dirty


def test_user_pause_is_kept():
    engine = _engine()
    engine.scheduler.pause_state = PauseState.PAUSED_NEW  # paused by the user
    _update(engine)
    assert engine.scheduler.pause_state == PauseState.PAUSED_NEW


@pytest.mark.parametrize("failing", ["update_weights", "finish_weight_update"])
def test_failed_update_refuses_until_full_update(failing):
    engine = _engine()
    engine.model_executor.fail = {failing}
    with pytest.raises(RuntimeError, match="failed"):
        _update(engine)
    assert engine._weights_dirty
    assert engine.scheduler.pause_state != PauseState.UNPAUSED
    with pytest.raises(RuntimeError, match="Cannot resume generation"):
        engine.resume_scheduler()
    with pytest.raises(RuntimeError, match="Cannot accept requests"):
        engine.add_request(SimpleNamespace(request_id="r"))
    # other utilities (and weight-update RPCs) still work
    engine.collective_rpc("get_model_inspection")
    engine.model_executor.fail = set()
    _update(engine)
    assert not engine._weights_dirty
    assert engine.scheduler.pause_state == PauseState.UNPAUSED


def test_failed_start_is_not_dirty():
    engine = _engine()
    engine.model_executor.fail = {"start_weight_update"}
    with pytest.raises(RuntimeError):
        engine.collective_rpc("start_weight_update")
    assert not engine._weights_dirty
    assert engine.scheduler.pause_state == PauseState.UNPAUSED


def test_dp_engine_not_auto_paused():
    engine = _engine(dp=2)
    engine.collective_rpc("start_weight_update")
    assert engine.scheduler.pause_state == PauseState.UNPAUSED


@pytest.mark.parametrize("paused", [False, True])
def test_dp_dirty_engine_rejects_but_joins_wave(paused):
    """A dirty DP engine answers an add with an error and still steps the wave
    (dummy batches): peers' collectives need it, and the offline client waits
    for the wave to complete (it hung before)."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import DPEngineCoreProc

    engine = object.__new__(DPEngineCoreProc)
    engine.scheduler = _Scheduler()
    if paused:
        engine.scheduler.pause_state = PauseState.PAUSED_ALL
    engine._weights_dirty = True
    engine.engines_running = False
    engine.shutdown_state = None
    engine._reject_add_in_shutdown = lambda req: False
    errors = []
    engine._send_error_outputs_to_client = lambda ids, idx: errors.append(ids)
    req = SimpleNamespace(request_id="r", client_index=0)
    engine._handle_client_request(EngineCoreRequestType.ADD, (req, 0))
    assert errors == [["r"]]
    assert engine.engines_running is not paused
