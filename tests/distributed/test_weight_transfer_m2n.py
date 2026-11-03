# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding, whole-plan
validation, and the vLLM-owned source contract. The transfer itself needs the
M2N runtime and multiple GPUs, so it is exercised separately.
"""

from contextlib import contextmanager
from functools import cached_property
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pybase64 as base64
import pytest
import ray
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    ParamMeta,
    Replicate,
    Shard,
    WeightSource,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    REPLICATE,
    REPLICATED,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    agree_destination_layouts,
    candidate_layouts,
    check_data_plane_agreement,
    check_plan_limits,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    create_m2n_communicators,
    expected_local_shape,
    m2n_param_meta,
    resolve_layout,
    source_groups,
    source_plan_digest,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    M2NWeightTransferEngine,
    M2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
    _Step,
)
from vllm.distributed.weight_transfer.m2n_trainer import (
    M2NTrainerInitInfo,
    M2NTrainerWeightTransferEngine,
)
from vllm.distributed.weight_transfer.nccl_common import (
    stateless_init_metadata_group,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


@pytest.mark.parametrize(
    "engine_type", [M2NWeightTransferEngine, M2NTrainerWeightTransferEngine]
)
def test_shutdown_releases_resources_once(monkeypatch, engine_type):
    engine = object.__new__(engine_type)
    handle = Mock()
    engine._handle = handle
    group = Mock()
    engine._comms = (
        [group, None] if engine_type is M2NTrainerWeightTransferEngine else [group]
    )
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    if engine_type is M2NTrainerWeightTransferEngine:
        engine._metadata_group = Mock()

    engine.shutdown()
    engine.shutdown()

    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    group.abort.assert_called_once_with()
    group.destroy.assert_not_called()
    assert engine._handle is None
    assert engine._comms == []
    if engine_type is M2NTrainerWeightTransferEngine:
        assert engine._metadata_group is None


@pytest.mark.parametrize(
    "engine_type", [M2NWeightTransferEngine, M2NTrainerWeightTransferEngine]
)
@pytest.mark.parametrize("failure", ["synchronize", "handle"])
def test_shutdown_attempts_all_cleanup_after_failure(monkeypatch, engine_type, failure):
    engine = object.__new__(engine_type)
    handle = Mock()
    group = Mock()
    engine._handle = handle
    engine._comms = [group]
    if engine_type is M2NTrainerWeightTransferEngine:
        engine._metadata_group = Mock()

    synchronize = Mock()
    if failure == "synchronize":
        synchronize.side_effect = RuntimeError("synchronize failed")
    else:
        handle.destroy.side_effect = RuntimeError("handle destroy failed")
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    with pytest.raises(RuntimeError, match=failure):
        engine.shutdown()

    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    group.abort.assert_called_once_with()
    assert engine._handle is None
    assert engine._comms == []

    engine.shutdown()
    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    group.abort.assert_called_once_with()


class _Shard(torch.Tensor):
    """A CPU tensor that records the streams it was handed to, since the tests
    run `_send` against a mocked stream."""

    def record_stream(self, stream):
        self.__dict__.setdefault("recorded", []).append(stream)


def _shard(shape=(4,), dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype).as_subclass(_Shard)


class _StaticSource(WeightSource):
    def __init__(self, metadata, values):
        self._metadata = metadata
        self._values = values

    @cached_property
    def metadata(self):
        return list(self._metadata)

    def iter_local(self):
        yield from self._values


def _fake_m2n():
    return SimpleNamespace(
        Mesh=lambda dims, start_rank=0: (tuple(dims), start_rank),
        Replicate=lambda: REPLICATE,
        Shard=lambda dim: dim,
        reshard=Mock(),
    )


class TestLayout:
    def test_replicated_splits_nothing(self):
        """A replicated tensor imposes no divisibility constraint. The shape
        dims are deliberately coprime with the mesh size: a layout that actually
        split the tensor would reject them."""
        mesh = M2NMesh((2, 2), start_rank=1)

        resolved_mesh, placements = resolve_layout(mesh, REPLICATED)
        assert resolved_mesh.dims == (4, 1)
        assert placements == (REPLICATE, 0)

    def test_replicated_keeps_the_same_ranks(self):
        """Replication re-factors the mesh to get a size-1 axis for its no-op
        shard. That is only sound if it still covers exactly the same GPUs."""
        mesh = M2NMesh((2, 3), start_rank=4)
        resolved_mesh, _ = resolve_layout(mesh, REPLICATED)
        assert resolved_mesh.size == mesh.size
        assert resolved_mesh.start_rank == mesh.start_rank

    def test_sharded_keeps_its_own_factorization(self):
        """Rank order decides who owns which shard, so a sharded tensor must
        not be re-factored the way a replicated one is."""
        mesh = M2NMesh((2, 3), start_rank=4)
        resolved_mesh, placements = resolve_layout(mesh, (REPLICATE, 0))
        assert resolved_mesh == mesh
        assert placements == (REPLICATE, 0)

    @pytest.mark.parametrize(
        ("placements", "match"),
        [((REPLICATE, 1), "tensor dim 1"), ((REPLICATE, 0), "divide evenly")],
    )
    def test_layout_validation_always_returns_shape_or_raises(self, placements, match):
        meta = M2NParamMeta(
            "w",
            torch.float32,
            (5,),
            M2NLayout(M2NMesh((1, 2), 0), placements),
            owners=(0, 1),
        )

        with pytest.raises(ValueError, match=match):
            expected_local_shape(meta)

    def test_cross_sharded_fanout_over_capacity_is_rejected(self):
        src = (M2NMesh((1, 2), 0), (REPLICATE, 0))
        dst = (M2NMesh((1, 65), 2), (REPLICATE, 1))

        with pytest.raises(ValueError, match=r"parameter 'w'.*MAX_TARGETS=64"):
            check_plan_limits(src, dst, "w", (2, 65))


class TestSourceLayout:
    def test_declared_layout_converts_to_an_m2n_layout(self):
        """A base `ParamMeta` layout gains m2n's size-1 second axis; an
        unowned parameter is replicated over every trainer rank."""
        meta = ParamMeta(
            "w", torch.float32, (4, 4), owners=(0, 1), placements=(Shard(1),)
        )
        assert m2n_param_meta(meta, 2).source_layout == M2NLayout(
            M2NMesh((2, 1), 0), (1, REPLICATE)
        )
        plain = ParamMeta("b", torch.float32, (4,))
        assert m2n_param_meta(plain, 2).source_layout == M2NLayout(
            M2NMesh((2, 1), 0), REPLICATED
        )

    def test_partitioned_owners_form_source_groups(self):
        """A parameter owned by some trainer ranks (a pipeline stage) travels on
        its owners' communicator, with its source mesh over them."""
        stage = ParamMeta("w", torch.float32, (4,), owners=(1,))
        meta = m2n_param_meta(stage, 2)
        assert meta.owners == (1,)
        assert meta.source_layout == M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        everyone = m2n_param_meta(ParamMeta("b", torch.float32, (4,)), 2)
        assert source_groups([meta, everyone, meta]) == [(1,), (0, 1)]
        with pytest.raises(ValueError, match="there are 2 trainer ranks"):
            m2n_param_meta(ParamMeta("x", torch.float32, (4,), owners=(2,)), 2)

    def test_trainer_boundary_rejects_duplicate_names(self):
        meta = ParamMeta("w", torch.float32, (4,))
        source = _StaticSource([meta, meta], [])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match="duplicate parameter 'w'"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    @pytest.mark.parametrize(
        ("dtype", "shape"),
        [(torch.complex64, (4,)), (torch.float32, (2, 2, 2, 2))],
    )
    def test_transport_capabilities_rejected_during_initialization(self, dtype, shape):
        source = _StaticSource([ParamMeta("w", dtype, shape)], [])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match=r"parameter 'w'.*(dtype|rank)"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    def test_uneven_shards_rejected_during_initialization(self):
        meta = ParamMeta(
            "w", torch.float32, (5,), owners=(0, 1), placements=(Shard(0),)
        )
        source = _StaticSource([meta], [])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match=r"parameter 'w'.*divide evenly"):
            engine._prepare_source_plan(source, num_trainer_ranks=2)


class TestPreflightAgreement:
    def test_source_error_is_reported_before_data_plane(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {
                "phase": "source",
                "digest": None,
                "declared_digest": None,
                "error": "ValueError: bad local manifest",
            },
            {
                "phase": "source",
                "digest": None,
                "declared_digest": None,
                "error": "ValueError: bad local manifest",
            },
        ]

        with pytest.raises(RuntimeError, match="bad local manifest"):
            check_source_plan_agreement(
                group, [], local_error="ValueError: bad local manifest"
            )

    def test_data_plane_identity_must_match(self):
        group = Mock(world_size=2)

        def gather(envelope):
            peer = dict(envelope)
            peer["mode"] = "uid"
            peer["uid_digest"] = "different"
            return [envelope, peer]

        group.all_gather_obj.side_effect = gather
        with pytest.raises(RuntimeError, match="data-plane identity disagrees"):
            check_data_plane_agreement(group, None)

    def test_runtime_error_is_shared_before_nccl4py(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {"phase": "runtime_ready", "error": None},
            {"phase": "runtime_ready", "error": "ImportError: no nccl_m2n"},
        ]

        with pytest.raises(RuntimeError, match="no nccl_m2n"):
            check_runtime_ready_agreement(group, None)


class TestWireTypes:
    @staticmethod
    def _param(
        name="w",
        dtype_name="bfloat16",
        shape=(16, 16),
        src_mesh_dims=(1, 1),
        src_placements=None,
        owners=(0,),
    ):
        return M2NWireParam(
            name=name,
            dtype_name=dtype_name,
            shape=shape,
            src_mesh_dims=src_mesh_dims,
            src_placements=src_placements,
            owners=owners,
            allow_full_fallback=True,
        )

    def _init_info(self, **overrides):
        param = self._param()
        fields = dict(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=3,
            source_digest=source_plan_digest([param]),
            params=[param.to_dict()],
        )
        fields.update(overrides)
        if "params" in overrides and "source_digest" not in overrides:
            params = [M2NWireParam.from_dict(value) for value in fields["params"]]
            fields["source_digest"] = source_plan_digest(params)
        return M2NWeightTransferInitInfo(**fields)

    def test_accepts_a_consistent_plan(self):
        assert [param.name for param in self._init_info().parse_wire_params()] == ["w"]

    def test_m2n_communicator_uses_nccl4py_uid_under_unpinned_env(self, monkeypatch):
        nccl = Mock()
        group = Mock(rank=1, world_size=3)
        unique_id = object()
        communicator = object()
        nccl.UniqueId.from_bytes.return_value = unique_id
        active = False

        @contextmanager
        def unpinned():
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def init(**kwargs):
            assert active
            assert kwargs == {"nranks": 3, "rank": 1, "unique_id": unique_id}
            return communicator

        nccl.Communicator.init.side_effect = init
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_common.unpinned_nccl_env",
            unpinned,
        )

        result = create_m2n_communicators(nccl, group, b"shared-id", [(0,)], 1)

        assert result == [communicator]
        nccl.UniqueId.from_bytes.assert_called_once_with(b"shared-id")
        nccl.Communicator.init.assert_called_once_with(
            nranks=3, rank=1, unique_id=unique_id
        )
        group.broadcast_obj.assert_not_called()

    def test_communicator_per_source_group(self):
        """Trainer rank 1 joins only its group's communicator, as rank 0 of it;
        the worker (metadata rank 2) joins both, after the group's trainers."""
        nccl = Mock()
        nccl.Communicator.init.side_effect = lambda **kw: (kw["nranks"], kw["rank"])
        group = Mock(world_size=3)
        group.broadcast_obj.return_value = {"unique_ids": [b"a", b"b"], "error": None}
        groups = [(0, 1), (1,)]

        group.rank = 1
        assert create_m2n_communicators(nccl, group, None, groups, 2) == [
            (3, 1),
            (2, 0),
        ]
        group.rank = 2
        assert create_m2n_communicators(nccl, group, None, groups, 2) == [
            (3, 2),
            (2, 1),
        ]
        group.rank = 0
        assert create_m2n_communicators(nccl, group, None, groups, 2) == [
            (3, 0),
            None,
        ]
        with pytest.raises(ValueError, match="pre-shared"):
            create_m2n_communicators(nccl, group, b"uid", groups, 2)

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda value: value.pop("shape"),
            lambda value: value.update(extra="field"),
            lambda value: value.update(src_mesh_dims=[1]),
            lambda value: value.update(src_placements=[0, 1]),
        ],
    )
    def test_malformed_wire_parameter_rejected(self, mutation):
        value = self._param().to_dict()
        mutation(value)
        with pytest.raises((TypeError, ValueError)):
            M2NWireParam.from_dict(value)

    def test_source_digest_authenticates_ordered_plan(self):
        params = [self._param("a"), self._param("b")]
        info = self._init_info(
            params=[param.to_dict() for param in params],
            source_digest=source_plan_digest(list(reversed(params))),
        )
        with pytest.raises(ValueError, match="source digest"):
            info.parse_wire_params()

    @pytest.fixture
    def worker_init_env(self, monkeypatch):
        metadata_group = Mock(rank=1, world_size=3)
        metadata_group.all_gather_obj.side_effect = lambda envelope: [envelope] * 3
        init_metadata = Mock(return_value=metadata_group)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", Mock()
        )
        handle = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(Mock(), 0, handle)),
        )
        communicator = Mock()
        make_communicator = Mock(return_value=[communicator])
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicators",
            make_communicator,
        )
        monkeypatch.setattr(M2NWeightTransferEngine, "_plan_loads", Mock())
        vllm_config = SimpleNamespace(
            parallel_config=Mock(pipeline_parallel_size=1),
            model_config=SimpleNamespace(quantization=None),
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )
        return SimpleNamespace(
            engine=engine,
            metadata_group=metadata_group,
            init_metadata=init_metadata,
            handle=handle,
            communicator=communicator,
            make_communicator=make_communicator,
        )

    def test_trainer_disagreement_is_rejected_before_nccl4py(self, worker_init_env):
        group = worker_init_env.metadata_group

        def gather(envelope):
            peer = dict(envelope)
            peer["digest"] = "different"
            return [envelope, peer, envelope]

        group.all_gather_obj.side_effect = gather
        with pytest.raises(RuntimeError, match="source digest disagrees"):
            worker_init_env.engine.init_transfer_engine(self._init_info())

        worker_init_env.make_communicator.assert_not_called()

    def test_worker_rejects_active_reinit_but_allows_init_after_shutdown(
        self, monkeypatch, worker_init_env
    ):
        engine = worker_init_env.engine
        old_handle = Mock()
        old_group = Mock()
        engine._handle = old_handle
        engine._comms = [old_group]

        with pytest.raises(RuntimeError, match="already initialized"):
            engine.init_transfer_engine(self._init_info(params=[]))
        worker_init_env.init_metadata.assert_not_called()

        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        engine.shutdown()

        engine.init_transfer_engine(self._init_info(params=[]))

        assert engine._handle is worker_init_env.handle
        assert engine._comms == [worker_init_env.communicator]

    def test_partial_worker_initialization_releases_transport_resources(
        self, worker_init_env
    ):
        worker_init_env.make_communicator.side_effect = RuntimeError(
            "late init failure"
        )

        with pytest.raises(RuntimeError, match="late init failure"):
            worker_init_env.engine.init_transfer_engine(self._init_info(params=[]))

        worker_init_env.handle.destroy.assert_called_once_with()
        assert worker_init_env.engine._comms == []
        assert worker_init_env.engine._handle is None

    def test_world_must_hold_a_trainer_and_a_worker(self):
        with pytest.raises(ValueError, match="rank_offset"):
            self._init_info(rank_offset=3, world_size=3)

    @pytest.mark.parametrize("dtype_name", ["not_a_dtype", "Tensor"])
    def test_invalid_dtype_name_names_parameter(self, dtype_name):
        engine = object.__new__(M2NWeightTransferEngine)
        param = self._param(dtype_name=dtype_name)

        with pytest.raises(ValueError, match=r"parameter 'w'.*dtype"):
            engine._prepare_source_plan((param,), num_trainer_ranks=1)

    def test_candidates_respect_m2n_limits(self):
        """32 source shards on dim 0: a destination sharded on dim 1, or
        replicated, would feed one destination from all 32 (over 16)."""
        meta = M2NParamMeta(
            "w",
            torch.float32,
            (32, 8),
            M2NLayout(M2NMesh((1, 32), 0), (REPLICATE, 0)),
            owners=tuple(range(32)),
        )
        layouts = candidate_layouts(meta, 4)
        assert layouts and all(0 in placements for _, placements in layouts)
        assert None not in layouts
        # a worker that needs the full tensor then has no layout
        group = Mock()
        group.all_gather_obj.return_value = [{"masks": None, "error": None}] * 32 + [
            {"masks": [0], "error": None}
        ] * 4
        with pytest.raises(RuntimeError, match="no destination layout of 'w'"):
            agree_destination_layouts(group, [meta], 32, None)

    def test_update_preflights_all_names_before_reshard(self, monkeypatch):
        reshard = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.reshard", reshard
        )
        engine = object.__new__(M2NWeightTransferEngine)
        engine._handle = object()
        engine._comms = [object()]
        engine._index = {"valid": 0}
        engine._metas = [
            M2NParamMeta(
                "valid",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                owners=(0,),
            )
        ]

        with pytest.raises(ValueError, match=r"parameter 'unknown'"):
            engine.receive_weights(
                M2NWeightTransferUpdateInfo(names=["valid", "unknown"])
            )

        reshard.assert_not_called()

    def test_receive_follows_the_planned_steps(self, monkeypatch):
        """Direct names land in their target, staged ones are copied out of the
        staging buffer, fallbacks go through load_weights, skipped names are
        not resharded; all in request order."""
        loaded: list[tuple[str, torch.Tensor]] = []
        model = SimpleNamespace(
            load_weights=lambda w: loaded.extend((n, t.clone()) for n, t in w)
        )
        direct, staged = torch.zeros(2), torch.zeros(4)
        engine = object.__new__(M2NWeightTransferEngine)
        engine._m2n = engine._handle = object()
        engine._comms = [object()]
        engine._group_of = [0] * 4
        engine._num_workers = 1
        engine.device = torch.device("cpu")
        engine.model = model
        engine._index = {"direct": 0, "staged": 1, "fallback": 2, "skip": 3}
        engine._metas = [
            M2NParamMeta(
                name,
                torch.float32,
                (2,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                owners=(0,),
            )
            for name in engine._index
        ]
        full = ((0, 2),)
        engine._steps = [
            _Step("direct", None, full, target=direct),
            # the box lands in every other element of `staged`
            _Step("staged", None, full, copies=(((slice(0, 2),), staged[1::2]),)),
            _Step("fallback", None, full),
            _Step("skip"),
        ]
        engine._staging_bytes = 8
        order = []

        def reshard(m2n, handle, comm, stream, meta, layout, n, src, dst):
            order.append(meta.name)
            dst.copy_(torch.tensor([1.0, 2.0]) * len(order))

        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.reshard", reshard
        )
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())

        engine.receive_weights(M2NWeightTransferUpdateInfo(names=list(engine._index)))

        assert order == ["direct", "staged", "fallback"]
        assert direct.tolist() == [1, 2]
        assert staged.tolist() == [0, 2, 0, 4]
        assert [(n, t.tolist()) for n, t in loaded] == [("fallback", [3, 6])]

    def test_trainer_rank_must_be_a_trainer_rank(self):
        """A trainer rank must fall within the trainer portion of the group."""
        with pytest.raises(ValueError, match="num_trainer_ranks"):
            M2NTrainerInitInfo(
                master_address="127.0.0.1",
                master_port=1234,
                world_size=4,
                num_trainer_ranks=2,
                rank=2,
            )

    def test_sender_is_trainer_rank_zero(self):
        """Trainer rank 0 drives the inference control plane."""
        info = M2NTrainerInitInfo(
            master_address="127.0.0.1", master_port=1234, world_size=4, rank=0
        )
        assert info.is_sender


def test_metadata_group_receives_reserved_socket(monkeypatch):
    reservation = Mock()
    create = Mock(return_value=Mock())
    monkeypatch.setattr("vllm.distributed.utils.StatelessProcessGroup.create", create)

    stateless_init_metadata_group(
        "127.0.0.1",
        1234,
        rank=0,
        world_size=2,
        listen_socket=reservation,
    )

    create.assert_called_once_with(
        host="127.0.0.1",
        port=1234,
        rank=0,
        world_size=2,
        listen_socket=reservation,
    )


class TestTrainerSourceContract:
    @staticmethod
    def _meta(name="weight"):
        return ParamMeta(name, torch.float32, (4,))

    @staticmethod
    def _engine(source):
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )
        engine._m2n = _fake_m2n()
        engine._handle = object()
        engine._prepare_source_plan(source, num_trainer_ranks=2)
        engine._layouts = [None] * len(engine._metas)
        engine._skip = [False] * len(engine._metas)
        engine._group_of = [0] * len(engine._metas)
        engine._num_workers = 1
        engine._comms = [Mock(comm=1)]
        return engine

    def test_partial_trainer_initialization_releases_transport_resources(
        self, monkeypatch
    ):
        metadata_group = Mock(rank=0, world_size=2)
        metadata_group.all_gather_obj.side_effect = lambda envelope: [envelope] * 2
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer."
            "stateless_init_metadata_group",
            Mock(return_value=metadata_group),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.import_m2n", Mock()
        )
        handle = Mock()
        nccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.prepare_m2n_local_runtime",
            Mock(return_value=(nccl, 0, handle)),
        )
        communicator = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.create_m2n_communicators",
            Mock(return_value=[communicator]),
        )
        source = _StaticSource([], [])
        client = Mock()
        client.init_weight_transfer_engine.side_effect = RuntimeError(
            "worker init failed"
        )
        engine = M2NTrainerWeightTransferEngine(
            client=client, source=source, is_sender=True
        )
        init_info = M2NTrainerInitInfo(
            rank=0,
            master_address="127.0.0.1",
            master_port=1234,
            world_size=2,
        )

        with pytest.raises(RuntimeError, match="worker init failed"):
            engine._initialize(init_info, source)

        communicator.abort.assert_called_once_with()
        communicator.destroy.assert_not_called()
        handle.destroy.assert_called_once_with()
        assert engine._comms == []
        assert engine._handle is None

    def test_non_sender_observes_rank_zero_finish_failure(self):
        outcomes: dict[int, dict[str, object]] = {}
        sender_group = Mock()
        sender_group.send_obj.side_effect = lambda outcome, rank: outcomes.__setitem__(
            rank, outcome
        )
        sender_group.recv_obj.return_value = {"phase": "round_outcome_consumed"}

        source = _StaticSource([], [])
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = sender_group
        sender._num_trainer_ranks = 2
        sender._send = Mock()
        sender.client = Mock()
        sender.client.finish_weight_update.side_effect = RuntimeError("finish failed")

        with pytest.raises(RuntimeError, match="finish failed"):
            sender.send_weights()

        sender._send.assert_called_once_with()
        sender_group.recv_obj.assert_called_once_with(1)
        receiver_group = Mock()
        receiver_group.recv_obj.side_effect = lambda _source_rank: outcomes.pop(1)
        receiver = self._engine(source)
        receiver._metadata_group = receiver_group
        receiver._num_trainer_ranks = 2
        receiver._send = Mock()

        with pytest.raises(
            RuntimeError,
            match=(
                "worker update failed on trainer rank zero: RuntimeError: finish failed"
            ),
        ):
            receiver.send_weights()

        receiver._send.assert_called_once_with()
        receiver_group.recv_obj.assert_called_once_with(0)
        receiver_group.send_obj.assert_called_once_with(
            {"phase": "round_outcome_consumed"}, 0
        )
        assert not outcomes

    def test_sender_rejects_invalid_round_outcome_acknowledgement(self):
        source = _StaticSource([], [])
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = Mock()
        sender._metadata_group.recv_obj.return_value = {"phase": "wrong"}
        sender._num_trainer_ranks = 2
        sender._send = Mock()
        sender.client = Mock()

        with pytest.raises(RuntimeError, match="invalid round-outcome acknowledgement"):
            sender.send_weights()

    def test_non_sender_consumes_outcome_after_local_failure(self):
        source = _StaticSource([], [])
        receiver = self._engine(source)
        receiver._send = Mock(side_effect=[ValueError("local send failed"), None])
        receiver._metadata_group = Mock()
        receiver._metadata_group.recv_obj.side_effect = [
            {"ok": False, "error": "RuntimeError: first round failed"},
            {"ok": True, "error": None},
        ]

        with pytest.raises(ValueError, match="local send failed"):
            receiver.send_weights()
        receiver.send_weights()

        assert receiver._send.call_count == 2
        assert receiver._metadata_group.recv_obj.call_count == 2
        assert receiver._metadata_group.send_obj.call_args_list == [
            call({"phase": "round_outcome_consumed"}, 0),
            call({"phase": "round_outcome_consumed"}, 0),
        ]

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("missing", "first missing: 'weight'"),
            ("extra", "first extra"),
            ("reordered", "yielded 'other'.*declared 'weight'"),
        ],
    )
    def test_source_cardinality_and_order(self, monkeypatch, case, match):
        shard = _shard()
        values = {
            "missing": [],
            "extra": [("weight", shard), ("extra", shard)],
            "reordered": [("other", shard)],
        }[case]
        source = _StaticSource([self._meta()], values)
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match=match):
            engine._send()

    @pytest.mark.parametrize(
        "shard", [_shard(dtype=torch.float16), _shard(shape=(2,))], ids=str
    )
    def test_shard_must_match_declared_metadata(self, monkeypatch, shard):
        source = _StaticSource([self._meta()], [("weight", shard)])
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(ValueError, match="layout implies"):
            engine._send()

        engine._m2n.reshard.assert_not_called()

    def test_shard_is_resharded_and_recorded_on_the_send_stream(self, monkeypatch):
        shard = _shard()
        source = _StaticSource([self._meta()], [("weight", shard)])
        engine = self._engine(source)
        stream = Mock()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        assert engine._m2n.reshard.call_args.args[0] is shard
        assert shard.recorded == [stream]
        stream.synchronize.assert_called_once_with()

    def test_rank_sends_only_what_it_owns_and_nobody_skips(self, monkeypatch):
        metas = [
            ParamMeta("a", torch.float32, (4,), owners=(0,)),
            ParamMeta("b", torch.float32, (4,), owners=(1,)),
            ParamMeta("c", torch.float32, (4,), owners=(0,)),
        ]
        a, c = _shard(), _shard()
        engine = self._engine(_StaticSource(metas, [("a", a), ("c", c)]))
        engine.rank = 0
        engine._skip = [False, False, True]
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        assert [c.args[0] for c in engine._m2n.reshard.call_args_list] == [a]


class TestRegistration:
    def test_both_registries_expose_the_backend(self):
        """Both worker and trainer factories register nccl_m2n."""
        assert "nccl_m2n" in WeightTransferEngineFactory._registry
        assert "nccl_m2n" in WeightTransferTrainerFactory._registry

    def test_init_info_dispatches_to_the_backend(self):
        """Trainer init info selects the nccl_m2n factory entry."""
        assert M2NTrainerInitInfo.backend == "nccl_m2n"


# ---------------------------------------------------------------------------
# End-to-end transfer
#
# Transport-only, in the style of the NCCL/sparse tests in
# test_weight_transfer.py: two Ray tasks with one GPU each drive the two engines
# directly, so no HTTP server or LLM instance is involved. The worker is not an
# RPC endpoint here, so the trainer engine gets a no-op control-plane client --
# the NCCL rendezvous and the reshard itself are the real thing.
# ---------------------------------------------------------------------------

# Each case: trainer ranks, worker count, and per parameter (name, shape, owners,
# source mesh, source placements). The trainers send `_full(name, shape)`.
_CASES: dict[str, tuple[int, int, list[tuple[Any, ...]]]] = {
    # one trainer, one worker; tcp or a pre-shared uid
    "basic": (1, 1, [("weight", (64, 32), (0,), (1,), None)]),
    # one trainer, two workers holding TP row shards
    "tp2": (1, 2, [("weight", (64, 32), (0,), (1,), None)]),
    # two trainers with different source meshes per parameter
    "mixed": (
        2,
        2,
        [
            ("row_sharded", (4, 4), (0, 1), (1, 2), (Replicate(), Shard(0))),
            ("column_sharded", (4, 4), (0, 1), (2,), (Shard(1),)),
        ],
    ),
    # a pipeline-parallel trainer: each rank owns its own stage's parameters
    "stages": (
        2,
        2,
        [
            ("first", (8, 4), (0,), (1,), None),
            ("second", (8, 4), (1,), (1,), None),
            ("shared", (8, 4), (0, 1), (2,), (Shard(0),)),
        ],
    ),
}


def _full(name: str, shape) -> torch.Tensor:
    base = sum(map(ord, name))
    return torch.arange(base, base + torch.Size(shape).numel(), dtype=torch.float32)


def _init_ray() -> None:
    if ray.is_initialized():
        return
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "VLLM_RELOAD_MODE": "direct",
            }
        },
    )


def _assigned_device() -> "torch.device":
    gpu_ids = ray.get_gpu_ids()
    device = torch.device(f"cuda:{int(gpu_ids[0])}" if gpu_ids else "cuda:0")
    current_platform.set_device(device)
    return device


@ray.remote(num_gpus=1)
def _m2n_trainer_send(
    master_address: str,
    master_port: int,
    case: str,
    trainer_rank: int = 0,
    nccl_unique_id_b64: str | None = None,
) -> bool:
    """Send the case's parameters: this rank's local shard of each it owns."""
    device = _assigned_device()

    from functools import cached_property

    from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

    class NoopClient:
        def init_weight_transfer_engine(self, init_info):
            pass

        def start_weight_update(self):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self, weight_version=None):
            pass

    num_trainers, num_workers, params = _CASES[case]

    class CaseSource(WeightSource):
        @cached_property
        def metadata(self):
            return [
                ParamMeta(
                    name,
                    torch.float32,
                    shape,
                    owners=owners,
                    mesh_shape=mesh,
                    placements=placements,
                )
                for name, shape, owners, mesh, placements in params
            ]

        def iter_local(self):
            for name, shape, owners, _, placements in params:
                if trainer_rank not in owners:
                    continue
                local = _full(name, shape).reshape(shape).to(device)
                for p in placements or ():
                    if isinstance(p, Shard):
                        local = local.chunk(len(owners), p.dim)[
                            owners.index(trainer_rank)
                        ]
                yield name, local.contiguous()

    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=M2NTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            nccl_unique_id_b64=nccl_unique_id_b64,
            world_size=num_trainers + num_workers,
            num_trainer_ranks=num_trainers,
            rank=trainer_rank,
        ),
        client=NoopClient(),
        source=CaseSource(),
    )
    engine.send_weights()
    torch.accelerator.synchronize()
    engine.shutdown()
    return True


@ray.remote(num_gpus=1)
def _m2n_worker_receive(
    master_address: str,
    master_port: int,
    case: str,
    worker_rank: int = 0,
    nccl_unique_id_b64: str | None = None,
) -> dict:
    """Receive the case's parameters into a tiny model that keeps its TP row
    shard (`tp2`) or the full tensor of each, after a cold start, and return
    each parameter's value and how many tensors went through load_weights."""
    from unittest.mock import MagicMock

    device = _assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.m2n_engine import (
        M2NWeightTransferEngine,
        M2NWeightTransferInitInfo,
        M2NWeightTransferUpdateInfo,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    num_trainers, num_workers, params = _CASES[case]
    tp = num_workers if case == "tp2" else 1

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fallbacks = 0
            for name, shape, *_ in params:
                local = (shape[0] // tp, *shape[1:])
                self.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.zeros(local, device=device), requires_grad=False
                    ),
                )

        def load_weights(self, weights):
            for name, tensor in weights:
                self.fallbacks += 1
                getattr(self, name).copy_(tensor.chunk(tp)[worker_rank % tp])

    model_config = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False,
        quantization=None,
        dtype=torch.float32,
    )
    model = Tiny()
    model.load_weights(
        (name, torch.zeros(shape, device=device)) for name, shape, *_ in params
    )
    process_weights_after_loading(model, model_config, device)

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = worker_rank
    parallel_config.world_size = num_workers
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = num_workers
    parallel_config.pipeline_parallel_size = 1
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = model_config

    engine = M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"), vllm_config, device, model
    )
    wire = []
    for name, shape, owners, mesh, placements in params:
        meta = m2n_param_meta(
            ParamMeta(
                name,
                torch.float32,
                shape,
                owners=owners,
                mesh_shape=mesh,
                placements=placements,
            ),
            num_trainers,
        )
        wire.append(
            M2NWireParam(
                name,
                "float32",
                shape,
                meta.source_layout.mesh.dims,
                meta.source_layout.placements,
                meta.owners,
                True,
            )
        )
    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address=master_address,
            master_port=master_port,
            nccl_unique_id_b64=nccl_unique_id_b64,
            rank_offset=num_trainers,
            world_size=num_trainers + num_workers,
            source_digest=source_plan_digest(wire),
            params=[param.to_dict() for param in wire],
        )
    )
    model.fallbacks = 0  # planning traced load_weights
    engine.start_weight_update()
    engine.receive_weights(M2NWeightTransferUpdateInfo(names=[p[0] for p in params]))
    engine.finish_weight_update()
    torch.accelerator.synchronize()
    result = {
        "fallbacks": model.fallbacks,
        "exact": {
            name: torch.equal(
                getattr(model, name).cpu(),
                _full(name, shape).reshape(shape).chunk(tp)[worker_rank % tp],
            )
            for name, shape, *_ in params
        },
    }
    engine.shutdown()
    return result


def _run_case(case: str, nccl_unique_id_b64: str | None = None) -> list[dict]:
    pytest.importorskip("nccl.m2n", reason="nccl_m2n backend needs the m2n runtime")
    num_trainers, num_workers, params = _CASES[case]
    if torch.accelerator.device_count() < num_trainers + num_workers:
        pytest.skip(f"needs {num_trainers + num_workers} GPUs")
    _init_ray()
    address, port = "127.0.0.1", get_open_port()
    workers = [
        _m2n_worker_receive.remote(address, port, case, rank, nccl_unique_id_b64)
        for rank in range(num_workers)
    ]
    trainers = [
        _m2n_trainer_send.remote(address, port, case, rank, nccl_unique_id_b64)
        for rank in range(num_trainers)
    ]
    results = ray.get(trainers + workers, timeout=300)
    assert results[:num_trainers] == [True] * num_trainers
    for result in results[num_trainers:]:
        assert result["fallbacks"] == 0, "a direct name went through load_weights"
        assert all(result["exact"].values()), result
    return results[num_trainers:]


@pytest.mark.parametrize("data_plane", ["tcp", "uid"])
def test_m2n_weight_transfer_between_processes(data_plane):
    """A parameter survives a real reshard from a trainer process to a worker,
    written straight into the model."""
    nccl_unique_id_b64 = None
    if data_plane == "uid":
        nccl = pytest.importorskip("nccl.core")
        nccl_unique_id_b64 = base64.b64encode(
            bytes(nccl.get_unique_id().as_bytes)
        ).decode()
    _run_case("basic", nccl_unique_id_b64)


def test_m2n_weight_transfer_to_tp2_shards():
    """Two inference workers receive only their own TP shard, in place."""
    _run_case("tp2")


def test_m2n_transfer_with_per_parameter_source_meshes():
    """One collective sequence accepts heterogeneous source factorizations."""
    _run_case("mixed")


def test_m2n_transfer_from_partitioned_source_groups():
    """Parameters owned by different trainer ranks travel on their own
    communicators, interleaved with one owned by both."""
    _run_case("stages")
