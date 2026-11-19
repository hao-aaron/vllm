# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side engine for the ModelExpress backend.

ModelExpress publishes each WeightVersion itself (`ModelExpressTrainerClient`),
so this engine only drives the inference-side install of a published version.
It needs no ModelExpress import.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)


@dataclass
class ModelExpressTrainerInitInfo(TrainerInitInfo):
    """Trainer init info; `worker_init_info` is forwarded verbatim to the workers'
    `ModelExpressWeightTransferInitInfo`."""

    backend: ClassVar[str] = "modelexpress"

    worker_init_info: dict[str, Any] = field(default_factory=dict)


class ModelExpressTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[ModelExpressTrainerInitInfo]
):
    """Install a published ModelExpress WeightVersion on every inference engine."""

    init_info_cls = ModelExpressTrainerInitInfo

    @classmethod
    def trainer_init(
        cls,
        init_info: ModelExpressTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is not None:
            raise ValueError(
                "ModelExpress takes no WeightSource; publish versions with "
                "ModelExpressTrainerClient and pass the version_id to send_weights."
            )
        engine = cls(client=client, is_sender=init_info.is_sender, rank=init_info.rank)
        engine._apply_pause_config(init_info)
        if engine.is_sender:
            client.init_weight_transfer_engine(dict(init_info.worker_init_info))
        return engine

    def send_weights(self, version_id: str) -> None:
        """Install the READY WeightVersion `version_id`, pausing only for it."""
        if not self.is_sender:
            return
        with self._pause_for_update():
            self.client.start_weight_update()
            self.client.update_weights({"version_id": version_id})
            self.client.finish_weight_update(weight_version=version_id)
