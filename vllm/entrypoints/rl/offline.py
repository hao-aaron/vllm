# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.entrypoints.common.offline import OfflineInferenceMixin


class RLOfflineMixin(OfflineInferenceMixin):
    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info

        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def shutdown_weight_transfer_engine(self) -> None:
        """Release the trainer connection without shutting down inference."""
        self.llm_engine.collective_rpc("shutdown_weight_transfer_engine")

    def check_reload_support(self) -> dict:
        """Whether this engine's weights can be updated in place correctly.

        Runs the reload verifier on every rank (report only: it never refuses
        an update). Returns `{"status": SAFE|UNVERIFIED|UNSAFE, "items": [...]}`,
        the worst status over ranks and every rank's non-SAFE items (with
        their rank). See "Writing a reloadable model" in the contributing docs.
        """
        from vllm.model_executor.model_loader.reload import merge_reload_reports

        return merge_reload_reports(
            self.llm_engine.collective_rpc("check_reload_support")
        )

    def start_weight_update(
        self, partial: bool | None = None, modules: list[str] | None = None
    ) -> None:
        """Start a new weight update.

        Generation is paused until the update finishes. If it fails after
        starting, the engine refuses to serve until an update succeeds that
        re-sends every module the failed one wrote.

        Args:
            partial: True if the update sends only part of the model; False
                to require every module the cold load wrote (finish raises
                otherwise); None (default) to only log modules it missed.
            modules: the module names (prefixes) a partial update sends; every
                rank checks it declares the same set.

        """
        kwargs: dict = {} if partial is None else {"partial": partial}
        if modules is not None:
            kwargs["modules"] = list(modules)
        self.llm_engine.collective_rpc("start_weight_update", kwargs=kwargs)

    def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        self.llm_engine.collective_rpc("start_draft_weight_update")

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info

        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        self.llm_engine.collective_rpc("finish_weight_update")
        if weight_version is not None:
            self.llm_engine.set_weight_version(weight_version)

    def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        self.llm_engine.set_weight_version(new_version)

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self.llm_engine.get_weight_version()
