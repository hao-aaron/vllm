# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.parallel_sampling import ParentRequest


def test_parallel_sampling_child_requests_preserve_session_id() -> None:
    request = make_request(SamplingParams(n=2))
    request.session_id = "session-1"
    parent_request = ParentRequest(request)

    for idx in range(parent_request.n):
        request_id, child_params = parent_request.get_child_info(idx)
        child_request = request if idx == parent_request.n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = child_params

        assert child_request.session_id == "session-1"


def make_request(sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="parent_id",
        external_req_id="ext_parent_id",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
