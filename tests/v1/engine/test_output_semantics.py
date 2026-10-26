# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a client assembles from OutputProcessor, across output kinds,
stream intervals, reader speeds and n, must match the FINAL_ONLY result."""

import itertools
from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np
import pytest
from transformers import AutoTokenizer

from tests.v1.engine.utils import (
    FULL_STRINGS,
    NUM_PROMPT_LOGPROBS_UNDER_TEST,
    NUM_SAMPLE_LOGPROBS_UNDER_TEST,
    PROMPT_LEN,
    STOP_STRINGS,
    TOKENIZER_NAME,
    generate_dummy_prompt_logprobs_tensors,
    generate_dummy_sample_logprobs,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tracing import SpanAttributes
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.outputs import LogprobsLists, SamplingMaskLists

DELTA = RequestOutputKind.DELTA
CUMULATIVE = RequestOutputKind.CUMULATIVE
FINAL_ONLY = RequestOutputKind.FINAL_ONLY

ABORT_AFTER_STEP = 4


@dataclass
class Vectors:
    tokenizer: object
    prompt_tokens: list[list[int]]
    generation_tokens: list[list[int]]
    generation_logprobs: list
    prompt_logprobs: list


@pytest.fixture(scope="module")
def vectors() -> Vectors:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    full = [tokenizer(text).input_ids for text in FULL_STRINGS[:2]]
    prompt_tokens = [ids[:PROMPT_LEN] for ids in full]
    generation_tokens = [ids[PROMPT_LEN:] for ids in full]
    return Vectors(
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        generation_logprobs=[
            generate_dummy_sample_logprobs(
                sampled_tokens_list=tokens,
                num_logprobs=NUM_SAMPLE_LOGPROBS_UNDER_TEST,
                tokenizer=tokenizer,
            )
            for tokens in generation_tokens
        ],
        prompt_logprobs=[
            generate_dummy_prompt_logprobs_tensors(
                prompt_tokens_list=tokens,
                num_logprobs=NUM_PROMPT_LOGPROBS_UNDER_TEST,
                tokenizer=tokenizer,
            )
            for tokens in prompt_tokens
        ],
    )


SLOW_READER_PERIOD = 3


@dataclass
class Assembled:
    """Per-index view a client reconstructs from what generate() yielded."""

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[dict[int, float]] = field(default_factory=list)
    sampling_mask: list[list[int]] = field(default_factory=list)
    cumulative_logprob: float | None = None
    routed_experts: np.ndarray | None = None
    spec_decode_metrics: object = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def set_from(self, c: CompletionOutput) -> None:
        self.text = c.text
        self.token_ids = list(c.token_ids)
        self.logprobs = _logprob_values(c)
        self.sampling_mask = list(c.sampling_mask.token_ids) if c.sampling_mask else []
        self._set_latest(c)

    def append(self, c: CompletionOutput) -> None:
        self.text += c.text
        self.token_ids.extend(c.token_ids)
        self.logprobs.extend(_logprob_values(c))
        if c.sampling_mask is not None:
            self.sampling_mask.extend(c.sampling_mask.token_ids)
        self._set_latest(c)

    def _set_latest(self, c: CompletionOutput) -> None:
        self.cumulative_logprob = c.cumulative_logprob
        if c.routed_experts is not None:
            self.routed_experts = c.routed_experts
        if c.spec_decode_metrics is not None:
            self.spec_decode_metrics = c.spec_decode_metrics
        self.finish_reason = c.finish_reason
        self.stop_reason = c.stop_reason

    def assert_equal(self, other: "Assembled") -> None:
        assert self.text == other.text
        assert self.token_ids == other.token_ids
        assert self.logprobs == other.logprobs
        assert self.sampling_mask == other.sampling_mask
        assert self.cumulative_logprob == pytest.approx(other.cumulative_logprob)
        np.testing.assert_array_equal(self.routed_experts, other.routed_experts)
        assert self.spec_decode_metrics == other.spec_decode_metrics
        assert self.finish_reason == other.finish_reason
        assert self.stop_reason == other.stop_reason


def _logprob_values(c: CompletionOutput) -> list[dict[int, float]]:
    if not c.logprobs:
        return []
    return [
        {tok: lp.logprob for tok, lp in c.logprobs[i].items()}
        for i in range(len(c.logprobs))
    ]


@dataclass
class Run:
    completions: dict[int, Assembled]
    prompt_logprobs_len: int | None
    kv_transfer_params: dict | None
    num_yields: int
    finished_flags: list[bool]


def _drive(
    vectors,
    kind: RequestOutputKind,
    stream_interval: int,
    reader_period: int,
    n: int,
    finish_by: str,
) -> Run:
    """Feed synthetic EngineCoreOutputs for one request through
    OutputProcessor and assemble what a client of generate() would see."""
    processor = OutputProcessor(
        vectors.tokenizer, log_stats=False, stream_interval=stream_interval
    )
    params = SamplingParams(
        n=n,
        output_kind=kind,
        logprobs=NUM_SAMPLE_LOGPROBS_UNDER_TEST,
        prompt_logprobs=NUM_PROMPT_LOGPROBS_UNDER_TEST,
        stop=STOP_STRINGS[:2] if finish_by == "stop_string" else None,
    )
    request = EngineCoreRequest(
        request_id="req-int",
        external_req_id="req",
        prompt_token_ids=vectors.prompt_tokens[0],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    queue = RequestOutputCollector(kind, request.request_id)
    if n == 1:
        child_ids = [request.request_id]
        processor.add_request(request, None, None, 0, queue)
    else:
        parent = ParentRequest(request)
        child_ids = []
        for idx in range(n):
            child_id, child_params = parent.get_child_info(idx)
            child = EngineCoreRequest(
                request_id=child_id,
                external_req_id=request.external_req_id,
                prompt_token_ids=request.prompt_token_ids,
                mm_features=None,
                sampling_params=child_params,
                pooling_params=None,
                arrival_time=0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
            processor.add_request(child, None, parent, idx, queue)
            child_ids.append(child_id)

    # Sequence i samples generation_tokens[i]; they differ in length, so with
    # n=2 the children finish at different steps.
    spec_metrics = [f"spec-{i}" for i in range(n)]
    received: list[RequestOutput] = []

    def read() -> None:
        if (out := queue.get_nowait()) is not None:
            received.append(out)

    for step in itertools.count():
        if finish_by == "abort" and step == ABORT_AFTER_STEP:
            processor.abort_requests([request.external_req_id], internal=False)
            read()
            break
        outputs = []
        for idx, child_id in enumerate(child_ids):
            tokens = vectors.generation_tokens[idx]
            if not processor.has_request(child_id) or step >= len(tokens):
                continue
            last = step == len(tokens) - 1
            lp_ids, lp_vals, rank = vectors.generation_logprobs[idx][step]
            outputs.append(
                EngineCoreOutput(
                    request_id=child_id,
                    new_token_ids=[tokens[step]],
                    new_logprobs=LogprobsLists(
                        np.array([lp_ids]), np.array([lp_vals]), np.array([rank])
                    ),
                    new_prompt_logprobs_tensors=(
                        vectors.prompt_logprobs[0] if step == 0 else None
                    ),
                    new_sampling_mask=SamplingMaskLists(
                        np.array([tokens[step], step + 1000 * idx])
                    ),
                    routed_experts=np.full((1, 2, 2), step + 100 * idx, np.int32),
                    finish_reason=FinishReason.LENGTH if last else None,
                    spec_decode_metrics=spec_metrics[idx] if last else None,
                    kv_transfer_params={"child": idx} if last else None,
                )
            )
        if not outputs:
            break
        processor.process_outputs(outputs)
        if (step + 1) % reader_period == 0:
            read()
    read()

    completions: dict[int, Assembled] = {}
    for out in received:
        for c in out.outputs:
            assembled = completions.setdefault(c.index, Assembled())
            if kind == DELTA:
                assembled.append(c)
            else:
                assembled.set_from(c)
    first_prompt_logprobs = next(
        (o.prompt_logprobs for o in received if o.prompt_logprobs), None
    )
    return Run(
        completions=completions,
        prompt_logprobs_len=(
            None if first_prompt_logprobs is None else len(first_prompt_logprobs)
        ),
        kv_transfer_params=received[-1].kv_transfer_params if received else None,
        num_yields=len(received),
        finished_flags=[o.finished for o in received],
    )


def _expected_fast_reader_yields(
    vectors, kind: RequestOutputKind, stream_interval: int, n: int, finish_by: str
) -> int | None:
    """Number of generate() wakeups for a reader that keeps up."""
    if kind == FINAL_ONLY:
        return 1
    if finish_by == "stop_string":
        return None  # depends on where the stop string lands
    lengths = [len(vectors.generation_tokens[i]) for i in range(n)]
    num_steps = max(lengths)
    if finish_by == "abort":
        num_steps = ABORT_AFTER_STEP
    emit_steps = set()
    for length in lengths:
        sent = 0
        for step in range(min(length, num_steps)):
            num_tokens = step + 1
            if (
                num_tokens == length
                or sent == 0
                or num_tokens - sent >= stream_interval
            ):
                emit_steps.add(step)
                sent = num_tokens
    yields = len(emit_steps)
    if finish_by == "abort":
        yields += 1  # the final ABORT output
    return yields


MATRIX = list(
    itertools.product(
        list(RequestOutputKind),
        [1, 2, 3],  # stream_interval
        [1, SLOW_READER_PERIOD],  # reader period
        [1, 2],  # n
        ["length", "stop_string", "abort"],
    )
)


@pytest.mark.parametrize("kind,interval,reader,n,finish_by", MATRIX)
def test_client_view_matches_final_only(vectors, kind, interval, reader, n, finish_by):
    reference = _drive(vectors, FINAL_ONLY, 1, 1, n, finish_by)
    run = _drive(vectors, kind, interval, reader, n, finish_by)

    assert sorted(run.completions) == list(range(n))
    for idx in range(n):
        run.completions[idx].assert_equal(reference.completions[idx])
    assert run.prompt_logprobs_len == reference.prompt_logprobs_len
    assert run.finished_flags[-1]
    assert not any(run.finished_flags[:-1])
    if finish_by != "abort" and n == 1:
        assert run.kv_transfer_params == reference.kv_transfer_params


@pytest.mark.parametrize(
    "kind,interval,n,finish_by",
    [
        pytest.param(
            *p,
            marks=pytest.mark.xfail(
                p[0] == CUMULATIVE and p[1] > 1,
                reason="CUMULATIVE ignores stream_interval",
                strict=True,
            ),
        )
        for p in itertools.product(
            list(RequestOutputKind), [1, 2, 3], [1, 2], ["length", "abort"]
        )
    ],
)
def test_fast_reader_wakeups(vectors, kind, interval, n, finish_by):
    expected = _expected_fast_reader_yields(vectors, kind, interval, n, finish_by)
    run = _drive(vectors, kind, interval, 1, n, finish_by)
    assert run.num_yields == expected


def _make_request(
    request_id: str = "req-int",
    external_req_id: str = "req",
    output_kind: RequestOutputKind = DELTA,
    **kwargs,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=external_req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(output_kind=output_kind, max_tokens=8),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        **kwargs,
    )


@pytest.mark.xfail(reason="B1: aborts skip finished-request stats", strict=True)
def test_abort_records_finished_request_stats():
    """B1: a client abort produces a finished-request record."""
    processor = OutputProcessor(None, log_stats=True)
    for i in range(2):
        processor.add_request(
            _make_request(f"req-{i}-int", f"req-{i}"),
            None,
            queue=RequestOutputCollector(DELTA, f"req-{i}-int"),
        )
    processor.process_outputs(
        [EngineCoreOutput(f"req-{i}-int", [7]) for i in range(2)],
        engine_core_timestamp=0.0,
        iteration_stats=IterationStats(),
    )

    processor.abort_requests(["req-0"], internal=False)

    iteration_stats = IterationStats()
    processor.process_outputs(
        [EngineCoreOutput("req-1-int", [8])],
        engine_core_timestamp=0.0,
        iteration_stats=iteration_stats,
    )
    finished = {r.request_id: r for r in iteration_stats.finished_requests}
    assert finished["req-0"].finish_reason == FinishReason.ABORT
    assert finished["req-0"].num_generation_tokens == 1


@pytest.mark.xfail(reason="B1: aborts skip tracing", strict=True)
def test_abort_emits_tracing_span():
    """B1: a client abort ends the request's tracing span."""
    processor = OutputProcessor(None, log_stats=True, tracing_enabled=True)
    processor.add_request(
        _make_request(trace_headers={"traceparent": "x"}),
        None,
        queue=RequestOutputCollector(DELTA, "req-int"),
    )
    processor.add_request(
        _make_request("other-int", "other"),
        None,
        queue=RequestOutputCollector(DELTA, "other-int"),
    )
    with patch("vllm.v1.engine.output_processor.instrument_manual") as span:
        processor.abort_requests(["req"], internal=False)
        processor.process_outputs(
            [EngineCoreOutput("other-int", [8])],
            engine_core_timestamp=0.0,
            iteration_stats=IterationStats(),
        )
    assert span.call_count == 1
    attributes = span.call_args.kwargs["attributes"]
    assert attributes[SpanAttributes.GEN_AI_REQUEST_ID] == "req"


def test_duplicate_request_id_is_rejected():
    """B2: with request ID randomization disabled, a second request reusing a
    running ID is rejected and the original completes normally."""
    processor = OutputProcessor(None, log_stats=False)
    queue = RequestOutputCollector(FINAL_ONLY, "req")
    processor.add_request(_make_request("req", "req", FINAL_ONLY), None, queue=queue)

    with pytest.raises(ValueError, match="already running"):
        processor.add_request(
            _make_request("req", "req", FINAL_ONLY),
            None,
            queue=RequestOutputCollector(FINAL_ONLY, "req"),
        )

    processor.process_outputs([EngineCoreOutput("req", [5])])
    processor.process_outputs(
        [EngineCoreOutput("req", [6], finish_reason=FinishReason.LENGTH)]
    )
    output = queue.get_nowait()
    assert output is not None and output.finished
    assert list(output.outputs[0].token_ids) == [5, 6]


def _assert_maps_consistent(processor: OutputProcessor) -> None:
    states = dict(processor.request_states)
    mapped = [i for ids in processor.external_req_ids.values() for i in ids]
    assert sorted(mapped) == sorted(states)
    for state in states.values():
        assert state.request_id in processor.external_req_ids[state.external_req_id]
        if state.parent_req is not None:
            assert state.parent_req.request_id in processor.parent_requests
    for parent_id, parent in processor.parent_requests.items():
        assert any(s.parent_req is parent for s in states.values()), parent_id


@pytest.mark.xfail(reason="aborting children without a queue leaks the parent")
@pytest.mark.parametrize("seed", range(5))
def test_request_maps_stay_consistent(seed):
    """Request maps agree after any sequence of add, finish and abort."""
    rng = np.random.default_rng(seed)
    processor = OutputProcessor(None, log_stats=False)
    live: list[str] = []
    for i in range(40):
        action = rng.choice(["add", "add_n", "finish", "abort_ext", "abort_int"])
        if action == "add":
            rid = f"r{i}"
            processor.add_request(_make_request(f"{rid}-int", rid), None)
            live.append(f"{rid}-int")
        elif action == "add_n":
            parent_req = _make_request(f"p{i}-int", f"p{i}")
            parent_req.sampling_params.n = 2
            parent = ParentRequest(parent_req)
            for idx in range(2):
                child_id, child_params = parent.get_child_info(idx)
                child = _make_request(child_id, f"p{i}")
                child.sampling_params = child_params
                processor.add_request(child, None, parent, idx)
                live.append(child_id)
        elif live:
            target = live.pop(int(rng.integers(len(live))))
            if action == "finish":
                processor.process_outputs(
                    [EngineCoreOutput(target, [1], finish_reason=FinishReason.STOP)]
                )
            elif action == "abort_int":
                processor.abort_requests([target], internal=True)
            else:
                state = processor.request_states[target]
                processor.abort_requests([state.external_req_id], internal=False)
                live = [r for r in live if processor.has_request(r)]
        _assert_maps_consistent(processor)
