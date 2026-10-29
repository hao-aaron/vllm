# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
    RequestOutput,
    SamplingMask,
    WeightVersionSpan,
)
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    RequestSpecDecodeMetrics,
    RequestStateStats,
    SchedulerStats,
)
from vllm.v1.outputs import SamplingMaskLists

if TYPE_CHECKING:
    from vllm.v1.engine.admission_control import SharedAdmissionStats

# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


class RequestOutputCollector:
    """Turns the deltas the output processor produces for one request into
    what its consumer asked for, for hand-off to the generate task.

    DELTA passes deltas through, CUMULATIVE yields the running total, and
    FINAL_ONLY yields once, when every sequence has finished. Unread outputs
    are merged if the producer gets ahead of the consumer.
    """

    def __init__(
        self, output_kind: RequestOutputKind, request_id: str, drain: bool = False
    ):
        self.kind = output_kind
        self.request_id = request_id
        # Created and drained by the OutputProcessor (LLMEngine).
        self.drain = drain
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        self.ready = asyncio.Event()

        # Running total for CUMULATIVE, and for FINAL_ONLY with n > 1.
        self._accum: RequestOutput | None = None

        self._input_stream_task: asyncio.Task | None = None

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if isinstance(output, RequestOutput) and output is not STREAM_FINISHED:
            output = self._collect(output)
            if output is None:
                return
        self._deliver(output)

    def _collect(self, delta: RequestOutput) -> RequestOutput | None:
        if self.kind == RequestOutputKind.DELTA:
            return delta
        if self._accum is None:
            self._accum = delta
        else:
            self._accum.add(delta)
            self._accum.finished = delta.finished
        if self.kind == RequestOutputKind.FINAL_ONLY:
            if not delta.finished:
                return None  # n > 1: wait for every sequence
            self._accum.outputs.sort(key=lambda completion: completion.index)
            return self._accum
        # New objects; the token lists are shared with the accumulator.
        snapshot = copy(self._accum)
        snapshot.outputs = [copy(completion) for completion in self._accum.outputs]
        return snapshot

    def _merge_unread(
        self, unread: RequestOutput, output: RequestOutput
    ) -> RequestOutput:
        """DELTA: concatenate unread deltas. Otherwise the newest replaces."""
        if self.kind != RequestOutputKind.DELTA:
            return output
        unread.add(output)
        return unread

    def _deliver(
        self, output: RequestOutput | PoolingRequestOutput | Exception
    ) -> None:
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            self.output = self._merge_unread(self.output, output)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    def __del__(self):
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


def _is_chunk_boundary(
    output: RequestOutput | PoolingRequestOutput | Exception | None,
) -> bool:
    """Whether the output marks the end of a streaming-input chunk: a
    finish_reason on a request that has not finished."""
    return (
        isinstance(output, RequestOutput)
        and not output.finished
        and any(completion.finished() for completion in output.outputs)
    )


class StreamingInputOutputCollector(RequestOutputCollector):
    """Collector for streaming-input requests. Doesn't merge across input
    chunk boundaries (a finish_reason on a non-finished output)."""

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        super().__init__(output_kind, request_id)
        self._backlog: deque[RequestOutput] = deque()

    def _deliver(
        self, output: RequestOutput | PoolingRequestOutput | Exception
    ) -> None:
        if isinstance(output, Exception):
            self._backlog.clear()
        elif self._backlog or _is_chunk_boundary(self.output):
            assert isinstance(output, RequestOutput)
            if self._backlog and not _is_chunk_boundary(self._backlog[-1]):
                self._backlog[-1] = self._merge_unread(self._backlog[-1], output)
            else:
                self._backlog.append(output)
            return
        super()._deliver(output)

    async def get(self) -> RequestOutput | PoolingRequestOutput:
        output = await super().get()
        if self._backlog:
            self.output = self._backlog.popleft()
            self.ready.set()
        return output

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        output = super().get_nowait()
        if self._backlog:
            self.output = self._backlog.popleft()
            self.ready.set()
        return output


@dataclass
class OutputProcessorOutput:
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]


@dataclass
class StreamingUpdate:
    """Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.
    """

    prompt: str | None
    prompt_token_ids: list[int] | None
    arrival_time: float
    final: bool = False


class RequestState:
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
        remote_prefill_cached_tokens: int | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0
        self.num_cache_creation_tokens = 0
        self.remote_prefill_cached_tokens = remote_prefill_cached_tokens
        # Per-sequence spec-decode accumulator; arrives once (on finish) via
        # EngineCoreOutput, then attached to this sequence's CompletionOutput.
        self.spec_decode_metrics: RequestSpecDecodeMetrics | None = None

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None
        self.trace_headers = trace_headers

        # Routed experts accumulation (prompt + sample chunks)
        self.routed_experts_chunks: list[np.ndarray] = []
        self.sampling_mask_chunks: list[SamplingMaskLists] = []
        # (weight version, first output-token index) for each span.
        self.weight_version_starts: list[tuple[str, int]] = []

        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # Output tokens already sent.

        # Streaming input: resumable is fixed, streaming_input turns False
        # once the final input chunk arrives.
        self.resumable = stream_input
        self.streaming_input = stream_input
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )

    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        # Apply the update to the request state.
        self.streaming_input = not update.final
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        if update.prompt:
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
        if self.prompt_token_ids:
            self.prompt_token_ids.extend(update.prompt_token_ids or ())
        else:
            self.prompt_token_ids = update.prompt_token_ids or []
        assert self.prompt_token_ids is not None
        self.prompt_len = len(self.prompt_token_ids)
        if self.stats is not None:
            self.stats.arrival_time = update.arrival_time
        self.is_prefilling = True

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        remote_prefill_cached_tokens = None
        if sampling_params := request.sampling_params:
            # In a remote prefill scenario, report cached tokens
            # as cache hit rate on the remote P worker.
            if sampling_params.extra_args:
                kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                if kv_transfer_params and kv_transfer_params.get("do_remote_prefill"):
                    cached = kv_transfer_params.get("remote_prefill_cached_tokens")
                    if isinstance(cached, int):
                        remote_prefill_cached_tokens = cached
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            if sampling_params.stream_interval is not None:
                # clamp to the engine-level stream interval.
                stream_interval = max(sampling_params.stream_interval, stream_interval)
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        state = cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            stream_input=request.resumable,
            remote_prefill_cached_tokens=remote_prefill_cached_tokens,
            trace_headers=request.trace_headers,
        )
        if state.stats is not None and sampling_params is not None:
            state.stats.is_watermarked = bool(sampling_params.watermarking)
            state.stats.is_watermark_skipped = sampling_params._watermarking_skipped
        return state

    def make_request_output(
        self,
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        error: RequestError | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        """Everything new since the previous output, if an output is due.

        The output kind only decides when an output is due; the collector
        turns deltas into what the consumer asked for.
        """
        finished = finish_reason is not None

        if pooling_output is not None:
            if not finished and self.output_kind == RequestOutputKind.FINAL_ONLY:
                return None
            return self._new_request_output(
                [self._new_pooling_output(pooling_output)], finished, error=error
            )

        if not self._should_emit(finished):
            return None
        # With n > 1, the request finishes with its last running child.
        parent = self.parent_req
        return self._new_request_output(
            [self._new_completion_output(finish_reason, stop_reason)],
            finished and (parent is None or parent.child_requests <= {self.request_id}),
            kv_transfer_params,
            ec_transfer_params,
        )

    def _should_emit(self, finished: bool) -> bool:
        if finished:
            return True
        if self.output_kind == RequestOutputKind.FINAL_ONLY:
            return False
        if self.stream_interval <= 1:
            return True
        assert self.detokenizer is not None
        # The first output, then every stream_interval tokens.
        unsent = self.detokenizer.num_output_tokens() - self.sent_tokens_offset
        return self.sent_tokens_offset == 0 or unsent >= self.stream_interval

    def _new_request_output(
        self,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        error: RequestError | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=self.external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
                error=error,
            )
        assert self.logprobs_processor is not None
        return RequestOutput(
            request_id=self.external_req_id,
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            # Sent once: the logprobs processor forgets them.
            prompt_logprobs=self.logprobs_processor.pop_prompt_logprobs(),
            prompt_token_id_logprobs=(
                self.logprobs_processor.pop_prompt_token_id_logprobs()
            ),
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            num_cache_creation_tokens=self.num_cache_creation_tokens,
            metrics=self.stats,
        )

    def _new_completion_output(
        self,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        """Per-token fields are sliced by offset since the previous output, so
        an output with no new tokens has empty slices."""
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        start = self.sent_tokens_offset
        self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        logprobs = self.logprobs_processor.logprobs
        if logprobs is not None:
            logprobs = logprobs[start:]

        sampling_mask = None
        if self.sampling_mask_chunks:
            sampling_mask = SamplingMask(
                [
                    position
                    for chunk in self.sampling_mask_chunks
                    for position in chunk.to_nested_list()
                ]
            )
            self.sampling_mask_chunks.clear()

        # Concatenate routed experts on finish
        routed_experts = None
        if finished and self.routed_experts_chunks:
            routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)

        return CompletionOutput(
            index=self.request_index,
            text=self.detokenizer.get_next_output_text(finished),
            token_ids=self.detokenizer.output_token_ids_since(start),
            routed_experts=routed_experts,
            sampling_mask=sampling_mask,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
            spec_decode_metrics=self.spec_decode_metrics if finished else None,
            weight_versions=self._weight_version_spans() if finished else None,
        )

    def track_weight_version(self, weight_version: str) -> None:
        """Start a span when the weight version changes."""
        assert self.detokenizer is not None
        starts = self.weight_version_starts
        if not starts or starts[-1][0] != weight_version:
            starts.append((weight_version, self.detokenizer.num_output_tokens()))

    def _weight_version_spans(self) -> list[WeightVersionSpan] | None:
        assert self.detokenizer is not None
        num_tokens = self.detokenizer.num_output_tokens()
        starts = self.weight_version_starts
        if not starts and num_tokens:
            return None  # No label was reported for these tokens.
        ends = [start for _, start in starts[1:]] + [num_tokens]
        return [WeightVersionSpan(v, s, e) for (v, s), e in zip(starts, ends)]

    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        return PoolingOutput(data=pooling_output)


class RequestRegistry:
    """Owns the request maps, and is the only code that adds or removes
    requests. Every removal goes through unregister(), so finished, aborted
    and rejected requests run the same side effects."""

    def __init__(
        self,
        log_stats: bool,
        tracing_enabled: bool,
        admission_stats: "SharedAdmissionStats | None",
    ):
        self.states: dict[str, RequestState] = {}
        self.parents: dict[str, ParentRequest] = {}
        self.external_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.lora_states = LoRARequestStates(log_stats)
        self.tracing_enabled = tracing_enabled
        self.admission_stats = admission_stats
        # Requests removed outside an engine step (client aborts). Their
        # stats and spans go into the next step's IterationStats.
        self._pending_finished: list[tuple[RequestState, FinishReason]] = []

    def register(self, state: RequestState) -> None:
        assert state.request_id not in self.states
        self.states[state.request_id] = state
        if parent_req := state.parent_req:
            self.parents[parent_req.request_id] = parent_req
        self.external_ids[state.external_req_id].append(state.request_id)
        self._update_admission_stats()

    def sibling_queue(
        self, parent_req: ParentRequest | None
    ) -> RequestOutputCollector | None:
        """The collector shared by the registered children of a parent."""
        if parent_req is not None:
            for child_id in parent_req.child_requests:
                if (state := self.states.get(child_id)) is not None:
                    return state.queue
        return None

    def resolve(self, request_ids: Iterable[str], internal: bool) -> list[RequestState]:
        """Map request IDs to the registered requests they refer to.

        External IDs map to every request registered under them. Internal
        IDs are request IDs or parallel-sampling parent IDs, which map to the
        parent's registered children.
        """
        states: dict[str, RequestState] = {}
        for request_id in request_ids:
            if not internal:
                internal_ids = self.external_ids.get(request_id, ())
            elif parent := self.parents.get(request_id):
                internal_ids = parent.child_requests
            else:
                internal_ids = (request_id,)
            for internal_id in internal_ids:
                if state := self.states.get(internal_id):
                    states[internal_id] = state
        return list(states.values())

    def unregister(
        self,
        state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None = None,
    ) -> None:
        """Remove a request and run every removal side effect.

        finish_reason is None for requests that should leave no
        finished-request record, e.g. ones the engine rejected on arrival.
        """
        request_id = state.request_id
        del self.states[request_id]
        internal_ids = self.external_ids[state.external_req_id]
        internal_ids.remove(request_id)
        if not internal_ids:
            del self.external_ids[state.external_req_id]
        if parent_req := state.parent_req:
            parent_req.child_requests.discard(request_id)
            if not parent_req.child_requests:
                self.parents.pop(parent_req.request_id, None)

        self.lora_states.request_finished(request_id, state.lora_name)
        if finish_reason is not None and state.stats is not None:
            if iteration_stats is None:
                self._pending_finished.append((state, finish_reason))
            else:
                self._record_finished(state, finish_reason, iteration_stats)
        self._update_admission_stats()

    def flush_pending(self, iteration_stats: IterationStats) -> None:
        for state, finish_reason in self._pending_finished:
            self._record_finished(state, finish_reason, iteration_stats)
        self._pending_finished.clear()

    def _update_admission_stats(self) -> None:
        if self.admission_stats is not None:
            self.admission_stats.set_num_requests(len(self.states))

    def _record_finished(
        self,
        state: RequestState,
        finish_reason: FinishReason,
        iteration_stats: IterationStats,
    ) -> None:
        assert state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=state.external_req_id,
            num_prompt_tokens=state.prompt_len,
            max_tokens_param=state.max_tokens_param,
            req_stats=state.stats,
            num_cached_tokens=state.num_cached_tokens,
        )
        ParentRequest.observe_finished_request(
            state.parent_req, iteration_stats, state.stats.num_generation_tokens
        )
        if self.tracing_enabled:
            self._do_tracing(state, iteration_stats)

    @staticmethod
    def _do_tracing(state: RequestState, iteration_stats: IterationStats) -> None:
        assert state.stats is not None
        metrics = state.stats
        arrival_time_ns = int(metrics.arrival_time * 1e9)
        trace_context = extract_trace_context(state.trace_headers)
        prompt_length = length_from_prompt_token_ids_or_embeds(
            state.prompt_token_ids, state.prompt_embeds
        )

        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        queued_time, prefill_time, decode_time, inference_time = metrics.phase_times()

        # Build attributes dict
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            SpanAttributes.GEN_AI_REQUEST_ID: state.external_req_id,
        }

        # Add optional request parameters
        if state.top_p:
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = state.top_p
        if state.max_tokens_param:
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                state.max_tokens_param
            )
        if state.temperature:
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = state.temperature
        if state.n:
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = state.n

        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )


class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
        admission_stats: "SharedAdmissionStats | None" = None,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self._registry = RequestRegistry(log_stats, tracing_enabled, admission_stats)
        # Read-only aliases; add and remove requests through _registry.
        self.request_states = self._registry.states
        self.parent_requests = self._registry.parents
        self.external_req_ids = self._registry.external_ids
        self.lora_states = self._registry.lora_states

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_request(self, request_id: str) -> bool:
        return request_id in self.request_states

    def get_num_queued_tokens(self) -> int:
        """Total prompt tokens of requests currently in the prefill phase.

        Uses ``prompt_len`` rather than remaining prefill work because the
        scheduler's ``num_computed_tokens`` is not propagated to the API
        server until prefill completes.  See ``SchedulerConfig`` docs.
        """
        return sum(
            req.prompt_len for req in self.request_states.values() if req.is_prefilling
        )

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""
        for state in self.request_states.values():
            state.queue.put(e)

    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        request_ids_to_abort = []
        for req_state in self._registry.resolve(request_ids, internal):
            request_ids_to_abort.append(req_state.request_id)
            # Produce final abort output.
            if not req_state.queue.drain and (
                request_output := req_state.make_request_output(
                    # Set pooling_output is not None to
                    # correctly enter the abort pooling branch
                    pooling_output=EMPTY_CPU_TENSOR
                    if req_state.detokenizer is None
                    else None,
                    finish_reason=FinishReason.ABORT,
                    stop_reason=None,
                    kv_transfer_params=None,
                    ec_transfer_params=None,
                )
            ):
                req_state.queue.put(request_output)
            self._registry.unregister(req_state, FinishReason.ABORT)
        return request_ids_to_abort

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            if not req_state.resumable:
                raise ValueError(f"Request id {request_id} already running.")
            self._update_streaming_request_state(req_state, request, prompt)
            return

        if queue is None:
            queue = self._registry.sibling_queue(parent_req) or RequestOutputCollector(
                request.params.output_kind, request_id, drain=True
            )
        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self._registry.register(req_state)

    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """Queue a streaming update instead of immediately applying it."""
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                self._registry.unregister(req_state, None)
                # Unblock the generate() loop.
                req_state.queue.put(STREAM_FINISHED)
            elif req_state.input_chunk_queue:
                req_state.input_chunk_queue[-1].final = True
            else:
                req_state.streaming_input = False
            return

        update = StreamingUpdate(
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            arrival_time=request.arrival_time,
        )

        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:
            req_state.apply_streaming_update(update)
            req_state.input_chunk_queue = deque()
        else:
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
        weight_version: str | None = None,
    ) -> OutputProcessorOutput:
        """Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Put each request's delta into its RequestOutputCollector:
            * AsyncLLM passes the collectors, and the per-request
              generate() tasks read them.

            * Collectors created here (for LLMEngine) are drained into
              the returned list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """
        if iteration_stats is not None:
            self._registry.flush_pending(iteration_stats)

        to_drain: dict[RequestOutputCollector, None] = {}
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            error = engine_core_output.error
            if error is not None and not req_state.queue.drain:
                # Rejected on arrival, e.g. by a paused engine; nothing was run.
                req_state.queue.put(error.to_exception())
                self._registry.unregister(req_state, None)
                if req_state.streaming_input:
                    # Input chunks sent since may re-create the session.
                    reqs_to_abort.append(req_id)
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            ec_transfer_params = engine_core_output.ec_transfer_params
            if engine_core_output.routed_experts is not None:
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                    req_state.num_cache_creation_tokens = (
                        engine_core_output.prefill_stats.num_cache_creation_tokens
                    )
                if req_state.remote_prefill_cached_tokens is not None:
                    req_state.num_cached_tokens = req_state.remote_prefill_cached_tokens
                req_state.is_prefilling = False

            if engine_core_output.spec_decode_metrics is not None:
                req_state.spec_decode_metrics = engine_core_output.spec_decode_metrics

            request_error = None
            if (
                req_state.detokenizer is None
                and engine_core_output.mm_cache_miss_hashes
                and finish_reason == FinishReason.ERROR
            ):
                request_error = RequestError(
                    code="multimodal_cache_miss",
                    message="Multi-modal processor cache miss.",
                    retryable=True,
                )
                pooling_output = EMPTY_CPU_TENSOR

            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                if new_token_ids and weight_version is not None:
                    req_state.track_weight_version(weight_version)
                # 2) Detokenize the token ids into text and perform stop checks.
                num_prev_tokens = req_state.detokenizer.num_output_tokens()
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string
                    self._trim_surplus_tokens(
                        req_state, engine_core_output, num_prev_tokens
                    )

                if engine_core_output.new_sampling_mask is not None:
                    req_state.sampling_mask_chunks.append(
                        engine_core_output.new_sampling_mask
                    )

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            streaming_chunk_boundary = req_state.streaming_input and (
                finish_reason not in (FinishReason.ABORT, FinishReason.ERROR)
            )

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
                error=request_error,
            ):
                if streaming_chunk_boundary:
                    request_output.finished = False

                req_state.queue.put(request_output)
                if req_state.queue.drain:
                    to_drain[req_state.queue] = None

            # Free completed requests.
            if finish_reason is not None:
                if streaming_chunk_boundary:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._registry.unregister(req_state, finish_reason, iteration_stats)
                    if req_state.streaming_input or not engine_core_output.finished:
                        # EngineCore may still have the request: a stop string
                        # was detected here, or an in-flight streaming-input
                        # chunk may re-create the session.
                        reqs_to_abort.append(req_id)

        return OutputProcessorOutput(
            request_outputs=[
                output
                for queue in to_drain
                if (output := queue.get_nowait()) is not None
            ],
            reqs_to_abort=reqs_to_abort,
        )

    @staticmethod
    def _trim_surplus_tokens(
        req_state: RequestState, output: EngineCoreOutput, num_prev_tokens: int
    ) -> None:
        """Drop new tokens generated past a stop string matched by the
        detokenizer, e.g. when multiple tokens are accepted in one step."""
        assert req_state.detokenizer is not None
        num_kept = req_state.detokenizer.num_output_tokens() - num_prev_tokens
        num_dropped = len(output.new_token_ids) - num_kept
        if num_dropped <= 0:
            return
        del output.new_token_ids[num_kept:]
        if output.new_logprobs is not None:
            output.new_logprobs = output.new_logprobs.slice_request(0, num_kept)
        if output.new_sampling_mask is not None:
            output.new_sampling_mask = output.new_sampling_mask.slice_request(
                0, num_kept
            )
        if output.routed_experts is not None:
            req_state.routed_experts_chunks[-1] = output.routed_experts[:-num_dropped]

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        self.lora_states.update_scheduler_stats(scheduler_stats)

    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            req_state.is_prefilling,
            req_state.stats,
            self.lora_states,
            req_state.lora_name,
        )
