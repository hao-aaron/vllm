# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, MutableSequence
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
import torch
from typing_extensions import TypeVar

from vllm.exceptions import GenerationError, RetryableRequestError, VLLMValidationError
from vllm.logger import init_logger
from vllm.logprobs import PromptLogprobs, SampleLogprobs
from vllm.lora.request import LoRARequest
from vllm.v1.metrics.stats import RequestSpecDecodeMetrics, RequestStateStats

logger = init_logger(__name__)


@dataclass(frozen=True)
class RequestError:
    """A request-level error that does not mark the engine as unhealthy."""

    code: str
    message: str
    retryable: bool = False

    def to_exception(self) -> GenerationError | VLLMValidationError:
        if self.code == "validation_error":
            return VLLMValidationError(self.message)
        if self.retryable:
            return RetryableRequestError(self.message)
        return GenerationError(self.message)


@dataclass
class SamplingMask:
    """Per-token sampling support sets aligned with completion token IDs.

    Each inner list contains the vocabulary token IDs that survived
    top-k / top-p / min-p filtering for the corresponding generated token.
    """

    token_ids: list[list[int]]


@dataclass
class WeightVersionSpan:
    """Output tokens `[start, end)` were sampled by weights labelled `version`."""

    version: str
    start: int
    end: int


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        sampling_mask: The post-processing token support set for each generated
            token, if requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
        spec_decode_metrics: Per-sequence speculative-decoding acceptance metrics,
            populated on finish when speculative decoding ran and
            ``--per-request-spec-decode-metrics`` is enabled; None otherwise.
            Surfaced in the response as ``metrics.speculative_decoding`` for
            single-sequence (``n == 1``) requests.
        weight_versions: Contiguous weight version spans over output-token indexes.
            Set on the final output only.

    """

    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    routed_experts: np.ndarray | None = None  # [seq_len,layer_num,topk]
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    lora_request: LoRARequest | None = None
    sampling_mask: SamplingMask | None = None
    spec_decode_metrics: RequestSpecDecodeMetrics | None = None
    weight_versions: list[WeightVersionSpan] | None = None

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"routed_experts={self.routed_experts}, "
            f"sampling_mask={self.sampling_mask}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason})"
        )


@dataclass
class PoolingOutput:
    """The output data of one pooling output of a request.

    Args:
        data: The extracted hidden states.

    """

    data: torch.Tensor

    def __repr__(self) -> str:
        return f"PoolingOutput(data={self.data})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and bool(
            (self.data == other.data).all()
        )


class RequestOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        prompt_token_id_logprobs: Logprobs of prompt_logprob_token_ids, shaped
            [prompt_len - 1 - prompt_logprob_start, num_ids]; row i holds the
            logprobs of its own IDs as predictions of prompt token
            prompt_logprob_start + i + 1, with -inf in padded columns.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
        num_cache_creation_tokens: Prompt tokens currently counted as local
            prefix-cache writes for this request.
        kv_transfer_params: The params for remote K/V transfer.
        ec_transfer_params: The params for remote encoder-cache transfer.

    """

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestStateStats | None = None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        num_cache_creation_tokens: int | None = None,
        *,
        prompt_token_id_logprobs: np.ndarray | None = None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        # Forward compatibility, code that uses args added in new release can
        # still run with older versions of vLLM without breaking.
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning_once(
                "RequestOutput: Ignoring extra arguments: %s", str(kwargs)
            )
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.prompt_token_id_logprobs = prompt_token_id_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens
        self.kv_transfer_params = kv_transfer_params
        self.ec_transfer_params = ec_transfer_params

    def add(self, next_output: "RequestOutput") -> None:
        """Merge the next output of the same request into this one, per
        REQUEST_OUTPUT_MERGE and COMPLETION_OUTPUT_MERGE."""
        _merge_fields(REQUEST_OUTPUT_MERGE, self, next_output)
        for next_completion in next_output.outputs:
            for completion in self.outputs:
                if completion.index == next_completion.index:
                    _merge_fields(COMPLETION_OUTPUT_MERGE, completion, next_completion)
                    break
            else:
                self.outputs.append(next_completion)

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"encoder_prompt={self.encoder_prompt!r}, "
            f"encoder_prompt_token_ids={self.encoder_prompt_token_ids}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"prompt_token_id_logprobs={self.prompt_token_id_logprobs}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished}, "
            f"metrics={self.metrics}, "
            f"lora_request={self.lora_request}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"num_cache_creation_tokens={self.num_cache_creation_tokens})"
        )


def _append(current: Any, new: Any) -> Any:
    """Per-token data: concatenate."""
    if new is None:
        return current
    if current is None:
        return new
    if isinstance(current, str):
        return current + new
    if isinstance(current, SamplingMask):
        current.token_ids.extend(new.token_ids)
        return current
    if not isinstance(current, MutableSequence):
        current = list(current)
    current.extend(new)
    return current


def _latest(current: Any, new: Any) -> Any:
    return new


def _on_finish(current: Any, new: Any) -> Any:
    """Sent once, on the final output."""
    return current if new is None else new


def _once(current: Any, new: Any) -> Any:
    """Sent once, on the first output."""
    return new if current is None else current


def _any(current: Any, new: Any) -> Any:
    return current or new


MergeFn = Callable[[Any, Any], Any]

# How RequestOutput.add() combines each field; None means the field is the
# same for every output of a request and is kept. Every CompletionOutput field
# and RequestOutput constructor argument must have an entry;
# tests/v1/engine/test_output_semantics.py checks this.
COMPLETION_OUTPUT_MERGE: dict[str, MergeFn | None] = {
    "index": None,
    "text": _append,
    "token_ids": _append,
    "cumulative_logprob": _latest,
    "logprobs": _append,
    "routed_experts": _on_finish,
    "finish_reason": _latest,
    "stop_reason": _latest,
    "lora_request": None,
    "sampling_mask": _append,
    "spec_decode_metrics": _on_finish,
    "weight_versions": _on_finish,
}

REQUEST_OUTPUT_MERGE: dict[str, MergeFn | None] = {
    "request_id": None,
    "prompt": None,
    "prompt_token_ids": None,
    "prompt_logprobs": _once,
    # Matched by index in RequestOutput.add().
    "outputs": None,
    "finished": _any,
    "metrics": None,
    "lora_request": None,
    "encoder_prompt": None,
    "encoder_prompt_token_ids": None,
    "num_cached_tokens": _once,
    "num_cache_creation_tokens": _once,
    "prompt_token_id_logprobs": _once,
    "kv_transfer_params": _latest,
    "ec_transfer_params": _latest,
}


def _merge_fields(rules: dict[str, MergeFn | None], current: Any, new: Any) -> None:
    for name, merge in rules.items():
        if merge is not None:
            setattr(current, name, merge(getattr(current, name), getattr(new, name)))


# Sentinel to indicate request is finished, used with streaming inputs.
STREAM_FINISHED = RequestOutput(
    request_id="",
    prompt=None,
    prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[],
    finished=True,
)

_O = TypeVar("_O", default=PoolingOutput)


@dataclass(frozen=True)
class LateChunk:
    token_range: tuple[int, int]
    char_range: tuple[int, int] | None


@dataclass(frozen=True)
class LateChunkingMetadata:
    chunk_size: int
    input_tokens: int
    chunks: list[LateChunk]
    strategy: str = "fixed_tokens"


class PoolingRequestOutput(Generic[_O]):
    """The output data of a pooling request to the LLM.

    Args:
        request_id (str): A unique identifier for the pooling request.
        outputs (PoolingOutput): The pooling results for the given input.
        prompt_token_ids (list[int]): A list of token IDs used in the prompt.
        num_cached_tokens: The number of tokens with prefix cache hit.
        finished (bool): A flag indicating whether the pooling is completed.
        error: Structured request-level error information, if the request failed.
        late_chunking: Token and source ranges for fixed-length late chunking.

    """

    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
        *,
        error: RequestError | None = None,
        late_chunking: LateChunkingMetadata | None = None,
    ):
        self.request_id = request_id
        self.late_chunking = late_chunking
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.finished = finished
        self.outputs = outputs
        self.error = error

    def __repr__(self) -> str:
        error = f", error={self.error!r}" if self.error is not None else ""
        return (
            f"{type(self).__name__}(request_id={self.request_id!r}, "
            f"outputs={self.outputs!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"finished={self.finished}"
            f"{error})"
        )


@dataclass
class EmbeddingOutput:
    """The output data of one embedding output of a request.

    Args:
        embedding: The embedding vector, which is a list of floats.
            Its length depends on the hidden dimension of the model.

    """

    embedding: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> "EmbeddingOutput":
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D embedding vector")

        return EmbeddingOutput(pooled_data.tolist())

    @property
    def hidden_size(self) -> int:
        return len(self.embedding)

    def __repr__(self) -> str:
        return f"EmbeddingOutput(hidden_size={self.hidden_size})"


class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(
        request_output: PoolingRequestOutput,
    ) -> "EmbeddingRequestOutput":
        outputs = (
            EmbeddingOutput([])
            if request_output.error is not None
            else EmbeddingOutput.from_base(request_output.outputs)
        )
        return EmbeddingRequestOutput(
            request_id=request_output.request_id,
            outputs=outputs,
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
            error=request_output.error,
        )


@dataclass
class ClassificationOutput:
    """The output data of one classification output of a request.

    Args:
        probs: The probability vector, which is a list of floats.
            Its length depends on the number of classes.

    """

    probs: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> "ClassificationOutput":
        # pooling_output shape: (num_classes)
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D probability vector")

        return ClassificationOutput(pooled_data.tolist())

    @property
    def num_classes(self) -> int:
        return len(self.probs)

    def __repr__(self) -> str:
        return f"ClassificationOutput(num_classes={self.num_classes})"


class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(
        request_output: PoolingRequestOutput,
    ) -> "ClassificationRequestOutput":
        outputs = (
            ClassificationOutput([])
            if request_output.error is not None
            else ClassificationOutput.from_base(request_output.outputs)
        )
        return ClassificationRequestOutput(
            request_id=request_output.request_id,
            outputs=outputs,
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
            error=request_output.error,
        )


@dataclass
class ScoringOutput:
    """The output data of one scoring output of a request.

    Args:
        score: The similarity score, which is a scalar value.

    """

    score: float

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> "ScoringOutput":
        # pooling_output shape:
        #   classify task: (num_classes) num_classes == 1
        #   embed task: a scalar value
        pooled_data = pooling_output.data.squeeze()
        if pooled_data.ndim != 0:
            raise ValueError("pooled_data should be a scalar score")

        return ScoringOutput(pooled_data.item())

    def __repr__(self) -> str:
        return f"ScoringOutput(score={self.score})"


class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(
        request_output: PoolingRequestOutput,
    ) -> "ScoringRequestOutput":
        outputs = (
            ScoringOutput(0.0)
            if request_output.error is not None
            else ScoringOutput.from_base(request_output.outputs)
        )
        return ScoringRequestOutput(
            request_id=request_output.request_id,
            outputs=outputs,
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
            error=request_output.error,
        )
