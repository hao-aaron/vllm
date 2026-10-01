# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loaders shared by the DeepSeek-V4 family."""

import torch
import torch.nn as nn


def attn_sink_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load this rank's attention sinks into the padded sink parameter.

    The parameter is padded to the kernel's head count; the padded tail keeps
    its create-time -inf. Going through the parameter's loader (instead of a
    direct slice copy in `load_weights`) lets weight reload see the write."""
    num_heads = loaded_weight.shape[0]
    if num_heads > param.shape[0]:
        raise ValueError(
            f"Attention sink has {num_heads} rows, "
            f"but the destination has only {param.shape[0]}"
        )
    param[:num_heads].copy_(loaded_weight)


def make_attn_sink(padded_heads: int, num_local_heads: int) -> nn.Parameter:
    """The padded attention-sink parameter: -inf (no sink effect) beyond the
    `num_local_heads` rows the checkpoint fills."""
    attn_sink = nn.Parameter(
        torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
        requires_grad=False,
    )
    attn_sink.weight_loader = attn_sink_weight_loader
    # Weight reload: only the local heads are loaded, and the padding must stay
    # -inf rather than the zeros reload starts other tensors from
    attn_sink.weight_loader_numel = num_local_heads
    attn_sink.reload_keep_init_value = True
    return attn_sink
