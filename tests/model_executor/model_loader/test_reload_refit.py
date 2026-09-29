# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit-equals-fresh for streaming weight reload (modulewise reload).

An engine that loaded checkpoint A, with CUDA graphs captured, reloads B the
way NCCL/IPC weight transfer does (device tensors through the model's own
`load_weights`). Afterwards every registered tensor and every tensor reachable
from module attributes (quant methods, kernels, quant configs: the
"off-registry" state captured graphs read) must equal a fresh load of B, and
none of them may have moved: a moved tensor is one a captured graph no longer
reads. Checksums, not logits, are compared: logits of separately started
processes can differ (kernel selection), weights cannot.
"""

import pytest

from vllm.platforms import current_platform

from .test_reload import _fp8_reload_unsupported

HARNESS = "tests.model_executor.model_loader.reload_harness.ReloadHarnessExtension"

CASES = [
    pytest.param(
        "Qwen/Qwen3-0.6B",
        "inference-optimization/Qwen3-0.6B-debug-multiply",
        {},
        id="bf16-dense-tied",
    ),
    pytest.param(
        "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
        "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
        {},
        id="fp8-block-dense",
    ),
    pytest.param(
        "inference-optimization/DeepSeek-V3-debug-empty",
        "inference-optimization/DeepSeek-V3-debug-multiply",
        {},
        id="mla-moe",
    ),
    pytest.param(
        "allenai/OLMoE-1B-7B-0924",
        "allenai/OLMoE-1B-7B-0924-Instruct",
        {"quantization": "fp8_per_tensor"},
        id="online-fp8-moe",
        marks=[pytest.mark.slow_test],
    ),
]


def _diff_ptrs(before: dict, after: dict) -> dict[str, list[str]]:
    out = {}
    for kind in ("registered", "off_registry"):
        b, a = before[kind], after[kind]
        out[kind] = sorted(
            [k for k in b if k in a and a[k][0] != b[k][0]]
            + [k for k in a if k not in b]
        )
    return out


def _diff_checksums(got: dict, expected: dict) -> dict[str, list[str]]:
    return {
        kind: sorted(
            k
            for k in set(got[kind]) | set(expected[kind])
            if got[kind].get(k) != expected[kind].get(k)
        )
        for kind in ("registered", "off_registry")
    }


@pytest.mark.parametrize("direct_load", [False, True], ids=["stage", "direct"])
@pytest.mark.parametrize("per_expert", [False, True], ids=["module", "per_expert"])
@pytest.mark.parametrize("model_a,model_b,kwargs", CASES)
def test_refit_equals_fresh(
    vllm_runner, model_a, model_b, kwargs, direct_load, per_expert
):
    if not current_platform.is_cuda():
        pytest.skip("CUDA graphs and device-tensor streaming")
    if per_expert and "quantization" not in kwargs:
        pytest.skip("per-expert completion applies to online-quantized MoE")
    if "FP8" in model_a and _fp8_reload_unsupported():
        pytest.skip("Requires FP8 support")

    common = dict(
        enable_prefix_caching=False,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        worker_extension_cls=HARNESS,
        # graphs on; autotuning off keeps kernel choice stable across engines
        kernel_config={"enable_flashinfer_autotune": False},
        **kwargs,
    )
    with vllm_runner(model_b, **common) as fresh:
        expected = fresh.collective_rpc("mw_checksums")

    with vllm_runner(model_a, **common) as llm:
        before = llm.collective_rpc("mw_ptr_snapshot")
        for _ in range(2):  # the second reload runs against built-once kernels
            stats = llm.collective_rpc(
                "mw_reload",
                kwargs=dict(
                    path=model_b, direct_load=direct_load, per_expert=per_expert
                ),
            )
        after = llm.collective_rpc("mw_ptr_snapshot")
        got = llm.collective_rpc("mw_checksums")

    for rank, (g, e, b, a, st) in enumerate(zip(got, expected, before, after, stats)):
        assert _diff_checksums(g, e) == {"registered": [], "off_registry": []}, (
            f"rank {rank}: tensors differ from a fresh load"
        )
        assert _diff_ptrs(b, a) == {"registered": [], "off_registry": []}, (
            f"rank {rank}: tensors moved (captured graphs read the old ones)"
        )
        assert st["end_over_base"] == 0, f"rank {rank}: reload left memory held"
