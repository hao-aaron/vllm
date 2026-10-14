# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-update wall time and peak memory of an in-place weight update.

Streams a checkpoint into a running engine the way NCCL/IPC weight transfer
does (device tensors through the model's own `load_weights`, in batches) and
reports, per update, the wall time and the device memory held above the
engine's baseline. Compare the reload core with the legacy layerwise path:

    python benchmarks/benchmark_weight_reload.py --model Qwen/Qwen3-30B-A3B \\
        --checkpoint Qwen/Qwen3-30B-A3B --path core
    python benchmarks/benchmark_weight_reload.py ... --path legacy
    python benchmarks/benchmark_weight_reload.py ... --path core --modulewise

Run from the repository root (it uses the test harness worker extension).
"""

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", help="checkpoint to stream (default: model)")
    parser.add_argument("--path", choices=["core", "legacy"], default="core")
    parser.add_argument(
        "--modulewise", action="store_true", help="VLLM_RELOAD_DIRECT=0"
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--order", default="natural")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--extra", default="{}", help="extra LLM kwargs (JSON)")
    args = parser.parse_args()

    os.environ["VLLM_RELOAD_LEGACY"] = "1" if args.path == "legacy" else "0"
    os.environ["VLLM_RELOAD_DIRECT"] = "0" if args.modulewise else "1"
    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        worker_extension_cls=(
            "tests.model_executor.model_loader.reload_harness.ReloadHarnessExtension"
        ),
        **json.loads(args.extra),
    )
    rows = []
    for i in range(args.updates):
        stats = llm.collective_rpc(
            "mw_reload",
            kwargs=dict(
                path=args.checkpoint or args.model,
                order=args.order,
                batch_size=args.batch_size,
            ),
        )
        route = llm.collective_rpc("mw_route")[0]
        rows.append(
            {
                "update": i,
                # (without the harness's per-batch gc and memory reads, whose
                # cost grows with the Python heap, not with the update)
                "seconds": max(s["seconds"] - s["measure_seconds"] for s in stats),
                "measure_seconds": max(s["measure_seconds"] for s in stats),
                "peak_gib": max(s["peak_over_base"] for s in stats) / 2**30,
                "held_gib": max(s["max_held_after_batch"] for s in stats) / 2**30,
                "plan": route["last_plan"],
                "legacy": route["legacy"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    steady = rows[1:] or rows
    print(
        json.dumps(
            {
                "model": args.model,
                "path": args.path + ("-modulewise" if args.modulewise else ""),
                "mean_seconds": sum(r["seconds"] for r in steady) / len(steady),
                "max_peak_gib": max(r["peak_gib"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
