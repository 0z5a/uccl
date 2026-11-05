#!/usr/bin/env python3
"""Collect a no-P2P diagnostic; never label different hardware a paper reproduction."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PAPER = "https://arxiv.org/pdf/2609.40093"


def command(args, env=None):
    result = subprocess.run(args, text=True, capture_output=True, env=env, check=False)
    return {
        "argv": args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def summarize(
    samples, operation, sizes, repeats, iterations, ranks, router_backend="cpu-router"
):
    """Require every checked iteration; summarize means, not best observations."""
    expected = {
        (size, backend, repeat, iteration)
        for size in sizes
        for backend in ("nccl", router_backend)
        for repeat in range(repeats)
        for iteration in range(iterations)
    }
    values = {}
    canonical = {"allgather": "all_gather", "reducescatter": "reduce_scatter"}[
        operation
    ]
    for row in samples:
        if row["collective"] != canonical or row["in_place"] != "0":
            raise ValueError("unexpected operation or placement")
        key = (
            int(row["bytes"]),
            row["backend"],
            int(row["repeat"]),
            int(row["iteration"]),
        )
        value = float(row["slowest_rank_us"])
        if (
            key not in expected
            or key in values
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("unexpected, duplicate, or invalid timing record")
        values[key] = value
    if set(values) != expected:
        raise ValueError("incomplete run: missing required timing records")
    result = []
    for size in sizes:
        backend_stats = {}
        for backend in ("nccl", router_backend):
            run_means = [
                statistics.mean(values[size, backend, r, i] for i in range(iterations))
                for r in range(repeats)
            ]
            backend_stats[backend] = (
                statistics.mean(run_means),
                min(run_means),
                max(run_means),
            )
        for backend, (mean, low, high) in backend_stats.items():
            result.append(
                {
                    "collective": operation,
                    "full_array_bytes": size,
                    "ranks": ranks,
                    "dtype": "bf16",
                    "backend": backend,
                    "runs": repeats,
                    "iterations_per_run": iterations,
                    "mean_us": mean,
                    "min_run_mean_us": low,
                    "max_run_mean_us": high,
                    "algorithm_GBs": size / mean / 1000,
                    "nccl_bus_GBs": size / mean / 1000 * (ranks - 1) / ranks,
                    "speedup_vs_local_nccl": backend_stats["nccl"][0] / mean,
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--begin", type=int, default=1024)
    parser.add_argument("--end", type=int, default=1 << 30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--required-nccl", type=int, default=23007)
    parser.add_argument("--packet-bytes", type=int, default=512 << 10)
    parser.add_argument(
        "--gpu-reduce",
        action="store_true",
        help="Opt-in destination-GPU reduction ablation",
    )
    parser.add_argument(
        "--interleave-scatter",
        action="store_true",
        help="GPU reduction with all-destination packet stripes",
    )
    args = parser.parse_args()
    if (
        args.gpus < 2
        or args.gpus > 32
        or args.begin < 128
        or args.begin % (2 * args.gpus)
        or args.end < args.begin
        or args.end > 1 << 30
        or args.iterations < 1
        or args.warmup < 1
        or args.packet_bytes < 128 << 10
        or args.packet_bytes > 32 << 20
        or args.packet_bytes % 16
    ):
        parser.error(
            "invalid size/count; full-array bytes must divide into BF16 shards"
        )
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
    overrides = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_DEBUG": "INFO",
    }
    env.update(overrides)
    router_backend = (
        "cpu-router-gpu-striped"
        if args.interleave_scatter
        else "cpu-router-gpu-reduce" if args.gpu_reduce else "cpu-router"
    )
    sizes = []
    size = args.begin
    while size <= args.end:
        sizes.append(size)
        size *= 2
    manifest = {
        "paper_url": PAPER,
        "paper_pdf_sha256": "977b985bcb74cfb1c886e68805aad93682f44ffb03010e8968099b32bc8d7941",
        "paper_setup_reproduced": False,
        "comparison_scope": "local eager BF16 primitives, not ThunderEP or end-to-end",
        "router_backend": router_backend,
        "router_packet_bytes": args.packet_bytes,
        "measurement_scope": "slowest-rank CUDA events; includes eager host submission gaps",
        "aggregation": "arithmetic mean of five per-run arithmetic means; shared process",
        "status": "running",
        "start_unix": time.time(),
        "required_nccl": args.required_nccl,
        "repeats": 5,
        "sizes_full_array_bytes": sizes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "gpus": args.gpus,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "environment_overrides": overrides,
        "inherited_environment": {
            k: os.environ[k]
            for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")
            if k in os.environ
        },
        "removed_nccl_overrides": {
            k: v for k, v in os.environ.items() if k.startswith("NCCL_")
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "observations": [
            command(cmd)
            for cmd in (
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                ["git", "-C", str(ROOT), "status", "--short"],
                ["nvidia-smi", "-q"],
                ["nvidia-smi", "topo", "-m"],
                ["lscpu"],
                ["uname", "-a"],
                ["ldd", str(binary)],
            )
        ],
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "benchmarks/cpu_router/collectives_bench.cu",
                ROOT / "benchmarks/cpu_router/nccl_identity.hpp",
                ROOT / "src/cpu_router/packet_pipeline.cu",
                ROOT / "include/uccl_lite/cpu_router/packet_pipeline.hpp",
                ROOT / "src/cpu_router/collective_dispatch.cc",
                Path(__file__).resolve(),
            ]
        },
        "commands": [],
    }

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    summary = []
    try:
        for operation in ("allgather", "reducescatter"):
            samples = out / f"{operation}.samples.csv"
            cmd = [
                str(binary),
                "--gpus",
                str(args.gpus),
                "--op",
                operation,
                "--begin",
                str(args.begin),
                "--end",
                str(args.end),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--repeats",
                "5",
                "--packet-bytes",
                str(args.packet_bytes),
                "--calibrate",
                str(out / f"{operation}.profile"),
                "--samples",
                str(samples),
            ]
            manifest["commands"].append(cmd)
            if args.gpu_reduce:
                cmd.append("--gpu-reduce")
            if args.interleave_scatter:
                cmd.append("--interleave-scatter")
            save()
            print(
                f"Running {operation}: {len(sizes)} sizes, 5 paired repetitions",
                flush=True,
            )
            result = command(cmd, env)
            (out / f"{operation}.stdout.txt").write_text(result["stdout"])
            (out / f"{operation}.stderr.txt").write_text(result["stderr"])
            if result["returncode"] or "# correctness: PASS" not in result["stdout"]:
                raise RuntimeError(f"{operation} failed; inspect retained logs")
            if f"fallback=nccl:{args.required_nccl};" not in result["stdout"]:
                raise RuntimeError(
                    "NCCL version mismatch; refusing to publish a summary"
                )
            routes = re.findall(
                r"Channel .*? via (\S+)", result["stdout"] + result["stderr"]
            )
            if not routes or any(not route.startswith("SHM/") for route in routes):
                raise RuntimeError(
                    "could not verify exclusively host-staged NCCL channels"
                )
            manifest.setdefault("verified_nccl_routes", {})[operation] = sorted(
                set(routes)
            )
            summary.extend(
                summarize(
                    csv.DictReader(io.StringIO(samples.read_text())),
                    operation,
                    sizes,
                    5,
                    args.iterations,
                    args.gpus,
                    router_backend,
                )
            )
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        manifest["status"] = "complete_local_diagnostic"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["end_unix"] = time.time()
        save()
    print(f"Saved {len(summary)} rows to {out}; no claim against ThunderEP.")


if __name__ == "__main__":
    main()
