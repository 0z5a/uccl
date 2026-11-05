#!/usr/bin/env python3
"""Run an explicitly labeled independent ThunderEP communication reimplementation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from compare_thunderep import command, summarize

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("thunderep-reimpl-dma", "thunderep-reimpl-sm", "uccl-lite-gpu-progress")


def summaries(rows, sizes, repeats, iterations, ranks):
    expected_names = {"nccl", *BACKENDS}
    if any(row["backend"] not in expected_names for row in rows):
        raise ValueError("unexpected backend")
    if any(row["collective"] not in {"all_gather", "reduce_scatter"} for row in rows):
        raise ValueError("unexpected collective")
    result = []
    for operation in ("allgather", "reducescatter"):
        canonical = "all_gather" if operation == "allgather" else "reduce_scatter"
        for index, backend in enumerate(BACKENDS):
            pair = [
                r
                for r in rows
                if r["collective"] == canonical and r["backend"] in {"nccl", backend}
            ]
            result.extend(
                r
                for r in summarize(
                    pair, operation, sizes, repeats, iterations, ranks, backend
                )
                if index == 0 or r["backend"] != "nccl"
            )
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "build/paper-reference/uccl_lite_host_staged_compare",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--begin", type=int, default=1024)
    parser.add_argument("--end", type=int, default=1 << 30)
    parser.add_argument("--chunk-bytes", type=int, default=8 << 20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if (
        args.begin < 1024
        or args.end < args.begin
        or args.end > 1 << 30
        or args.begin & (args.begin - 1)
        or args.end & (args.end - 1)
        or args.iterations < 1
        or args.warmup < 1
    ):
        parser.error("require powers-of-two sizes, positive counts, and end <= 1 GiB")
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sizes = [1 << n for n in range(args.begin.bit_length() - 1, args.end.bit_length())]
    env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
    env.update(
        NCCL_P2P_DISABLE="1",
        NCCL_IB_DISABLE="1",
        NCCL_NET_GDR_LEVEL="LOC",
        NCCL_DEBUG="INFO",
        CUDA_MODULE_LOADING="EAGER",
    )
    argv = [
        str(binary),
        "--gpus",
        "4",
        "--begin",
        str(args.begin),
        "--end",
        str(args.end),
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--repeats",
        "5",
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(out / "samples.csv"),
    ]
    sources = [
        ROOT / "Makefile",
        ROOT / "benchmarks/cpu_router/nccl_identity.hpp",
        Path(__file__),
        ROOT / "scripts/compare_thunderep.py",
    ]
    for directory in (
        "src/host_staged",
        "include/uccl_lite/host_staged",
        "src/cpu_router",
        "include/uccl_lite/cpu_router",
        "benchmarks/host_staged",
    ):
        sources.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    manifest = {
        "status": "running",
        "start_unix": time.time(),
        "argv": argv,
        "reference": "Independent implementation of arXiv:2609.40093v1 communication mechanisms; not author source",
        "paper_setup_reproduced": False,
        "model_level_evaluation": False,
        "scope": "4 L4, BF16, out-of-place, eager slowest-rank CUDA events including host-submission gaps; all backends coexist; correctness outside timing",
        "order": "four-backend order rotates by repetition; five repetitions in one process; 5 is not a multiple of 4",
        "ranks": 4,
        "sizes": sizes,
        "repeats": 5,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "chunk_bytes": args.chunk_bytes,
        "router_packet_bytes": 8 << 20,
        "binary_sha256": sha(binary),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))
        },
        "environment": {
            k: v
            for k, v in env.items()
            if k.startswith("NCCL_") or k == "CUDA_MODULE_LOADING"
        },
        "inherited_nccl_overrides_removed": {
            k: v for k, v in os.environ.items() if k.startswith("NCCL_")
        },
        "hardware": [
            command(["nvidia-smi", "-q"]),
            command(["nvidia-smi", "topo", "-m"]),
            command(["lscpu"]),
            command(["/usr/local/cuda/bin/nvcc", "--version"]),
        ],
    }

    def save():
        (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        with (
            (out / "stdout.txt").open("w") as stdout,
            (out / "stderr.txt").open("w") as stderr,
        ):
            completed = subprocess.run(
                argv, env=env, stdout=stdout, stderr=stderr, check=False
            )
        log = (out / "stdout.txt").read_text() + (out / "stderr.txt").read_text()
        if completed.returncode or "# correctness: PASS" not in log:
            raise RuntimeError("benchmark failed; inspect retained logs")
        if "# NCCL=23007 " not in log:
            raise RuntimeError("requires NCCL 2.30.7")
        routes = re.findall(r"Channel .*? via (\S+)", log)
        if not routes or any(not r.startswith("SHM/") for r in routes):
            raise RuntimeError("cannot verify exclusively host-staged NCCL channels")
        manifest["verified_nccl_routes"] = sorted(set(routes))
        with (out / "samples.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        result = summaries(rows, sizes, 5, args.iterations, 4)
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(result[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(result)
        manifest["data_sha256"] = {
            name: sha(out / name) for name in ("samples.csv", "summary.csv")
        }
        manifest["checked_timed_calls"] = len(rows)
        manifest["status"] = "complete_local_reimplementation_diagnostic"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["end_unix"] = time.time()
        save()
    print(
        f"Saved {len(rows)} checked timed calls to {out}; independent reimplementation, not author ThunderEP."
    )


if __name__ == "__main__":
    main()
