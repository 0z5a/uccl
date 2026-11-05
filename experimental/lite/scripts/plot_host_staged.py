#!/usr/bin/env python3
"""Plot measured independent reference implementations; never substitute paper numbers."""

import argparse
import csv
import hashlib
import json
import statistics

import plot_results as style
from compare_host_staged import summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    data = style.ROOT / "benchmarks/data/host-staged-reference"
    manifest = json.loads((data / "provenance.json").read_text())
    for name, digest in manifest["data_sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived data checksum mismatch: {name}")
    with (data / "samples.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    summary = summaries(
        raw,
        manifest["sizes"],
        manifest["repeats"],
        manifest["iterations"],
        manifest["ranks"],
    )
    lookup = {
        (r["collective"], r["backend"], r["full_array_bytes"]): r for r in summary
    }
    samples = {}
    for row in raw:
        key = (row["collective"], int(row["bytes"]), row["backend"], int(row["repeat"]))
        samples.setdefault(key, []).append(float(row["slowest_rank_us"]))
    variants = (
        ("thunderep-reimpl-dma", "ThunderEP reimpl. · DMA", style.LIGHT),
        ("thunderep-reimpl-sm", "ThunderEP reimpl. · SM", style.MEDIUM),
        ("uccl-lite-gpu-progress", "UCCL Lite · GPU progress", style.NAVY),
    )
    style.plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": style.GRID,
            "axes.labelcolor": style.MUTED,
            "text.color": style.INK,
            "xtick.color": style.MUTED,
            "ytick.color": style.MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "uccl-lite-host-staged-reference",
        }
    )
    fig, axes = style.figure(
        rows=2,
        title="No-P2P collectives · executable paper-reference comparison",
        subtitle="4 × NVIDIA L4  /  BF16  /  eager calls  /  NCCL 2.30.7  /  independent reimplementation, not author ThunderEP",
    )
    for row, operation in enumerate(("allgather", "reducescatter")):
        canonical = "all_gather" if operation == "allgather" else "reduce_scatter"
        for col, exponents in enumerate((range(10, 21), range(21, 31))):
            sizes = [1 << n for n in exponents]
            series = []
            for backend, label, color in variants:
                values, lows, highs = [], [], []
                for size in sizes:
                    values.append(
                        lookup[operation, backend, size]["speedup_vs_local_nccl"]
                    )
                    ratios = [
                        statistics.mean(samples[canonical, size, "nccl", r])
                        / statistics.mean(samples[canonical, size, backend, r])
                        for r in range(manifest["repeats"])
                    ]
                    lows.append(min(ratios))
                    highs.append(max(ratios))
                series.append((label, color, values, (lows, highs)))
            upper = max(1.25, max(max(s[3][1]) for s in series) * 1.12)
            labels = [
                style.size_label(s)
                .replace(" KiB", "K")
                .replace(" MiB", "M")
                .replace(" GiB", "G")
                for s in sizes
            ]
            title = ("AllGather" if row == 0 else "ReduceScatter") + (
                " · 1 KiB–1 MiB" if col == 0 else " · 2 MiB–1 GiB"
            )
            style.bars(
                axes[row, col],
                labels,
                series,
                title,
                "NCCL / measured backend (×)",
                ylim=upper,
                speedup=True,
            )
            axes[row, col].set_xlabel("Full-array size")
    (style.ROOT / "docs/assets/experiments").mkdir(parents=True, exist_ok=True)
    (style.ROOT / "build/figures/experiments").mkdir(parents=True, exist_ok=True)
    style.save(
        fig,
        "experiments/host-staged-reference",
        [(label, color) for _, label, color in variants],
        "Dashed line: measured NCCL = 1×. Five-run means; whiskers: min–max paired run ratios, not confidence intervals. All 21 sizes retained.\n"
        "Reference DMA and SM are separate curves, not a per-size oracle. Includes eager host-submission gaps; SM results are not decode-graph timings.\n"
        "No original-author implementation or model inference is measured. Different hardware and disclosed implementation choices prevent a paper-level superiority claim.",
        args.preview,
    )


if __name__ == "__main__":
    main()
