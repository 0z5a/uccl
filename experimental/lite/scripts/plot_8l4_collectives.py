#!/usr/bin/env python3
"""Plot the best measured eight-L4 BF16 UCCL Lite path against NCCL."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from figure_style import BASELINE, BLUE, GRID, MUTED, ROOT, configure, heading, save


DATA = ROOT / "benchmarks/data/collectives_8l4_20261110.csv"
OPERATIONS = (("all_reduce", "AllReduce"),
              ("all_gather", "AllGather"),
              ("reduce_scatter", "ReduceScatter"),
              ("all_to_all", "All-to-All"))
SIZES = ((1 << 20, "1 MiB per rank"),
         (1 << 26, "64 MiB per rank"),
         (1 << 30, "1 GiB per rank"))


def load() -> dict[tuple[str, int], dict[str, str]]:
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = row["collective"], int(row["bytes"])
        if key in result:
            raise ValueError(f"duplicate eight-L4 workload: {key}")
        for field in ("uccl_median_us", "nccl_median_us"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid timing for {key}: {field}")
        result[key] = row
    expected = {(operation, 1 << power) for operation, _ in OPERATIONS
                for power in range(20, 31)}
    if set(result) != expected:
        raise ValueError("incomplete eight-L4 BF16 sweep")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.3))
    heading(fig, "Eight-L4 single-node collectives",
            "BF16  ·  AR in place; AG, RS, A2A out of place  ·  slowest-rank CUDA events",
            left=0.075)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.70, bottom=0.20,
                        wspace=0.48)
    for ax, (size, title) in zip(axes, SIZES):
        y = np.arange(len(OPERATIONS))
        uccl = [float(data[operation, size]["uccl_median_us"]) / 1000
                for operation, _ in OPERATIONS]
        nccl = [float(data[operation, size]["nccl_median_us"]) / 1000
                for operation, _ in OPERATIONS]
        ax.barh(y + 0.18, nccl, height=0.32, color=BASELINE,
                label="NCCL 2.28.9")
        ax.barh(y - 0.18, uccl, height=0.32, color=BLUE,
                label="UCCL Lite best measured path")
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlim(0, max(uccl + nccl) * 1.24)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim(len(OPERATIONS) - 0.55, -0.55)
        ax.set_yticks(y, [name for _, name in OPERATIONS])
        ax.set_xlabel("Latency (ms) ↓")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.tick_params(axis="both", which="both", length=0, pad=6)
        for index, (candidate, baseline) in enumerate(zip(uccl, nccl)):
            offset = max(uccl + nccl) * 0.012
            label = lambda value: f"{value:.2f}" if value < 10 else f"{value:.1f}"
            ax.text(candidate + offset, index - 0.18, label(candidate),
                    va="center", fontsize=8.5)
            ax.text(baseline + offset, index + 0.18, label(baseline),
                    va="center", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right",
               bbox_to_anchor=(0.985, 0.85), ncol=2, frameon=False,
               fontsize=9.5)
    fig.text(0.11, 0.045,
             "UCCL: CPU router for AR/AG/RS, host staging for A2A. Each panel has its own linear millisecond scale.",
             fontsize=9, color=MUTED)
    save(fig, "collectives-8l4", args.preview)


if __name__ == "__main__":
    main()
