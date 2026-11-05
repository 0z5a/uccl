#!/usr/bin/env python3
"""Render the four-L4 AllReduce reference figure from the committed summary."""

import csv
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SERIES = (
    ("nccl218", "NCCL 2.18.3", "#94a3b8", "s"),
    ("nccl228", "NCCL 2.28.7", "#d97706", "D"),
    ("tccl_kernel", "TCCL", "#0f766e", "^"),
    ("uccl_lite_main", "UCCL Lite", "#2563eb", "o"),
)


def size_label(value, _position=None):
    for divisor, unit in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if value >= divisor:
            return f"{value / divisor:g} {unit}"
    return f"{value:g} B"


def main():
    with (ROOT / "benchmarks/data/allreduce_single_node.csv").open() as source:
        rows = list(csv.DictReader(source))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "svg.fonttype": "none",
            "svg.hashsalt": "uccl-lite-allreduce",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for backend, label, color, marker in SERIES:
        points = sorted(
            (row for row in rows if row["backend"] == backend),
            key=lambda row: int(row["bytes"]),
        )
        if not points:
            raise ValueError(f"Missing series: {backend}")
        for ax, bandwidth in zip(axes, (False, True)):
            selected = [
                row for row in points
                if (int(row["bytes"]) >= 1 << 20 if bandwidth else int(row["bytes"]) <= 1 << 20)
            ]
            sizes = [int(row["bytes"]) for row in selected]
            if bandwidth:
                values = [float(row["bus_bw_gbs"]) for row in selected]
                lows = [int(row["bytes"]) * 1.5 / float(row["max_us"]) / 1000 for row in selected]
                highs = [int(row["bytes"]) * 1.5 / float(row["min_us"]) / 1000 for row in selected]
            else:
                values = [float(row["median_us"]) for row in selected]
                lows = [float(row["min_us"]) for row in selected]
                highs = [float(row["max_us"]) for row in selected]
            ax.plot(sizes, values, label=label, color=color, marker=marker,
                    markersize=4, linewidth=2)
            ax.fill_between(sizes, lows, highs, color=color, alpha=0.12, linewidth=0)
    axes[0].set_title("Small messages · lower is better", loc="left", fontweight="bold")
    axes[0].set_ylabel("Latency (µs)")
    axes[0].set_yscale("log")
    axes[0].xaxis.set_major_locator(FixedLocator([1 << n for n in (10, 14, 18, 20)]))
    axes[1].set_title("Large messages · higher is better", loc="left", fontweight="bold")
    axes[1].set_ylabel("AllReduce bus bandwidth (GB/s)")
    axes[1].set_ylim(bottom=0)
    axes[1].xaxis.set_major_locator(FixedLocator([1 << n for n in (20, 24, 28, 30)]))
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(size_label))
        ax.set_xlabel("Message size per rank")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.96))
    fig.text(0.5, 0.02, "4 × NVIDIA L4 · float32 · three-run medians and min–max bands · best correct UCCL Lite variant per size",
             ha="center", fontsize=9, color="#64748b")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.79, wspace=0.25)
    output = ROOT / "docs/assets/allreduce.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    # Matplotlib leaves spaces at SVG path-line endings; normalize for Git.
    output.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    plt.close(fig)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
