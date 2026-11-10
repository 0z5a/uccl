#!/usr/bin/env python3
"""Draw absolute bandwidth and latency from paired two-node collective data."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import BASELINE, BLUE, GRID, ROOT
from figure_style import clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data/multinode_peer_20261110"


def read(name: str) -> dict[tuple[str, int], dict[str, str]]:
    with (DATA / name).open() as stream:
        return {(row["collective"], int(row["bytes"])): row
                for row in csv.DictReader(stream)}


def value(row: dict[str, str], backend: str) -> float:
    number = float(row[f"{backend}_median_us"])
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid {backend} timing")
    return number


def bus_bandwidth(row: dict[str, str], operation: str, backend: str) -> float:
    factor = 1.75 if operation == "allreduce" else 0.875
    return int(row["bytes"]) / (value(row, backend) * 1e-6) / 1e9 * factor


def bandwidth_figure(preview: bool, small, large, allreduce) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    heading(fig, "Two-node large-message bandwidth",
            "2 × 4 L4  ·  FP32  ·  three paired runs  ·  NCCL 2.28.7", left=0.075)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.71, bottom=0.17, wspace=0.34)
    for ax, size, title in ((axes[0], 1 << 26, "64 MiB per rank"),
                            (axes[1], 1 << 30, "1 GiB per rank")):
        clean_axis(ax)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([2, 1, 0], ["AllGather", "ReduceScatter", "AllReduce"])
        ax.set_xlabel("Bus bandwidth (GB/s) ↑")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for operation, y in (("allgather", 2), ("reducescatter", 1),
                             ("allreduce", 0)):
            records = allreduce if operation == "allreduce" else (
                small if size <= 1 << 26 else large)
            row = records[operation, size]
            candidate = bus_bandwidth(row, operation, "uccl")
            baseline = bus_bandwidth(row, operation, "nccl")
            ax.hlines(y, min(candidate, baseline), max(candidate, baseline),
                      color=GRID, linewidth=3)
            ax.scatter(baseline, y, color=BASELINE, s=72, zorder=2,
                       edgecolor="white", linewidth=0.8)
            ax.scatter(candidate, y, color=BLUE, s=82, zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.text(candidate + 0.35, y, f"{candidate:.1f}", va="center",
                    color=BLUE, fontsize=9.5, fontweight="bold")
    fig.legend(
        handles=[plt.Line2D([0], [0], marker="o", linestyle="", color=BLUE,
                            markersize=8, label="UCCL Lite"),
                 plt.Line2D([0], [0], marker="o", linestyle="", color=BASELINE,
                            markersize=8, label="NCCL 2.28.7")],
        loc="upper right", bbox_to_anchor=(0.985, 0.85), ncol=2,
        frameon=False, fontsize=9.5,
    )
    save(fig, "multinode-peer", preview)


def latency_figure(preview: bool, small) -> None:
    sizes = [1 << power for power in (10, 12, 14, 16, 18, 20)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    heading(fig, "Two-node small-message latency",
            "2 × 4 L4  ·  FP32  ·  out of place  ·  NCCL 2.28.7", left=0.075)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.71, bottom=0.17, wspace=0.29)
    for ax, operation, title in ((axes[0], "allgather", "AllGather"),
                                 (axes[1], "reducescatter", "ReduceScatter")):
        clean_axis(ax)
        ax.set_title(title, loc="left", pad=12)
        maximum = 0
        for backend, label, color in (("uccl", "UCCL Lite", BLUE),
                                      ("nccl", "NCCL 2.28.7", BASELINE)):
            latencies = [value(small[operation, size], backend) for size in sizes]
            maximum = max(maximum, *latencies)
            ax.plot(sizes, latencies, color=color, linewidth=2.2, marker="o",
                    markersize=5, markeredgecolor="white", markeredgewidth=0.7,
                    label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.3)
        ax.set_xticks([1 << power for power in (10, 14, 20)],
                      ["1 KiB", "16 KiB", "1 MiB"])
        ax.set_ylim(0, maximum * 1.16)
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel("Latency (µs) ↓")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.985, 0.85), ncol=2, frameon=False,
               fontsize=9.5)
    save(fig, "multinode-latency", preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    small = read("summary.csv")
    large = read("summary_large.csv")
    allreduce = read("summary_ar.csv")
    bandwidth_figure(args.preview, small, large, allreduce)
    latency_figure(args.preview, small)


if __name__ == "__main__":
    main()
