#!/usr/bin/env python3
"""Render absolute latency and bandwidth from recorded UCCL Lite results."""

import argparse
import csv
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import AMBER, BASELINE, BLUE, GRID, ROOT, SECOND_BASELINE
from figure_style import clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def positive(values, label: str) -> None:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"missing or invalid values in {label}")


def size_label(size: int) -> str:
    for divisor, suffix in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if size >= divisor:
            return f"{size / divisor:g} {suffix}"
    return f"{size} B"


def two_panels(title: str, subtitle: str, *, height: float = 4.7):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, height))
    heading(fig, title, subtitle, left=0.075)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.17, wspace=0.29)
    return fig, axes


def size_axis(ax, sizes, ticks, title: str, ylabel: str, top: float) -> None:
    clean_axis(ax)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(sizes) / 1.4, max(sizes) * 1.35)
    ax.set_xticks(ticks, [size_label(size) for size in ticks])
    ax.set_ylim(0, top)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Complete tensor bytes per rank")


def measured_line(ax, sizes, values, label: str, color: str, *, dashed=False):
    positive(values, label)
    ax.plot(sizes, values, color=color, linewidth=2.2,
            linestyle=(0, (5, 3)) if dashed else "-",
            marker="o", markersize=4.8, markeredgecolor="white",
            markeredgewidth=0.7, label=label)


def allreduce(preview: bool) -> None:
    data = {(int(row["bytes"]), row["backend"]): row
            for row in rows("allreduce_single_node.csv")}
    all_sizes = sorted(size for size, backend in data if backend == "uccl_lite_main")
    if len(all_sizes) != 21:
        raise ValueError("AllReduce sweep must have 21 sizes")
    small = [size for size in all_sizes if size <= 1 << 19]
    large = [size for size in all_sizes if size >= 1 << 20]
    series = (
        ("UCCL Lite", "uccl_lite_main", BLUE, False),
        ("NCCL 2.18.3", "nccl218", BASELINE, False),
        ("NCCL 2.28.7", "nccl228", SECOND_BASELINE, True),
    )
    fig, axes = two_panels(
        "Single-node AllReduce",
        "4 × L4  ·  FP32  ·  three-run medians  ·  UCCL variant chosen per size",
    )
    for ax, sizes, field, title, ylabel, ticks in (
        (axes[0], small, "median_us", "Small messages · latency ↓", "Latency (µs)",
         [1 << power for power in (10, 14, 18)]),
        (axes[1], large, "bus_bw_gbs", "Large messages · bus bandwidth ↑",
         "Bus bandwidth (GB/s)", [1 << power for power in (20, 22, 26, 30)]),
    ):
        values_by_series = []
        for label, backend, color, dashed in series:
            values = [float(data[size, backend][field]) for size in sizes]
            measured_line(ax, sizes, values, label, color, dashed=dashed)
            values_by_series.extend(values)
        size_axis(ax, sizes, ticks, title, ylabel, max(values_by_series) * 1.16)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.985, 0.855), ncol=3, frameon=False, fontsize=9.5)
    save(fig, "allreduce", preview)


def alltoall(preview: bool) -> None:
    data = {int(row["size_bytes"]): row for row in rows("collectives.csv")
            if row["collective"] == "alltoall"}
    all_sizes = sorted(data)
    small = [size for size in all_sizes if size <= 1 << 18]
    large = [size for size in all_sizes if size >= 1 << 20]
    fig, axes = two_panels(
        "Single-node All-to-All",
        "4 × L4  ·  FP32  ·  fastest correct UCCL schedule per size",
    )
    for ax, sizes, title, ylabel, ticks, bandwidth in (
        (axes[0], small, "Small messages · latency ↓", "Latency (µs)",
         [1 << power for power in (10, 14, 18)], False),
        (axes[1], large, "Large messages · aggregate bandwidth ↑",
         "Aggregate bandwidth (GB/s)",
         [1 << power for power in (20, 22, 26, 28)], True),
    ):
        values_by_series = []
        for label, field, color in (
            ("UCCL Lite", "uccl_lite_us", BLUE),
            ("NCCL 2.28.7", "nccl_us", BASELINE),
        ):
            latencies = [float(data[size][field]) for size in sizes]
            values = ([3 * size / latency / 1000
                       for size, latency in zip(sizes, latencies)]
                      if bandwidth else latencies)
            measured_line(ax, sizes, values, label, color)
            values_by_series.extend(values)
        size_axis(ax, sizes, ticks, title, ylabel, max(values_by_series) * 1.16)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.985, 0.855), ncol=2, frameon=False, fontsize=9.5)
    save(fig, "alltoall", preview)


def primitive_collectives(preview: bool) -> None:
    data = {(row["collective"], row["mode"], int(row["bytes"]), row["backend"]): row
            for row in rows("primitives_8l4.csv")}
    if len(data) != 288:
        raise ValueError("eight-L4 primitive archive is incomplete")
    for operation, name, title in (
        ("all_gather", "allgather", "AllGather"),
        ("reduce_scatter", "reducescatter", "ReduceScatter"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
        heading(fig, f"Historical eight-GPU {title}",
                "8 × L4  ·  BF16  ·  no GPU peer access  ·  three-run medians",
                left=0.075)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.74, bottom=0.105,
                            hspace=0.48, wspace=0.28)
        all_small_values = []
        all_large_values = []
        for row_index, mode in enumerate(("in_place", "out_of_place")):
            sizes = sorted(size for op, placement, size, backend in data
                           if op == operation and placement == mode and backend == "kernel")
            if len(sizes) != 24:
                raise ValueError(f"incomplete {title} {mode} sweep")
            small = [size for size in sizes if size <= 1 << 19]
            large = [size for size in sizes if size >= 1 << 20]
            placement = "In place" if mode == "in_place" else "Out of place"
            for column, selected, field, metric, ylabel, ticks, collected in (
                (0, small, "median_us", "latency ↓", "Latency (µs)",
                 [128, 4096, 131072], all_small_values),
                (1, large, "bus_bw_gbs", "bus bandwidth ↑",
                 "Bus bandwidth (GB/s)",
                 [1 << power for power in (20, 24, 28, 30)], all_large_values),
            ):
                ax = axes[row_index, column]
                for label, backend, color in (
                    ("CPU-router SM", "kernel", BLUE),
                    ("NCCL 2.27.5", "nccl", BASELINE),
                ):
                    values = [float(data[operation, mode, size, backend][field])
                              for size in selected]
                    measured_line(ax, selected, values, label, color)
                    collected.extend(values)
                size_axis(ax, selected, ticks, f"{placement} · {metric}", ylabel, 1)
        for row in range(2):
            axes[row, 0].set_ylim(0, max(all_small_values) * 1.15)
            axes[row, 1].set_ylim(0, max(all_large_values) * 1.15)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(0.985, 0.86), ncol=2, frameon=False,
                   fontsize=9.5)
        save(fig, name, preview)


def dense_records() -> dict[str, list[tuple[str, int, float, float]]]:
    text = (DATA / "dense_fusion.txt").read_text()
    records: dict[str, list[tuple[str, int, float, float]]] = {
        name: [] for name in ("AG", "RS", "AR")
    }
    pattern = (r"^(\S+)\s+(AG|RS)\s+\((\d+),(\d+),(\d+)\)\s+"
               r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$")
    for match in re.finditer(pattern, text, re.MULTILINE):
        model, operation, m, _, _, baseline, _, fused, _ = match.groups()
        records[operation].append((model, int(m), float(baseline), float(fused)))
    pattern_ar = (r"^(\S+)\s+\((\d+),\d+,\d+\)\s+\d+\s+"
                  r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+[\d.]+x$")
    for match in re.finditer(pattern_ar, text, re.MULTILINE):
        model, m, baseline, _, fused = match.groups()
        records["AR"].append((model, int(m), float(baseline), float(fused)))
    if any(len(points) != 8 for points in records.values()):
        raise ValueError("dense comparison must have eight shapes per operation")
    for operation, points in records.items():
        positive([value for _, _, baseline, fused in points
                  for value in (baseline, fused)], operation)
    return records


def dense_label(model: str, m: int) -> str:
    name = {
        "Llama-3.1-8B": "Llama 8B",
        "Llama-3.1-70B": "Llama 70B",
        "Qwen2.5-32B": "Qwen 32B",
        "Flux-reference": "Reference",
    }[model]
    return f"{name} · {m // 1024}k" if m >= 1024 else f"{name} · {m}"


def dense(preview: bool) -> None:
    records = dense_records()
    overview, ax = plt.subplots(figsize=(9.6, 4.0))
    heading(overview, "Dense fusion · large reference shapes",
            "4 × L4  ·  FP16  ·  complete operation latency")
    overview.subplots_adjust(left=0.27, right=0.96, top=0.77, bottom=0.19)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks([2, 1, 0],
                  ["AllGather + GEMM", "GEMM + ReduceScatter", "GEMM + AllReduce"])
    ax.tick_params(axis="both", which="both", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for operation, center in (("AG", 2), ("RS", 1), ("AR", 0)):
        _, _, baseline, fused = records[operation][-1]
        ax.hlines(center, fused, baseline, color=GRID, linewidth=3)
        ax.scatter(baseline, center, s=90, color=BASELINE,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(fused, center, s=100, color=BLUE,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("Latency (ms)  ·  lower is better")
    overview.legend(
        handles=[plt.Line2D([0], [0], marker="o", linestyle="", color=BLUE,
                            markersize=8, label="UCCL Lite fused"),
                 plt.Line2D([0], [0], marker="o", linestyle="", color=BASELINE,
                            markersize=8, label="cuBLAS + NCCL")],
        loc="upper right", bbox_to_anchor=(0.965, 0.86), ncol=2,
        frameon=False, fontsize=9.5,
    )
    save(overview, "dense-fusion", preview)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.0))
    heading(fig, "Dense fusion · all recorded shapes",
            "4 × L4  ·  FP16  ·  complete operation latency", left=0.075)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.70, bottom=0.13, wspace=0.62)
    for ax, operation, title in zip(
        axes,
        ("AG", "RS", "AR"),
        ("AllGather + GEMM", "GEMM + ReduceScatter", "GEMM + AllReduce"),
    ):
        points = records[operation]
        y = np.arange(len(points))
        ax.set_title(title, loc="left", pad=15)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 32)
        ax.set_ylim(len(points) - 0.4, -0.65)
        ax.set_xticks([0.1, 1, 10, 30], ["0.1", "1", "10", "30"])
        ax.set_yticks(y, [dense_label(model, m) for model, m, _, _ in points])
        ax.tick_params(axis="both", which="both", length=0, pad=5, labelsize=8.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for index, (_, _, baseline, fused) in enumerate(points):
            ax.hlines(index, min(baseline, fused), max(baseline, fused),
                      color=GRID, linewidth=2)
            ax.scatter(baseline, index, color=BASELINE, s=38, zorder=2,
                       edgecolor="white", linewidth=0.7)
            ax.scatter(fused, index, color=BLUE if fused < baseline else AMBER,
                       s=44, zorder=3, edgecolor="white", linewidth=0.7)
        ax.set_xlabel("Latency (ms) ↓")
    fig.legend(
        handles=[plt.Line2D([0], [0], marker="o", linestyle="", color=BLUE,
                            markersize=7, label="UCCL Lite fused"),
                 plt.Line2D([0], [0], marker="o", linestyle="", color=BASELINE,
                            markersize=7, label="cuBLAS + NCCL"),
                 plt.Line2D([0], [0], marker="o", linestyle="", color=AMBER,
                            markersize=7, label="Fused regression")],
        loc="upper right", bbox_to_anchor=(0.985, 0.84), ncol=3,
        frameon=False, fontsize=9,
    )
    save(fig, "dense-fusion-shapes", preview)


def moe_rows(name: str) -> dict[tuple[int, str], float]:
    points = {}
    for line in (DATA / name).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            points[int(fields[0]), fields[3]] = float(fields[4])
    return points


def moe(preview: bool) -> None:
    points = moe_rows("moe_layer.txt")
    small = [8, 32, 128, 512, 1024]
    large = [2048, 4096, 8192, 16384, 32768]
    fig, axes = two_panels(
        "Expert-parallel MoE · full layer",
        "4 × L4  ·  DeepSeek-shaped BF16  ·  routing-plan construction excluded",
    )
    for ax, sizes, title, ylabel, ticks, throughput in (
        (axes[0], small, "Small batches · latency ↓", "Latency (ms)",
         [8, 32, 128, 1024], False),
        (axes[1], large, "Large batches · token throughput ↑",
         "Throughput (k tokens/s/rank)", [2048, 8192, 32768], True),
    ):
        values_by_series = []
        for label, variant, color in (
            ("UCCL Lite fused", "fused", BLUE),
            ("NCCL + grouped cuBLAS", "nccl+cublas", BASELINE),
        ):
            latencies = [points[size, variant] / 1000 for size in sizes]
            values = ([size / latency for size, latency in zip(sizes, latencies)]
                      if throughput else latencies)
            measured_line(ax, sizes, values, label, color)
            values_by_series.extend(values)
        size_axis(ax, sizes, ticks, title, ylabel, max(values_by_series) * 1.16)
        ax.set_xticks(ticks, [f"{size // 1024}k" if size >= 1024 else str(size)
                              for size in ticks])
        ax.set_xlabel("Tokens per rank")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.985, 0.855), ncol=2, frameon=False,
               fontsize=9.5)
    save(fig, "moe-fusion", preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    allreduce(args.preview)
    alltoall(args.preview)
    primitive_collectives(args.preview)
    dense(args.preview)
    moe(args.preview)


if __name__ == "__main__":
    main()
