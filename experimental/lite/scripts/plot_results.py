#!/usr/bin/env python3
"""Render the recorded UCCL Lite results as compact, readable SVG figures."""

import argparse
import csv
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import AMBER, BLUE, GRID, MUTED, ROOT, TEAL
from figure_style import clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def positive(values, label: str) -> None:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"missing or invalid values in {label}")


def short_size(size: int) -> str:
    for divisor, suffix in ((1 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")):
        if size >= divisor:
            return f"{size / divisor:g}{suffix}"
    return str(size)


def line_figure(title: str, subtitle: str, ticks: list[int], ylim: tuple[float, float]):
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    heading(fig, title, subtitle)
    fig.subplots_adjust(left=0.095, right=0.965, top=0.79, bottom=0.19)
    clean_axis(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlim(ticks[0] / 1.4, ticks[-1] * 1.35)
    ax.set_xticks(ticks, [short_size(size) for size in ticks])
    ax.set_ylim(*ylim)
    ax.set_xlabel("Complete tensor bytes per rank  ·  K = KiB, M = MiB, G = GiB")
    return fig, ax


def speedup_series(ax, sizes, ratios, label, color, bounds=None, *, dashed=False):
    positive(ratios, label)
    if bounds is not None:
        low, high = bounds
        if any(lo > value or value > hi for lo, value, hi in zip(low, ratios, high)):
            raise ValueError(f"invalid timing range for {label}")
        ax.fill_between(sizes, low, high, color=color, alpha=0.09, linewidth=0)
    ax.plot(
        sizes,
        ratios,
        color=color,
        linewidth=2.2,
        linestyle=(0, (5, 3)) if dashed else "-",
        marker="o",
        markersize=4.8,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=label,
    )


def allreduce(preview: bool) -> None:
    data = {(int(row["bytes"]), row["backend"]): row for row in rows("allreduce_single_node.csv")}
    sizes = sorted(size for size, backend in data if backend == "uccl_lite_main")
    if len(sizes) != 21:
        raise ValueError("AllReduce sweep must have 21 sizes")
    fig, ax = line_figure(
        "Single-node AllReduce",
        "4 × L4  ·  FP32  ·  UCCL Lite chooses its fastest correct variant per size",
        [1 << power for power in (10, 14, 18, 22, 26, 30)],
        (0.75, 3.55),
    )
    for backend, label, color, dashed in (
        ("nccl218", "vs NCCL 2.18.3", BLUE, False),
        ("nccl228", "vs NCCL 2.28.7", TEAL, True),
    ):
        baseline = [data[size, backend] for size in sizes]
        candidate = [data[size, "uccl_lite_main"] for size in sizes]
        ratios = [float(b["median_us"]) / float(c["median_us"])
                  for b, c in zip(baseline, candidate)]
        low = [float(b["min_us"]) / float(c["max_us"])
               for b, c in zip(baseline, candidate)]
        high = [float(b["max_us"]) / float(c["min_us"])
                for b, c in zip(baseline, candidate)]
        speedup_series(ax, sizes, ratios, label, color, (low, high), dashed=dashed)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9.5)
    save(fig, "allreduce", preview)


def alltoall(preview: bool) -> None:
    data = {int(row["size_bytes"]): row for row in rows("collectives.csv")
            if row["collective"] == "alltoall"}
    sizes = sorted(data)
    ratios = [float(data[size]["nccl_us"]) / float(data[size]["uccl_lite_us"])
              for size in sizes]
    fig, ax = line_figure(
        "Single-node All-to-All",
        "4 × L4  ·  FP32  ·  fastest correct UCCL Lite schedule per size",
        [1 << power for power in (10, 14, 18, 22, 26, 28)],
        (0.8, 4.6),
    )
    speedup_series(ax, sizes, ratios, "vs default NCCL 2.28.7", BLUE)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
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
        fig, ax = line_figure(
            f"Historical eight-GPU {title}",
            "8 × L4  ·  BF16  ·  no GPU peer access  ·  CPU-router vs NCCL 2.27.5",
            [128, 4096, 131072, 4194304, 134217728, 1073741824],
            (0.8, 2.12),
        )
        for mode, label, color, dashed in (
            ("in_place", "In place", BLUE, False),
            ("out_of_place", "Out of place", TEAL, True),
        ):
            sizes = sorted(size for op, placement, size, backend in data
                           if op == operation and placement == mode and backend == "kernel")
            if len(sizes) != 24:
                raise ValueError(f"incomplete {title} {mode} sweep")
            router = [data[operation, mode, size, "kernel"] for size in sizes]
            nccl = [data[operation, mode, size, "nccl"] for size in sizes]
            ratios = [float(b["median_us"]) / float(r["median_us"])
                      for b, r in zip(nccl, router)]
            low = [float(b["min_us"]) / float(r["max_us"])
                   for b, r in zip(nccl, router)]
            high = [float(b["max_us"]) / float(r["min_us"])
                    for b, r in zip(nccl, router)]
            speedup_series(ax, sizes, ratios, label, color, (low, high), dashed=dashed)
        ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9.5)
        save(fig, name, preview)


def dense_records() -> dict[str, list[tuple[str, int, float]]]:
    text = (DATA / "dense_fusion.txt").read_text()
    records: dict[str, list[tuple[str, int, float]]] = {name: [] for name in ("AG", "RS", "AR")}
    pattern = (r"^(\S+)\s+(AG|RS)\s+\((\d+),(\d+),(\d+)\)\s+"
               r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$")
    for match in re.finditer(pattern, text, re.MULTILINE):
        model, operation, m, _, _, baseline, _, fused, _ = match.groups()
        records[operation].append((model, int(m), float(baseline) / float(fused)))
    pattern_ar = (r"^(\S+)\s+\((\d+),\d+,\d+\)\s+\d+\s+"
                  r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+[\d.]+x$")
    for match in re.finditer(pattern_ar, text, re.MULTILINE):
        model, m, baseline, _, fused = match.groups()
        records["AR"].append((model, int(m), float(baseline) / float(fused)))
    if any(len(points) != 8 for points in records.values()):
        raise ValueError("dense fusion comparison must have eight shapes per operation")
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
    heading(overview, "Dense communication + GEMM fusion",
            "4 × L4  ·  FP16  ·  eight shapes per operation; large dot = reference shape")
    overview.subplots_adjust(left=0.25, right=0.96, top=0.78, bottom=0.20)
    ax.set_xlim(0.55, 1.62)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                  ["0.6×", "0.8×", "1.0×", "1.2×", "1.4×", "1.6×"])
    ax.set_yticks([2, 1, 0],
                  ["AllGather + GEMM", "GEMM + ReduceScatter", "GEMM + AllReduce"])
    ax.tick_params(axis="both", which="both", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(1, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
    for operation, center in (("AG", 2), ("RS", 1), ("AR", 0)):
        values = [point[2] for point in records[operation]]
        positive(values, operation)
        ax.hlines(center, min(values), max(values), color=GRID,
                  linewidth=3, zorder=1)
        offsets = np.linspace(-0.16, 0.16, len(values))
        for value, offset in zip(values[:-1], offsets[:-1]):
            ax.scatter(value, center + offset, s=48,
                       color=BLUE if value >= 1 else AMBER,
                       alpha=0.72, edgecolor="white", linewidth=0.7, zorder=2)
        reference = values[-1]
        ax.scatter(reference, center, s=110, color=BLUE,
                   edgecolor="white", linewidth=1, zorder=3)
    ax.set_xlabel("Speedup over cuBLAS + NCCL  ·  dashed line = 1×")
    save(overview, "dense-fusion", preview)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.3))
    heading(fig, "Dense communication + GEMM fusion",
            "4 × L4  ·  FP16  ·  one fused variant across eight recorded shapes", left=0.075)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.77, bottom=0.15, wspace=0.62)
    for ax, operation, title in zip(
        axes,
        ("AG", "RS", "AR"),
        ("AllGather + GEMM", "GEMM + ReduceScatter", "GEMM + AllReduce"),
    ):
        points = records[operation]
        values = [point[2] for point in points]
        positive(values, title)
        y = np.arange(len(points))
        ax.set_title(title, loc="left", pad=15)
        ax.set_xlim(0.55, 1.58)
        ax.set_ylim(len(points) - 0.4, -0.65)
        ax.set_xticks([0.6, 1.0, 1.4], ["0.6×", "1.0×", "1.4×"])
        ax.set_yticks(y, [dense_label(model, m) for model, m, _ in points])
        ax.tick_params(axis="both", which="both", length=0, pad=5, labelsize=8.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.axvline(1, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
        for index, value in enumerate(values):
            color = BLUE if value >= 1 else AMBER
            ax.hlines(index, min(1, value), max(1, value), color=color, linewidth=2.4)
            ax.scatter([value], [index], color=color, s=42, zorder=3,
                       edgecolor="white", linewidth=0.7)
        ax.annotate(f"{values[-1]:.2f}×", (values[-1], len(points) - 1),
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=BLUE)
    save(fig, "dense-fusion-shapes", preview)


def moe_rows(name: str) -> dict[tuple[int, str], float]:
    points = {}
    for line in (DATA / name).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            points[int(fields[0]), fields[3]] = float(fields[4])
    return points


def moe(preview: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    heading(fig, "Expert-parallel MoE fusion",
            "4 × L4  ·  BF16  ·  speedup over NCCL All-to-All-v + grouped cuBLAS")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.77, bottom=0.20, wspace=0.26)
    for ax, name, baseline, fused, title, ticks in (
        (axes[0], "moe_dispatch.txt", "nccl+cublas-grp", "fused-cutlass-gather",
         "Dispatch + gate/up GEMM", [1 << power for power in (10, 12, 14, 16)]),
        (axes[1], "moe_layer.txt", "nccl+cublas", "fused",
         "Full layer", [1 << power for power in (3, 7, 11, 15)]),
    ):
        points = moe_rows(name)
        sizes = sorted(size for size, variant in points if variant == fused)
        values = [points[size, baseline] / points[size, fused] for size in sizes]
        positive(values, title)
        clean_axis(ax)
        ax.set_title(title, loc="left", pad=13)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.35)
        ax.set_xticks(ticks, [short_size(size) for size in ticks])
        ax.set_ylim(0.35, 1.75)
        ax.plot(sizes, values, color=BLUE, linewidth=2.3)
        ax.scatter(sizes, values, c=[BLUE if value >= 1 else AMBER for value in values],
                   s=43, zorder=3, edgecolor="white", linewidth=0.7)
        ax.annotate(f"{values[-1]:.2f}×", (sizes[-1], values[-1]),
                    xytext=(-5, 12), textcoords="offset points", ha="right",
                    fontsize=9.5, fontweight="bold", color=BLUE)
        ax.set_xlabel("Tokens per rank  ·  K = 1,024")
    axes[1].set_ylabel("")
    save(fig, "moe-fusion", preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG and PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    allreduce(args.preview)
    alltoall(args.preview)
    primitive_collectives(args.preview)
    dense(args.preview)
    moe(args.preview)


if __name__ == "__main__":
    main()
