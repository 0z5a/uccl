#!/usr/bin/env python3
"""Render UCCL Lite reference results as reproducible blue grouped bar charts."""

import argparse
import csv
from io import StringIO
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"
PALE, LIGHT, MEDIUM, NAVY = "#d8e8f5", "#a8c9e5", "#729dc6", "#173e6c"
INK, MUTED, GRID = "#19324d", "#61758b", "#e4edf5"


def read_csv(name):
    with (DATA / name).open() as source:
        return list(csv.DictReader(source))


def size_label(value):
    for divisor, unit in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if value >= divisor:
            return f"{value / divisor:g} {unit}"
    return f"{value:g} B"


def token_label(value):
    return f"{value // 1024}k" if value >= 1024 else str(value)


def figure(rows=1, columns=2, title="", subtitle=""):
    fig, axes = plt.subplots(
        rows, columns, figsize=(13.2, 2.95 * rows + 2.0), squeeze=False
    )
    fig.suptitle(title, x=0.065, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.065,
        0.98 - 0.25 / fig.get_figheight(),
        subtitle,
        fontsize=10,
        color=MUTED,
        va="top",
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=1 - 1.2 / fig.get_figheight(),
        bottom=1.08 / fig.get_figheight(),
        hspace=0.65,
        wspace=0.22,
    )
    return fig, axes


def bars(ax, labels, series, title, ylabel, ylim=None, speedup=False, annotate=False):
    """Series: (label, color, values[, (low_values, high_values)]). Axes start at zero."""
    positions = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, item in enumerate(series):
        label, color, raw = item[:3]
        values = np.asarray(raw, dtype=float)
        if (
            len(values) != len(labels)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(f"Invalid data in {title}: {label}")
        errors = None
        if len(item) == 4:
            low, high = map(np.asarray, item[3])
            errors = np.array([values - low, high - values])
            if np.any(errors < -1e-8):
                raise ValueError(f"Invalid range in {title}: {label}")
            errors = np.maximum(errors, 0)
        group = ax.bar(
            positions + (index - (len(series) - 1) / 2) * width,
            values,
            width=width * 0.91,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            yerr=errors,
            error_kw={"ecolor": MUTED, "capsize": 2, "elinewidth": 0.7},
        )
        if annotate and index == len(series) - 1:
            for bar, value in zip(group, values):
                ax.annotate(
                    f"{value:.2f}×" if speedup else f"{value:.1f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=NAVY,
                    fontweight="bold",
                )
    ax.set_xticks(positions, labels)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=6)
    if speedup:
        ax.axhline(1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))


def save(fig, name, legend, footer, preview):
    if legend:
        handles = [Patch(facecolor=color, label=label) for label, color in legend]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.52, 1 - 0.57 / fig.get_figheight()),
            frameon=False,
            fontsize=9.5,
            handlelength=1.5,
            columnspacing=2,
        )
    fig.text(0.065, 0.02, footer, fontsize=8.5, color=MUTED, va="bottom")
    output = ROOT / f"docs/assets/{name}.svg"
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    )
    if preview:
        folder = ROOT / "build/figures"
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=140)
        fig.savefig(
            folder / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
    plt.close(fig)
    print(output.relative_to(ROOT))


def collective_series(rows, sizes, backends, bandwidth=False):
    lookup = {(int(row["bytes"]), row["backend"]): row for row in rows}
    series = []
    for backend, label, color in backends:
        selected = [lookup[size, backend] for size in sizes]
        values = [
            float(row["bus_bw_gbs" if bandwidth else "median_us"]) for row in selected
        ]
        if bandwidth:
            lows = [
                v * float(r["median_us"]) / float(r["max_us"])
                for v, r in zip(values, selected)
            ]
            highs = [
                v * float(r["median_us"]) / float(r["min_us"])
                for v, r in zip(values, selected)
            ]
        else:
            lows = [float(row["min_us"]) for row in selected]
            highs = [float(row["max_us"]) for row in selected]
        series.append((label, color, values, (lows, highs)))
    return series


def allreduce(preview):
    backends = [
        ("nccl218", "NCCL 2.18.3", PALE),
        ("nccl228", "NCCL 2.28.7", LIGHT),
        ("tccl_kernel", "TCCL", MEDIUM),
        ("uccl_lite_main", "UCCL Lite", NAVY),
    ]
    rows = read_csv("allreduce_single_node.csv")
    fig, axes = figure(
        title="Single-node AllReduce",
        subtitle="4 × NVIDIA L4  /  float32  /  representative message sizes",
    )
    for ax, sizes, bandwidth, title, ylabel in (
        (
            axes[0, 0],
            [1 << n for n in (10, 14, 15, 17, 18)],
            False,
            "Latency ↓",
            "Latency (µs)",
        ),
        (
            axes[0, 1],
            [1 << n for n in (20, 21, 22, 26, 30)],
            True,
            "Bus bandwidth ↑",
            "Bus bandwidth (GB/s)",
        ),
    ):
        bars(
            ax,
            [size_label(s) for s in sizes],
            collective_series(rows, sizes, backends, bandwidth),
            title,
            ylabel,
        )
    save(
        fig,
        "allreduce",
        [(b[1], b[2]) for b in backends],
        "Three-run medians; whiskers show min–max. UCCL Lite selects the fastest correct variant per size.\n"
        "The 32 KiB and 1–4 MiB groups retain sizes where a reference implementation wins.",
        preview,
    )


def primitive_collectives(preview):
    """Historical eight-GPU CPU-router data, separate from the four-GPU kernels."""
    rows = read_csv("primitives_8l4.csv")
    assert len(rows) == 576
    backends = [
        ("nccl_tccl", "NCCL 2.18.3 (TCCL runtime)", PALE),
        ("nccl", "NCCL 2.27.5", LIGHT),
        ("tccl", "TCCL", MEDIUM),
        ("kernel", "UCCL Lite · CPU-router SM", NAVY),
    ]
    for collective, name, title in (
        ("all_gather", "allgather", "AllGather"),
        ("reduce_scatter", "reducescatter", "ReduceScatter"),
    ):
        fig, axes = figure(
            rows=2,
            title=f"Standalone {title} · eight-GPU primitives",
            subtitle="Single node  /  8 × NVIDIA L4  /  BF16  /  no GPU peer access  /  historical CPU-router branch",
        )
        for row_index, mode in enumerate(("in_place", "out_of_place")):
            selected = [
                r for r in rows if r["collective"] == collective and r["mode"] == mode
            ]
            assert len(selected) == 96
            placement = "In-place" if mode == "in_place" else "Out-of-place"
            for column, sizes, bandwidth, metric in (
                (0, [1 << n for n in (10, 14, 17, 20)], False, "Latency ↓"),
                (1, [1 << n for n in (22, 24, 26, 30)], True, "Bus bandwidth ↑"),
            ):
                bars(
                    axes[row_index, column],
                    [size_label(s) for s in sizes],
                    collective_series(selected, sizes, backends, bandwidth),
                    f"{placement} · {metric}",
                    "Bus bandwidth (GB/s)" if bandwidth else "Latency (µs)",
                )
        extent = "complete output" if collective == "all_gather" else "complete input"
        save(
            fig,
            name,
            [(b[1], b[2]) for b in backends],
            "Same-session three-run medians and min–max whiskers; identical per-collective nccl-tests binaries. No per-size backend selection.\n"
            f"Size = {extent} per rank; bus bandwidth = algorithm bandwidth × 7/8. Historical eight-GPU results; see CPU-router docs for port validation.",
            preview,
        )


def alltoall(preview):
    rows = {
        int(r["size_bytes"]): r
        for r in read_csv("collectives.csv")
        if r["collective"] == "alltoall"
    }
    fig, axes = figure(
        title="Single-node All-to-All",
        subtitle="4 × NVIDIA L4  /  float32  /  out-of-place transfers",
    )
    legend = [("NCCL 2.28.7 (default transport)", LIGHT), ("UCCL Lite", NAVY)]
    for ax, sizes, bandwidth in (
        (axes[0, 0], [1 << n for n in (10, 14, 16, 17, 18)], False),
        (axes[0, 1], [1 << n for n in (20, 22, 25, 28)], True),
    ):
        series = []
        for (label, color), field in zip(legend, ("nccl_us", "uccl_lite_us")):
            values = [
                (
                    3 * size / float(rows[size][field]) / 1000
                    if bandwidth
                    else float(rows[size][field])
                )
                for size in sizes
            ]
            series.append((label, color, values))
        bars(
            ax,
            [size_label(s) for s in sizes],
            series,
            "Aggregate bandwidth ↑" if bandwidth else "Latency ↓",
            "Aggregate bandwidth (GB/s)" if bandwidth else "Latency (µs)",
            annotate=True,
        )
    save(
        fig,
        "alltoall",
        legend,
        "Best recorded UCCL Lite variant per size; no repeat ranges are available in this summary.\n"
        "NCCL defaults to host staging on this testbed. Aggregate bandwidth = 3 × bytes per rank / latency.",
        preview,
    )


def dense(preview):
    text = (DATA / "dense_fusion.txt").read_text()
    pattern = r"^(\S+)\s+(AG|RS)\s+\((\d+),(\d+),(\d+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$"
    records = {operation: [] for operation in ("AG", "RS", "AR")}
    for match in re.finditer(pattern, text, re.MULTILINE):
        model, operation, m, n, k, base, unfused, fused, flux = match.groups()
        records[operation].append(
            (model, int(m), *map(float, (base, unfused, fused, flux)))
        )
    pattern_ar = r"^(\S+)\s+\((\d+),\d+,\d+\)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+[\d.]+x$"
    for match in re.finditer(pattern_ar, text, re.MULTILINE):
        model, m, base, unfused, fused = match.groups()
        records["AR"].append((model, int(m), *map(float, (base, unfused, fused))))
    fig, axes = figure(
        rows=3,
        columns=1,
        title="Dense communication–GEMM fusion",
        subtitle="4 × NVIDIA L4  /  FP16  /  speedup over the same cuBLAS + NCCL baseline ↑",
    )
    legend = [
        ("UCCL Lite · no overlap", LIGHT),
        ("Flux fused", MEDIUM),
        ("UCCL Lite · fused", NAVY),
    ]
    models = {
        "Llama-3.1-8B": "Llama 8B",
        "Llama-3.1-70B": "Llama 70B",
        "Qwen2.5-32B": "Qwen 32B",
        "Flux-reference": "Reference",
    }
    for ax, operation, title in zip(
        axes[:, 0],
        ("AG", "RS", "AR"),
        ("AllGather + GEMM", "GEMM + ReduceScatter", "GEMM + AllReduce"),
    ):
        points = records[operation]
        assert len(points) == 8, (operation, len(points))
        labels = [f"{models[p[0]]}\nM={p[1]:,}" for p in points]
        series = [(legend[0][0], LIGHT, [p[2] / p[3] for p in points])]
        if operation != "AR":
            series.append((legend[1][0], MEDIUM, [p[2] / p[5] for p in points]))
        series.append((legend[2][0], NAVY, [p[2] / p[4] for p in points]))
        bars(
            ax,
            labels,
            series,
            title,
            "Speedup (×)",
            ylim=1.65,
            speedup=True,
            annotate=True,
        )
    save(
        fig,
        "dense-fusion",
        legend,
        "Dashed line: cuBLAS + NCCL = 1×. Three-run median latencies; ratios computed from the current comparison block.\n"
        "Flux uses a different harness (1–2% differences are ties); no Flux GEMM + AllReduce result is available. Small-shape losses are retained.",
        preview,
    )


def moe_rows(name):
    points = {}
    for line in (DATA / name).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            points[int(fields[0]), fields[3]] = float(fields[4])
    return points


def ep_moe(preview):
    fig, axes = figure(
        rows=2,
        columns=1,
        title="Expert-parallel MoE fusion",
        subtitle="4 × NVIDIA L4  /  DeepSeek-V2-Lite shape  /  EP=4, BF16  /  speedup ↑",
    )
    legend = [
        ("UCCL Lite + grouped cuBLAS", PALE),
        ("UCCL Lite + CUTLASS · unfused", LIGHT),
        ("UCCL Lite · fused", NAVY),
    ]
    for ax, name, baseline, collective, fused, title in (
        (
            axes[0, 0],
            "moe_dispatch.txt",
            "nccl+cublas-grp",
            "ours+cublas-grp",
            "fused-cutlass-gather",
            "Dispatch + gate/up GEMM",
        ),
        (
            axes[1, 0],
            "moe_layer.txt",
            "nccl+cublas",
            "ours+cublas",
            "fused",
            "Full layer: dispatch → GEMMs / SwiGLU → combine",
        ),
    ):
        points = moe_rows(name)
        sizes = sorted(size for size, variant in points if variant == fused)
        series = [
            (label, color, [points[s, baseline] / points[s, variant] for s in sizes])
            for (label, color), variant in zip(
                legend, (collective, "ours+cutlass", fused)
            )
        ]
        bars(
            ax,
            [token_label(s) for s in sizes],
            series,
            title,
            "Speedup (×)",
            ylim=1.85,
            speedup=True,
            annotate=True,
        )
        ax.set_xlabel("Tokens per rank (k = 1,024)")
    save(
        fig,
        "moe-fusion",
        legend,
        "Dashed line: NCCL All-to-All-v + grouped cuBLAS = 1×. Routing-plan construction is excluded.\n"
        "Settled, interleaved timing; min over rounds. The two panels have different recorded token ranges. Full-layer small-batch regressions are shown.",
        preview,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also export PNG/PDF copies to ignored build/figures/.",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "uccl-lite-bars",
        }
    )
    for render in (
        allreduce,
        primitive_collectives,
        alltoall,
        dense,
        ep_moe,
    ):
        render(args.preview)


if __name__ == "__main__":
    main()
