#!/usr/bin/env python3
"""Plot eight-L4 BF16 latency and throughput for each collective."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import BASELINE, BLUE, ROOT, clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data/collectives_8l4_20261110.csv"
OPERATIONS = (
    ("all_reduce", "AllReduce", "allreduce", 14 / 8, "bus"),
    ("all_gather", "AllGather", "allgather", 7 / 8, "bus"),
    ("reduce_scatter", "ReduceScatter", "reducescatter", 7 / 8, "bus"),
)
SMALL_SIZES = [1 << power for power in range(10, 21)]
LARGE_SIZES = [1 << power for power in range(20, 31)]


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
    expected = {(operation, 1 << power) for operation, *_ in OPERATIONS
                for power in range(10, 31)}
    if set(result) != expected:
        raise ValueError("incomplete eight-L4 BF16 sweep")
    return result


def throughput(size: int, latency_us: float, factor: float) -> float:
    return factor * size / latency_us / 1000  # decimal GB/s


def plot_operation(data, operation, title, filename, factor, bandwidth_kind,
                   preview):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    subtitle = ("BF16  ·  128 KiB GPU progress; 256 KiB–4 MiB small packets"
                "  ·  slowest-rank CUDA events"
                if operation == "reduce_scatter" else
                "BF16  ·  complete bytes per rank  ·  slowest-rank CUDA events")
    heading(fig, f"Eight-L4 single-node {title}",
            subtitle, left=0.075)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.18,
                        wspace=0.28)
    for ax, sizes, measure, panel_title, ylabel, ticks in (
        (axes[0], SMALL_SIZES, lambda size, us: us,
         "1 KiB–1 MiB · latency ↓", "Latency (µs)",
         [1 << power for power in (10, 12, 14, 16, 18, 20)]),
        (axes[1], LARGE_SIZES, lambda size, us: throughput(size, us, factor),
         f"1 MiB–1 GiB · {bandwidth_kind} throughput ↑",
         f"{bandwidth_kind.title()} throughput (GB/s)",
         [1 << power for power in (20, 22, 24, 26, 28, 30)]),
    ):
        clean_axis(ax)
        ax.set_title(panel_title, loc="left", pad=12)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ax.set_xticks(ticks, [
            f"{size >> 10}K" if size < 1 << 20 else
            f"{size >> 20}M" if size < 1 << 30 else "1G"
            for size in ticks
        ])
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel(ylabel)
        upper = 0.0
        for field, label, color in (
            ("nccl_median_us", "NCCL 2.28.9", BASELINE),
            ("uccl_median_us", "uccl-lite", BLUE),
        ):
            values = [measure(size, float(data[operation, size][field]))
                      for size in sizes]
            upper = max(upper, *values)
            ax.plot(sizes, values, color=color, linewidth=2.2, marker="o",
                    markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, label=label)
        ax.set_ylim(0, upper * 1.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right",
               bbox_to_anchor=(0.985, 0.85), ncol=2, frameon=False,
               fontsize=9.5)
    save(fig, f"collectives-8l4-{filename}-v2", preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    data = load()
    for operation, title, filename, factor, bandwidth_kind in OPERATIONS:
        plot_operation(data, operation, title, filename, factor,
                       bandwidth_kind, args.preview)


if __name__ == "__main__":
    main()
