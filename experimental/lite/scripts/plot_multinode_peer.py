#!/usr/bin/env python3
"""Draw the matched large-message two-node comparison from validated CSVs."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import BLUE, PURPLE, ROOT, TEAL
from figure_style import clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data/multinode_peer_20261110"
SIZES = [1 << power for power in range(25, 31)]


def read(name: str) -> dict[tuple[str, int], dict[str, str]]:
    with (DATA / name).open() as stream:
        return {(row["collective"], int(row["bytes"])): row
                for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    small = read("summary.csv")
    large = read("summary_large.csv")
    allreduce = read("summary_ar.csv")
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    heading(fig, "Two-node collectives",
            "2 × 4 L4  ·  FP32  ·  AR in place; AG/RS out of place  ·  NCCL 2.28.7")
    fig.subplots_adjust(left=0.095, right=0.965, top=0.79, bottom=0.19)
    clean_axis(ax)
    for operation, label, color, records in (
        ("allgather", "AllGather", BLUE, small | large),
        ("reducescatter", "ReduceScatter", TEAL, small | large),
        ("allreduce", "AllReduce", PURPLE, allreduce),
    ):
        selected = [records[operation, size] for size in SIZES]
        values = [float(row["speedup"]) for row in selected]
        lower = [float(row["paired_min_speedup"]) for row in selected]
        upper = [float(row["paired_max_speedup"]) for row in selected]
        if any(not all(math.isfinite(x) and x > 0 for x in (lo, value, hi))
               or lo > value or value > hi
               for lo, value, hi in zip(lower, values, upper)):
            raise ValueError(f"invalid paired ratio range for {operation}")
        ax.fill_between(SIZES, lower, upper, color=color, alpha=0.10, linewidth=0)
        ax.plot(SIZES, values, marker="o", markersize=6, linewidth=2.4,
                markeredgecolor="white", markeredgewidth=0.7,
                color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlim(SIZES[0] / 1.25, SIZES[-1] * 1.23)
    ax.set_xticks(SIZES, ["32M", "64M", "128M", "256M", "512M", "1G"])
    ax.set_xlabel("Complete tensor bytes per rank  ·  M = MiB, G = GiB")
    ax.set_ylim(0.94, 1.29)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=9.5)
    save(fig, "multinode-peer", args.preview)


if __name__ == "__main__":
    main()
