#!/usr/bin/env python3
"""Draw the matched two-node AG/RS comparison from validated summary data."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/multinode_peer_20261110/summary.csv"
OUTPUT = ROOT / "docs/assets/multinode-peer.svg"
SIZES = [1 << power for power in range(20, 27)]


def main() -> None:
    with DATA.open() as stream:
        rows = {(row["collective"], int(row["bytes"])): row
                for row in csv.DictReader(stream)}
    fig, axis = plt.subplots(figsize=(10.2, 4.5))
    colors = {"allgather": "#174a78", "reducescatter": "#55a0c8"}
    names = {"allgather": "AllGather", "reducescatter": "ReduceScatter"}
    width = 0.36
    maximum_ratio = 1.35
    for operation, shift in (("allgather", -width / 2),
                             ("reducescatter", width / 2)):
        selected = [rows[(operation, size)] for size in SIZES]
        values = [float(row["speedup"]) for row in selected]
        lower = [float(row["paired_min_speedup"]) for row in selected]
        upper = [float(row["paired_max_speedup"]) for row in selected]
        maximum_ratio = max(maximum_ratio, max(upper) * 1.08)
        if any(lo > value or value > hi
               for lo, value, hi in zip(lower, values, upper)):
            raise ValueError("ratio of medians lies outside paired run range")
        axis.bar([index + shift for index in range(len(SIZES))], values,
                 width, color=colors[operation], label=names[operation],
                 yerr=[[value - lo for value, lo in zip(values, lower)],
                       [hi - value for value, hi in zip(values, upper)]],
                 capsize=2, error_kw={"elinewidth": 1})
    axis.axhline(1, color="#333333", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(SIZES)), [f"{size >> 20}" for size in SIZES])
    axis.set_xlabel("Complete output (AG) or input (RS) per rank (MiB)")
    axis.set_ylabel("Speedup over NCCL 2.28.7 (×)")
    axis.set_ylim(0, maximum_ratio)
    axis.set_title("Two nodes × four L4 GPUs · FP32 · out of place")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.01,
             "Bars: ratio of three-run medians; whiskers: paired run ratio "
             "range. Values below 1× are slower than NCCL.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT)
    # Matplotlib leaves trailing spaces in SVG path commands. They are not
    # meaningful geometry, and removing them keeps generated diffs lintable.
    OUTPUT.write_text("\n".join(line.rstrip() for line in OUTPUT.read_text().splitlines()) + "\n")
    if "--preview" in sys.argv[1:]:
        preview = ROOT / "build/figures/multinode-peer.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=160)


if __name__ == "__main__":
    main()
