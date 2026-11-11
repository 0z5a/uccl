#!/usr/bin/env python3
"""Draw optimized AR/AG/RS paths with the current eight-L4 figure style."""

import argparse

from figure_style import ROOT, configure
import plot_refresh_collectives as reference


OPTIMIZED = ROOT / "benchmarks/data/optimization_20261110"
MIDRANGE_AG = ROOT / "benchmarks/data/ag_midrange_4l4_20261111"
SERIAL_RS = ROOT / "benchmarks/data/rs_ar_serial_4l4_20261111"
OPERATIONS = (
    ("l40_selected.csv", "operation", "allgather",
     "collectives-4l4-allgather-optimized", 3 / 4),
    ("l40_selected.csv", "operation", "reducescatter",
     "collectives-4l4-reducescatter-optimized", 3 / 4),
    ("multinode_summary.csv", "collective", "allreduce",
     "collectives-2x4l4-allreduce-optimized", 14 / 8),
    ("multinode_summary.csv", "collective", "allgather",
     "collectives-2x4l4-allgather-optimized", 7 / 8),
    ("multinode_summary.csv", "collective", "reducescatter",
     "collectives-2x4l4-reducescatter-optimized", 7 / 8),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure()
    for source, field, operation, filename, factor in OPERATIONS:
        if filename == "collectives-4l4-allgather-optimized":
            reference.DATA = MIDRANGE_AG
        elif filename == "collectives-4l4-reducescatter-optimized":
            reference.DATA = SERIAL_RS
        else:
            reference.DATA = OPTIMIZED
        data = reference.load(source, field, operation)
        reference.plot_operation(data, filename, factor, args.preview)


if __name__ == "__main__":
    main()
