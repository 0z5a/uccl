#!/usr/bin/env python3
"""Build SHM figure data, reusing the measured 8xL4 AllGather router curve."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHM = ROOT / "benchmarks/data/shm_8l4_20261111"
AG = ROOT / "benchmarks/data/ag_packet_8l4_20261111/summary.csv"


def main():
    with (SHM / "summary.csv").open() as stream:
        shm = list(csv.DictReader(stream))
    with AG.open() as stream:
        ag_rows = {int(row["bytes"]): row for row in csv.DictReader(stream)}
    expected = {1 << power for power in range(10, 31)}
    if len(shm) != 63 or set(ag_rows) != expected:
        raise ValueError("incomplete 8xL4 input summaries")
    for row in shm:
        row["uccl_source"] = "shm_8l4_20261111"
        row["nccl_source"] = "shm_8l4_20261111"
        if row["collective"] != "all_gather":
            continue
        size = int(row["bytes"])
        ag = ag_rows[size]
        for field in ("uccl_run1_us", "uccl_run2_us", "uccl_median_us"):
            row[field] = ag[field]
        row["speedup"] = f'{float(row["nccl_median_us"]) / float(row["uccl_median_us"]):.4f}'
        row["uccl_source"] = "ag_packet_8l4_20261111"
    output = SHM / "figure_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=shm[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(shm)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
