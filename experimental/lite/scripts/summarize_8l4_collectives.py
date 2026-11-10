#!/usr/bin/env python3
"""Select the measured eight-L4 UCCL Lite path for each BF16 collective."""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"
SIZES = [1 << power for power in range(20, 31)]


def rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    text = path.read_text()
    if "# correctness: PASS" not in text:
        raise ValueError(f"missing output verification: {path}")
    result = {}
    for row in csv.DictReader(line for line in text.splitlines()
                              if not line.startswith("#")):
        operation = row.get("collective", "all_to_all")
        key = operation, int(row["bytes"]), row["backend"]
        if key in result:
            raise ValueError(f"duplicate measurement: {path}, {key}")
        timing = float(row["median_us"])
        if not math.isfinite(timing) or timing <= 0:
            raise ValueError(f"invalid timing: {path}, {key}")
        result[key] = row
    return result


def main() -> None:
    results = []
    local_dir = DATA / "local_fallback_8l4_20261110_v3"
    paired_dir = DATA / "router_vs_nccl_8l4_20261110"
    for operation, placement in (("all_reduce", "in_place"),
                                 ("all_gather", "out_of_place"),
                                 ("reduce_scatter", "out_of_place")):
        local = rows(local_dir / f"{placement}_calibration.csv")
        paired = rows(paired_dir / f"{placement}_calibration.csv")
        for size in SIZES:
            key = operation, size
            router = float(local[operation, size, "cpu-router"]["median_us"])
            fallback = float(local[operation, size, "fallback"]["median_us"])
            if router >= fallback:
                raise ValueError(f"CPU router is not the fastest local path: {key}")
            candidate = float(paired[operation, size, "cpu-router"]["median_us"])
            control = float(paired[operation, size, "fallback"]["median_us"])
            results.append((operation, size, placement, "cpu-router", candidate,
                            control, control / candidate))
    alltoall = rows(DATA / "alltoall_8l4_20261110/summary.csv")
    for size in SIZES:
        variants = (
            ("host-staging", float(alltoall["all_to_all", size, "uccl-host"]["median_us"])),
            ("peer-host", float(alltoall["all_to_all", size, "uccl-hybrid"]["median_us"])),
        )
        path, candidate = min(variants, key=lambda item: item[1])
        control = float(alltoall["all_to_all", size, "nccl"]["median_us"])
        results.append(("all_to_all", size, "out_of_place", path, candidate,
                        control, control / candidate))
    output = DATA / "collectives_8l4_20261110.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("collective", "bytes", "placement", "uccl_path",
                         "uccl_median_us", "nccl_median_us", "speedup"))
        writer.writerows(results)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
