# Standalone AllGather and ReduceScatter reference results

These are historical results from the related CPU-router primitive branch,
whose runtime is not included in this subtree. The current Makefile does not
build standalone AllGather or ReduceScatter executables. The included dense
fusion benchmarks exercise AG/RS as part of communication–GEMM pipelines;
their timings must not be used as standalone collective measurements.

## Provenance

The imported [summary](../benchmarks/data/primitives_8l4.csv) is the complete
matched four-backend rerun committed as
`9d088c97442659a41a3db2225659d71f3bac6ab5` in the source project's history.
It includes the ReduceScatter fix from `822fdcb`, not the earlier October 10
baseline or exploratory before/after tuning samples. The same summary remains
in the separate CPU-router/EP development worktree.

The byte-for-byte source CSV SHA-256 is
`ebd47273efe552d4cd03144c60a8e4d39cf6594f2aa79e0f1805f11eabb2edfe`.
It contains 576 summary rows: three collectives × two placements × four backends
× 24 sizes. AllReduce rows are retained for completeness; the README uses this
file only for the eight-GPU standalone AllGather and ReduceScatter figures.

No application integration, adapter library, or CPU-router runtime is imported
with this dataset. No new GPU benchmark was run for these figures.

## Hardware and method

- One node, eight NVIDIA L4 GPUs, CUDA 12.9, driver 580.178.04.
- NUMA map `[0,0,0,0,1,1,1,1]`; CUDA reports no GPU peer access.
- BF16, sum reduction for ReduceScatter; 128 B–1 GiB doubling sizes.
- One process and one submitting thread pinned to CPU 23.
- Five warmups, ten timed iterations, three repetitions with backend order
  rotated between repetitions; both in-place and out-of-place are measured.
- All four backends use the same per-collective unmodified nccl-tests 2.21.1
  executable, revision `afd59abab774a6a4b492eb344bd0c845a0b7bb4d`, built against
  NCCL 2.18.3 headers. Buffers use `cudaMalloc`; `-p 1` ensures allocation in
  that version's initialization path.
- The archived full rerun reports zero wrong values in all 1,728
  run/size/placement measurements across its 36 sweeps.

| CSV backend | Figure label | Meaning |
| --- | --- | --- |
| `kernel` | CPU-router SM · historical | GPU SM transfer kernels over registered host packets, CPU operators, two-slot ownership protocol |
| `nccl` | NCCL 2.27.5 | Native newer NCCL reference |
| `nccl_tccl` | NCCL 2.18.3 (TCCL runtime) | TCCL's shared library with `NCCL_ALGO` and `TCCL_XML_FILE` unset; default NCCL algorithms |
| `tccl` | TCCL | Same TCCL runtime with its topology-selected algorithm forced |

TCCL uses revision `351d064856e322ec6e4546808e7d1e433d24f941`. Its single-node
pathfinder was restricted to the eight-GPU topology; the runtime was unmodified.
The selected ring was `0→6→3→1→4→5→2→7→0` with host-memory staging. This is
not the two-node TCCL transport experiment elsewhere in this README.

## Primitive implementation and the ReduceScatter fix

The CPU-router runtime composes collectives from registered memory spans,
GPU/host channels, CPU operators, generation counters, and reader credits.
AllGather publishes host packet views and has each GPU gather rank-ordered
slices. ReduceScatter uses a BF16 reduction tree with FP32 accumulation and
documented BF16 rounding boundaries. Two packet slots overlap upload, processing,
and consumption; storage is not reused until its readers release ownership.

The measured ReduceScatter fix sizes packets from the destination shard extent,
within 128–512 KiB packet bounds, and raises the worker cap to 16 per NUMA node,
still constrained by the 8,192-element CPU grain. It improves medium-message
overlap. AllGather keeps its existing packet geometry. The figures use one
explicit SM transport, with no internal NCCL fallback or per-size backend switch.

## Reading the figures

For both operations, decimal bus bandwidth is algorithm bandwidth × `7/8`.
AllGather's byte count is the complete output on each rank; ReduceScatter's
byte count is the complete input on each rank. It is not a single peer's payload
and not one physical PCIe link's wire rate.

Each figure separates in-place and out-of-place and uses representative sizes
for latency and bandwidth. Bars are three-run medians; whiskers show the recorded
min–max range. Every numeric row remains available in the CSV. These eight-GPU
BF16 results cannot be directly ranked against the four-GPU float32 results.

At 1 GiB in-place:

| Collective | CPU-router SM | NCCL 2.27.5 | NCCL 2.18.3 (TCCL runtime) | TCCL |
| --- | ---: | ---: | ---: | ---: |
| AllGather | 5.25 GB/s | 5.04 GB/s | 5.09 GB/s | 5.09 GB/s |
| ReduceScatter | 5.59 GB/s | 4.86 GB/s | 4.89 GB/s | 4.88 GB/s |

Regenerate both plots, together with the other reference figures, using
`python3 scripts/plot_results.py --preview`. This redraws committed measurements;
reproducing the benchmark requires the separate source revision and eight-GPU
testbed described above.
