# Topology and hardware bandwidth profiler

[hardware_profiler.cu](../benchmarks/probes/hardware_profiler.cu) is a standalone
CUDA probe for planning local collective schedules. It reports each selected
GPU's PCI ID, sysfs path, NUMA node, current and maximum PCIe link speed/width,
and a CUDA peer-access matrix. Current PCIe speed can reflect an idle power
state; maximum link speed describes capability, not measured throughput.

The probe then runs all selected GPUs concurrently in each direction using
separate CUDA streams and pinned host buffers. It reports aggregate H2D and D2H
useful-byte bandwidth from a common host-clock interval. It measures each
directed GPU peer link separately, then launches a concurrent source-side
peer-copy ring in the specified order. It also pins CPU
workers to each NUMA node and measures concurrent memory copy and four-source
BF16 reduction. CPU buffers are allocated and first-touched on the pinned
workers. The reduction reads four BF16 arrays, accumulates in FP32, and writes
one rounded BF16 array. It is a hardware probe, not the CPU-router executor or
a prediction of collective latency.

```bash
make hardware_profiler
./build/uccl_lite_hardware_profiler
./build/uccl_lite_hardware_profiler --devices 0,1,2,3 \
  --peer-order 0,1,3,2 \
  --gpu-bytes 16777216 --cpu-bytes 8388608 \
  --iterations 10 --warmups 2 --repeats 3 --threads-per-node 4
```

`--gpu-bytes` is the buffer size per GPU and `--cpu-bytes` is one CPU output
buffer per worker. H2D and D2H are separate concurrent sweeps. Directed P2P
copies are measured one at a time; the ring has one source and one destination
per GPU concurrently. Peer copies use enabled CUDA peer access and separate
destination buffers. The probe does not measure simultaneous bidirectional
host traffic or arbitrary combinations of peer transfers.
It uses CUDA's pinned allocation for host copy buffers, whose NUMA placement is
not forced. CPU copy and reduction use first-touch node placement. CPU copy
counts one read plus one write; BF16 reduction counts four reads plus one write.
These are useful bytes per elapsed host time, not memory-controller counters.
Allocation, warmup, and checks are excluded. The common interval includes
submission and synchronization. Each row gives the minimum, median, and maximum
of complete repetitions. No per-size winner selection is performed.

## Recorded four-L4 result

On November 10, 2026, the command above ran on four NVIDIA L4s, PCIe Gen4
x16, two GPUs per NUMA node, dual Intel Xeon Silver 4410Y, CUDA runtime 12.9,
driver 595.58.03. It used 16 MiB per GPU, 8 MiB per CPU worker, four workers
per node, two warmups, ten timed iterations, three repetitions per run, and
three independent complete runs. There is no comparison baseline. The
[validated summary](../benchmarks/data/hardware_profile_4l4_20261110_summary.csv)
reports the median of the three run medians and the full min–max range. The
[raw runs 1](../benchmarks/data/hardware_profile_4l4_20261110_run1.txt),
[2](../benchmarks/data/hardware_profile_4l4_20261110_run2.txt), and
[3](../benchmarks/data/hardware_profile_4l4_20261110_run3.txt) include PCI
paths, peer matrices, every directed link, and per-run ranges.

| Concurrent workload | Median useful bandwidth |
| --- | ---: |
| Four-GPU H2D | 99.60 GB/s aggregate |
| Four-GPU D2H | 55.42 GB/s aggregate |
| Four-GPU peer ring `0→1→3→2→0` | 82.04 GB/s aggregate |
| Directed P2P links, separately | 22.63–22.64 GB/s each |
| CPU copy, NUMA node 0 / 1 | 69.50 / 48.72 GB/s |
| Four-source BF16 CPU reduction, NUMA node 0 / 1 | 47.56 / 48.98 GB/s |

The CPU copy ranges include slower first repetitions and a lower node-1 rate
than an earlier single run. These numbers characterize these runs, buffer
placement and thread count; they do not prove a router or
collective crossover. Use the paired collective calibration in
[CPU-router selection](cpu-router.md) to decide whether to route a workload.

A later [1.5-GiB-per-NUMA STREAM-style triad](../benchmarks/data/host_memory_bw_4l4_20261111/README.md)
measures sustained host DRAM useful bandwidth at about 87 GB/s per socket
with 12 physical-core threads, or about 175 GB/s for both sockets running
simultaneously. It uses a different kernel and larger working set than the
copy and BF16-reduction rows above.

Regenerate the summary without GPU work using:

```bash
python3 scripts/summarize_hardware_profile.py \
  benchmarks/data/hardware_profile_4l4_20261110_summary.csv \
  benchmarks/data/hardware_profile_4l4_20261110_run{1,2,3}.txt
```
