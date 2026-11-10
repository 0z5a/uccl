<div align="center">

# UCCL Lite

**Collectives and communication–GEMM fusion for PCIe GPU nodes.**

[Get started](#get-started) · [Results](#measured-results) · [CPU router](#optional-cpu-router) · [Documentation](#documentation)

</div>

UCCL Lite is an experimental collection of CUDA collective kernels, an
optional CPU router, fusion kernels, and hardware probes. The direct peer paths
target **four NVIDIA L4 GPUs on a PCIe Gen4 dual-socket node**; the network
paths target **two such nodes**, with two NIC rails per node. The CPU router is
a separate BF16 module for 1–32 local GPUs. Most collective and fusion paths
use MSCCL++; standalone peer AG/RS uses CUDA peer copies.

The measurements below describe those testbeds and workloads. Small messages
and small GEMMs can lose to NCCL; the full sweeps and negative results are linked
alongside the figures.

## What is included

| Component | Purpose |
| --- | --- |
| [Hardware profiler](docs/hardware-profiler.md) | Discover PCI/NUMA/peer topology; measure directed and concurrent GPU transfers, CPU copy, and BF16 reduction bandwidth. |
| [Single-node collectives](docs/collectives.md) | Float32 AllReduce and All-to-All schedules, plus [direct-peer AllGather and ReduceScatter](docs/peer-ag-rs.md). |
| [Two-node collectives](docs/multinode.md) | Eight-rank AllReduce, AllGather, and ReduceScatter over local CUDA IPC and two GPUDirect rails. |
| [Optional CPU router](docs/cpu-router.md) | BF16 AR/AG/RS over NUMA-local host memory, selected only with matching local benefit evidence. |
| [Fusion kernels](docs/fusion.md) | AG+GEMM, GEMM+RS, GEMM+AR, and expert-parallel MoE pipelines. |

The four-GPU topology is `(GPU 0, GPU 1)` on NUMA node 0 and `(GPU 2, GPU 3)`
on NUMA node 1. The large-message ring follows **0 → 1 → 3 → 2 → 0**. Moving
to another GPU order or interconnect needs a new topology check and crossover
measurements.

## Get started

From the UCCL repository root, with CUDA and MSCCL++ installed:

```bash
cd experimental/lite
make -j2 single-node MSCCLPP_HOME=/path/to/mscclpp
./build/uccl_lite_allreduce -b 1024 -e 1048576 -w 5 -n 10
./build/uccl_lite_alltoall -b 1024 -e 1048576 -w 5 -n 10

# CUDA-only topology and bandwidth probe
make hardware_profiler
./build/uccl_lite_hardware_profiler --peer-order 0,1,3,2
```

The collective executables start four local ranks and check their output by
default. Verify the device order with `nvidia-smi topo -m` before running them.
[Build details](docs/build.md) cover NCCL, CUTLASS, the optional CPU router,
and the two-node launcher.

## Measured results

Collective figures use **latency for small messages** and **bandwidth for large
messages**. Fusion figures show complete-operation time or token throughput.

### Single node: collective schedules

![AllReduce small-message latency and large-message bus bandwidth on four L4 GPUs](docs/assets/allreduce.svg)

On four L4s, the float32 AllReduce sweep has lower median time at **19 of 21**
sizes than both tested NCCL versions. At 1 KiB it takes **6.08 µs**, versus
**17.29/19.54 µs** for NCCL 2.18.3/2.28.7. At 1 GiB it reaches
**20.13 GB/s bus bandwidth**, versus **17.01/18.49 GB/s**. This historical
summary selects the fastest correct UCCL Lite variant per size; some mid-size
differences are ties. [Full sweep and method](benchmarks/README.md).

![All-to-All small-message latency and large-message aggregate bandwidth](docs/assets/alltoall.svg)

At 1 KiB, scheduled All-to-All takes **5.21 µs** versus **21.9 µs** for the
recorded default NCCL transport. At 256 MiB/rank it reaches **74.47 GB/s
aggregate** versus **55.79 GB/s**. Aggregate traffic differs from AllReduce
bus bandwidth. For direct-peer AG/RS at 32 MiB, measured bus bandwidth is
**18.19 / 17.56 GB/s** versus **16.46 / 15.42 GB/s** for NCCL 2.28.7;
both paths regress on small messages. [AG/RS sweep and ownership
rules](docs/peer-ag-rs.md).

### Two nodes: large-message collectives

![Two-node AllGather, ReduceScatter, and AllReduce bus bandwidth at 64 MiB and 1 GiB](docs/assets/multinode-peer.svg)

At 1 GiB on two four-L4 nodes, **AllGather / ReduceScatter / AllReduce** reach
**18.23 / 17.62 / 17.79 GB/s bus bandwidth**, versus **16.02 / 15.36 /
16.31 GB/s** for NCCL 2.28.7. These are three paired runs with slowest-rank
CUDA-event timing; AG/RS are out of place and AR is in place. Small messages
can lose: at 1 KiB, AG takes **97.66 µs** versus **19.36 µs** for NCCL.
The [two-node results](docs/multinode.md) include small-message latency,
timing ranges, and the complete sweeps.

### Communication–compute fusion

![Complete-operation latency for three dense fusion kernels at their large reference shapes](docs/assets/dense-fusion.svg)

At the large FP16 reference shapes, fusion changes complete-operation latency
from **22.87 → 15.58 ms** for AG+GEMM, **22.95 → 18.46 ms** for GEMM+RS, and
**25.66 → 19.77 ms** for GEMM+AR, compared with cuBLAS + NCCL. The
[labeled eight-shape plot](docs/fusion.md) retains small-shape regressions;
Flux and no-overlap controls remain in the [measurement record](benchmarks/README.md).

![Expert-parallel MoE full-layer small-batch latency and large-batch token throughput](docs/assets/moe-fusion.svg)

For DeepSeek-shaped BF16 expert parallelism, the full layer at 32k tokens/rank
takes **85.28 ms** versus **129.77 ms** for NCCL + grouped cuBLAS. That is
**384k versus 253k tokens/s/rank** in this kernel benchmark. At eight tokens,
fusion is slower: **2.46 versus 1.37 ms**. The full layer includes dispatch,
both projections, SwiGLU, weighted reduction, and combine; routing-plan
construction and framework work are outside timing. [Workload and controls](docs/fusion.md).

## Optional CPU router

The CPU router is a separately built library. A caller supplies its existing
backend as a callback. Automatic selection requires an **exact local** match
for collective, BF16 dtype, byte count, placement, topology/software identity,
and fallback identity. A candidate needs at least **3% median speedup with
non-overlapping timing ranges**; otherwise the callback runs without creating
router workers or buffers.

```bash
make -j2 cpu-router adaptive-collectives
./build/uccl_lite_collectives --gpus 4 --backend auto --op all
```

On the current four-L4 host, all **12** paired calibration cases retained
NCCL. Historical data from a different **eight-L4, no-P2P** system recorded
1-GiB in-place BF16 AllGather/ReduceScatter bandwidth of **5.25/5.59 GB/s**
versus **5.04/4.86 GB/s** for NCCL 2.27.5. That archive cannot enable the
router automatically on a new host; this port has not been rerun on eight GPUs.
[API and calibration](docs/cpu-router.md) · [Historical primitive figures and
provenance](docs/primitive-collectives.md).

## Documentation

- [Build targets and correctness commands](docs/build.md)
- [Hardware profiler methods and measurements](docs/hardware-profiler.md)
- [Collective algorithms](docs/collectives.md) and [two-node design](docs/multinode.md)
- [Fusion design and workloads](docs/fusion.md)
- [Full measurement files](benchmarks/README.md) and [validation limits](docs/validation.md)

UCCL Lite provides specialized research kernels and benchmarks, not an
NCCL-compatible API or a serving-framework integration. The project follows
the repository [license](../../LICENSE); the adapted CUTLASS wrapper retains
its [upstream notice](NOTICE.md).
