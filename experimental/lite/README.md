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

The recorded gains below describe those testbeds and workloads. Small messages
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

### Single node: collective schedules

![AllReduce speedup over two NCCL versions from 1 KiB to 1 GiB](docs/assets/allreduce.svg)

On four L4s, the float32 AllReduce sweep wins at **19 of 21** sizes against
both tested NCCL versions. At 1 KiB it takes **6.08 µs**, versus
**17.29/19.54 µs** for NCCL 2.18.3/2.28.7. At 1 GiB it reaches
**20.13 GB/s bus bandwidth**, versus **17.01/18.49 GB/s**. This historical
summary selects the fastest correct UCCL Lite variant per size; some mid-size
differences are ties. [Full sweep and method](benchmarks/README.md).

![All-to-All speedup over default NCCL transport](docs/assets/alltoall.svg)

The scheduled All-to-All path reaches **74.47 GB/s aggregate** at
256 MiB/rank, with **1.17–4.20×** speedup over the recorded default NCCL
transport. This aggregate traffic metric differs from AllReduce bus bandwidth.
The four-GPU direct-peer AG/RS path reaches **1.11× / 1.14×** over NCCL 2.28.7
at 32 MiB; both paths regress on small messages. [AG/RS sweep and ownership
rules](docs/peer-ag-rs.md).

### Two nodes: large-message collectives

![Two-node AllGather, ReduceScatter, and AllReduce speedup over NCCL at 32 MiB to 1 GiB](docs/assets/multinode-peer.svg)

At 1 GiB on two four-L4 nodes, **AllGather / ReduceScatter / AllReduce** reach
**1.14× / 1.15× / 1.09×** over NCCL 2.28.7. These are three paired runs with
slowest-rank CUDA-event timing; AG/RS are out of place and AR is in place.
The figure shows 32 MiB–1 GiB, where the direct ring paths are useful.
Small-message regressions, timing ranges, and the complete 1 KiB–1 GiB sweeps
remain in the [two-node results](docs/multinode.md).

### Communication–compute fusion

![Dense AG+GEMM, GEMM+RS, and GEMM+AR speedup across eight shapes](docs/assets/dense-fusion.svg)

At the large FP16 reference shapes, AG+GEMM, GEMM+RS, and GEMM+AR reach
**1.47× / 1.24× / 1.30×** over cuBLAS + NCCL. Each row shows all eight
recorded shapes; the large dot marks the reference shape and orange marks
regressions. The [fusion notes](docs/fusion.md) include a labeled shape plot,
Flux and no-overlap controls, and timing details; the underlying values are
in the [measurement record](benchmarks/README.md).

![Expert-parallel MoE dispatch and full-layer fusion speedup](docs/assets/moe-fusion.svg)

For DeepSeek-shaped BF16 expert parallelism, fusion reaches **1.48–1.61×**
for dispatch + GEMM and **1.39–1.52×** for the full layer at large token counts.
The full layer includes dispatch, both projections, SwiGLU, weighted reduction,
and combine. Routing-plan construction and framework work are outside timing;
small-token losses are visible in the figure. [Workload and controls](docs/fusion.md).

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
