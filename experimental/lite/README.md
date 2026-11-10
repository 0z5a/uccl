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

The current result refresh uses an **eight-L4 host with four GPUs per NUMA
node**. Four-L4, two-node, and fusion measurements will be refreshed on their
respective machines; their earlier records remain linked. Small messages and
small GEMMs can lose to NCCL; full sweeps and negative results are retained.

## What is included

| Component | Purpose |
| --- | --- |
| [Hardware profiler](docs/hardware-profiler.md) | Discover PCI/NUMA/peer topology; measure directed and concurrent GPU transfers, CPU copy, and BF16 reduction bandwidth. |
| [Single-node collectives](docs/collectives.md) | Float32 AllReduce and All-to-All schedules, plus [direct-peer AllGather and ReduceScatter](docs/peer-ag-rs.md). |
| [Eight-GPU local fallback](docs/collectives.md#eight-gpu-bf16-local-fallback) | Project-owned BF16 AR/AG/RS for two four-GPU NUMA groups, using peer copies within groups and pinned host staging across groups. |
| [Eight-GPU All-to-All](docs/collectives.md#eight-gpu-bf16-all-to-all) | BF16 host and peer/host paths for two four-GPU NUMA groups, with an explicit NCCL control. |
| [Two-node collectives](docs/multinode.md) | Eight-rank AllReduce, AllGather, and ReduceScatter over local CUDA IPC and two GPUDirect rails. |
| [Optional CPU router](docs/cpu-router.md) | BF16 AR/AG/RS over NUMA-local host memory, selected only with matching local benefit evidence. |
| [Fusion kernels](docs/fusion.md) | AG+GEMM, GEMM+RS, GEMM+AR, and expert-parallel MoE pipelines. |

The four-GPU peer kernels assume `(GPU 0, GPU 1)` on NUMA node 0 and
`(GPU 2, GPU 3)` on NUMA node 1, with a large-message ring
**0 → 1 → 3 → 2 → 0**. This eight-L4 machine instead has GPUs 0–3 and 4–7 on
the two NUMA nodes. Moving to another order or interconnect needs a topology
check and new crossover measurements.

## Get started

From the UCCL repository root on this eight-L4 machine:

```bash
cd experimental/lite
make -j2 adaptive-collectives eight-gpu-alltoall \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
./build/uccl_lite_collectives --gpus 8 --fallback nccl \
  --backend force --op allreduce --begin 1048576 --end 1048576 \
  --warmup 1 --iterations 1 --repeats 3
./build/uccl_lite_alltoall_8l4 --begin 1048576 --end 1048576 \
  --warmup 1 --iterations 1 --repeats 3
```

The eight-GPU benchmarks check every invocation. Verify the device order with
`nvidia-smi topo -m` before running them. The 2+2 four-GPU programs need their
own machine. [Build details](docs/build.md) cover NCCL, CUTLASS, MSCCL++, and
the two-node launcher.

## Measured results

The completed eight-L4 comparison shows **latency from 1 KiB to 1 MiB** and
**throughput from 1 MiB to 1 GiB** for each collective. The other topology and
fusion sections link their earlier records until the matching machines can be
rerun.

### Single-node collectives

#### Four L4 (earlier measurements; refresh pending)

AllReduce, AllGather, ReduceScatter, and All-to-All will be refreshed on a
four-L4 machine with the required 2+2 NUMA topology, using the best direct
peer UCCL Lite path against one matching NCCL control. No CPU router will be
used for that comparison. Earlier measurements remain in the
[single-node design](docs/collectives.md), [peer AG/RS record](docs/peer-ag-rs.md),
and [benchmark archive](benchmarks/README.md).

#### Eight L4 (this machine)

This BF16 comparison uses GPUs 0–3 on NUMA node 0 and 4–7 on node 1.
AllReduce is in place; AllGather and ReduceScatter are out of place. Each UCCL Lite curve
uses the fastest measured project-owned path at that size, chosen without
looking at NCCL. The CPU router is selected for all large AR/AG/RS sizes.
At 128 KiB, AR selects the local path and RS selects GPU reduction/progress;
RS uses the CPU-reduction router at other sizes.

![Eight-L4 AllReduce: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput versus NCCL](docs/assets/collectives-8l4-allreduce.svg)

![Eight-L4 AllGather: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput versus NCCL](docs/assets/collectives-8l4-allgather.svg)

![Eight-L4 ReduceScatter: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput versus NCCL](docs/assets/collectives-8l4-reducescatter.svg)

| Collective | UCCL Lite path | 1 MiB UCCL / NCCL | 64 MiB UCCL / NCCL | 1 GiB UCCL / NCCL |
| --- | --- | ---: | ---: | ---: |
| AllReduce | CPU router | 0.495 / 0.594 | 17.451 / 24.052 | 276.811 / 380.392 |
| AllGather | CPU router | 0.321 / 0.359 | 11.547 / 12.027 | 181.621 / 188.345 |
| ReduceScatter | CPU router | 0.340 / 0.353 | 10.761 / 13.942 | 172.255 / 220.849 |

The table gives **milliseconds of slowest-rank CUDA-event latency** at selected
large sizes. The plots use decimal GB/s: AR bus throughput is
`(14/8) × bytes/rank ÷ seconds`, and AG/RS bus throughput is
`(7/8) × bytes/rank ÷ seconds`. These metrics should only be compared within the
same collective. The full 1 KiB–1 GiB doubling sweep used BF16, CUDA 12.9,
driver 580.178.04, and NCCL 2.28.9. AR/AG use three alternating repetitions,
and the updated RS sweep uses five; each backend has five warmups and ten timed,
output-checked calls per repetition. Fill and verification were outside timing.
AR loses at three small sizes. RS at 64 KiB trails NCCL by 0.5 µs, with
overlapping five-run ranges. All losses are retained. The
[full selected-path summary](benchmarks/data/collectives_8l4_20261110.csv),
[small-message measurements](benchmarks/data/collectives_small_8l4_20261110/README.md),
[paired CPU-router/NCCL runs](benchmarks/data/router_vs_nccl_8l4_20261110/README.md),
and [updated ReduceScatter rerun](benchmarks/data/reducescatter_8l4_20261110_v2/README.md)
provide all sizes, timing ranges, and exact methods. The
[local/router calibration](benchmarks/data/local_fallback_8l4_20261110_v3/README.md)
supports the AR/AG path choice; the updated RS record has its own local/router
calibration. Neither is a third curve in this summary.

### Multi-node collectives

#### Two nodes × four L4 (earlier measurements; All-to-All pending)

AllReduce, AllGather, ReduceScatter, and All-to-All will be filled on the
two-node testbed. The earlier AR/AG/RS results, including the
[large-message latency bars](docs/assets/multinode-peer.svg), remain in the
[two-node design record](docs/multinode.md); two-node All-to-All has not yet
been measured.

### Fusion (earlier measurements; refresh pending)

Dense AG+GEMM, GEMM+RS, GEMM+AR, and expert-parallel MoE results will be
refreshed on their respective testbeds. The previous
[latency bar figures and workload definitions](docs/fusion.md), including
small-shape regressions, remain available until then.

## Optional CPU router

The CPU router is a separately built library. A caller supplies its existing
backend as a callback. The included eight-GPU example now supplies a **UCCL
Lite two-NUMA BF16 collective** as that backend; NCCL is an optional external
control. Automatic selection requires an **exact local** match for collective,
dtype, byte count, placement, topology/software identity, and fallback
identity. A candidate needs at least **3% median speedup with non-overlapping
timing ranges**; a fresh fallback-only dispatcher runs the local collective
without creating router workers.

```bash
make -j2 cpu-router local-collectives adaptive-collectives \
  NCCL_HOME=/path/to/nccl
./build/uccl_lite_collectives --gpus 8 --fallback local --op all \
  --in-place --begin 1048576 --end 1073741824 \
  --calibrate build/local-in-place.profile
./build/uccl_lite_collectives --gpus 8 --fallback local --op all \
  --in-place --begin 1048576 --end 1073741824 \
  --backend auto --profile build/local-in-place.profile
```

The eight-L4 table above reports the fastest **measured** UCCL Lite path.
Runtime `auto` is more conservative: with its matching local profile, it
retains local AllGather at 1 GiB because the CPU-router gain is below the 3%
selection margin. Without a matching profile it uses the local fallback.
The [local/router calibration and diagnostic figure](benchmarks/data/local_fallback_8l4_20261110_v3/README.md)
record the other UCCL Lite paths and negative results; the
[API and calibration guide](docs/cpu-router.md) explains runtime selection.

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
