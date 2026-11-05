# Two-node AllReduce

All network benchmarks use two nodes with four L4 GPUs each. Local ranks 0/1
map to `mlx5_0` and ranks 2/3 to `mlx5_1`. Reported latency is the slowest of
eight ranks; bus bandwidth uses the eight-rank factor `2*(8-1)/8 = 1.75`.

## Hierarchical GPUDirect

[multinode_allreduce.cu](../src/collectives/multinode_allreduce.cu) performs:

```text
local ring ReduceScatter
    → exchange reduced quarters with corresponding ranks on the other node
    → local ring AllGather
```

The local ring is `0 → 1 → 3 → 2 → 0`. All four GPUs inject traffic across two
rails. Independent local and IB progress paths overlap piecewise network traffic
with local communication. Immutable IB slots avoid overwriting a source while
the NIC still reads it.

The reference sweep searches 1/2/4/8 MiB pieces (`-k`) and pipeline distances
1/2/4 (`-d`). The per-size best median beats the faster tested NCCL version from
32 MiB upward, reaching 18.06 GB/s at 64 MiB and 17.80 GB/s at 1 GiB. Small
messages lose because dependent proxy phases impose an approximately 80 µs floor.

## Pinned-host CPU router

[multinode_cpu_router.cu](../src/collectives/multinode_cpu_router.cu) retains
the local GPU ring but routes inter-node shards through pinned host buffers and
MPI/UCX. The default has no GPU RDMA registration. NUMA-local rank placement and
1 MiB double-buffered pieces achieve 15.93 GB/s at 1 GiB.

The source also retains explicitly selected experiments:

| Option | Behavior | Recorded outcome |
| --- | --- | --- |
| `-r cpu` | Reduce received host partials on CPU before H2D | About 1.1% slower than GPU reduction |
| `-m fused --direct-recv 1` | Final local reduction writes GPU and mapped host output; NIC writes remote GPU | 16.75 GB/s at 1 GiB; requires receive-side GDR |
| Add `--inband 1` | Ordered RDMA host doorbell replaces per-piece MPI handshake | 16.73 GB/s; effectively tied |

The fused transport still completes the entire local ReduceScatter before
posting network pieces. Removing handshakes cannot recover overlap that the
producer schedule does not expose. A device-triggered mapped-host PortChannel
variant encountered IB remote-access errors; it is not the validated fast path.

## TCCL-style global ring

[tccl_ring_allreduce.cu](../src/collectives/tccl_ring_allreduce.cu) reproduces
the topology-selected schedule directly on MSCCL++:

```text
N0.G3 → N0.G2 → N0.G0 → N0.G1 → N1.G0 → N1.G1 → N1.G3 → N1.G2 → N0.G3
```

Each channel performs seven ReduceScatter and seven AllGather steps. Local edges
use CUDA IPC. In host-staged mode (`-H 1`), GPU SMs write mapped NUMA-local host
slots, MSCCL++ sends them over IB, and the receiving NIC writes GPU memory.
Multiple slots preserve in-flight sources and allow several steps to progress.

| 1 GiB configuration | Bus bandwidth |
| --- | ---: |
| TCCL reference, four channels | 19.64 GB/s |
| UCCL Lite host staged, eight channels / eight slots | 19.21 GB/s |
| UCCL Lite host staged, four channels / eight slots | 18.30 GB/s |
| UCCL Lite host staged, four channels / four slots | 17.13 GB/s |
| UCCL Lite GPU-memory GDR control, four channels | 14.16 GB/s |

The best three runs are 19.08, 19.21, and 19.27 GB/s. All recorded checks pass
with zero float32 relative error. The result remains 2.2% below TCCL. It shows
why matching a logical ring without matching its transport is insufficient.

The mapped-host registration patch is required for the tested MSCCL++ revision:
otherwise a mapped host pointer is misclassified as GPU memory and registered
through DMA-BUF. The patch changes only external MSCCL++ pointer classification;
it is distributed here but not applied automatically.

## No GPU P2P and no GPUDirect

[host_allreduce.cu](../src/collectives/host_allreduce.cu) is the strict fallback:
each piece uses synchronous D2H, CPU MPI_Allreduce, and synchronous H2D. It
registers no GPU memory with the NIC and makes no CUDA-IPC peer transfer.

[host_hierarchical_allreduce.cu](../src/collectives/host_hierarchical_allreduce.cu)
instead performs three local ReduceScatter steps through host buffers, exchanges
one reduced quarter between nodes, and performs three local AllGather steps.
Arithmetic remains on the GPU, but the reliable path serializes all staging.

At 1 GiB, the strict and hierarchical fallbacks reach 5.06 and 4.96 GB/s,
respectively. Under the same no-P2P/no-GDR restrictions, TCCL CPU↔CPU reaches
14.37 GB/s and NCCL 2.28.7 reaches 14.23 GB/s. The bottleneck is serialized
staging, CPU work, and synchronization, not simply network bandwidth.

**Do not use `--persistent 1` as a correctness-qualified implementation.**
That opt-in mapped-host prototype intermittently loses or reuses complete rank
segments. It is retained for investigation and excluded from reported results.
Its shared-memory names also assume one job at a time per host. The default
synchronous path is the reliable comparison.
