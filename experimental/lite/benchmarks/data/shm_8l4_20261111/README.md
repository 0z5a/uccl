# 8xL4 SHM testbed (2026-11-11)

This testbed uses the same eight-L4 host as the 8xL4 GPU result: GPUs 0–3
are on NUMA node 0, and GPUs 4–7 are on NUMA node 1. The hardware still
supports 24 directed peer links within those groups. `--shm-only` disables
all peer links in UCCL-Lite's benchmark topology, rejects the P2P local
fallback, and routes every UCCL-Lite collective through mapped host memory.
It is a **logical no-P2P testbed**, not a change to the PCIe hardware.
The host runs driver 580.178.04 and CUDA 12.9.

NCCL 2.28.9 runs with `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`,
`NCCL_IB_DISABLE=1`, and `NCCL_NET=Socket`. The
[NCCL initialization probe](nccl_shm_probe.log) reports all 16 GPU channel
edges as `via SHM/direct/direct`. UCCL-Lite uses NUMA-local registered host
packets and does not issue GPU peer copies.
`--fallback nccl` supplies the external timing control; the forced
UCCL-Lite path never calls it.

## Algorithms and measurement

- AllReduce is in place; AllGather and ReduceScatter are out of place.
- At 128 KiB AllReduce, the router uses four CPU workers per NUMA node and
  16 transport blocks instead of the usual 23 workers and 64 blocks.
- At 64 and 128 KiB ReduceScatter, GPU progress handles packet handoff;
  the rest use the CPU-router placement appropriate to their operation.
- AllGather uses CUDA copy engines (`cudaMemcpyAsync`) to upload each GPU's
  shard to pinned host packets and gather remote shards back to each GPU.
  The local shard uses a device-local copy. CPU workers publish packet-ready
  signals after all eight uploads; they do not copy the AllGather payload.
  The DMA path uses 16 MiB packet storage and two alternating slots. At a
  1 GiB output size, each GPU contributes 128 MiB in 64 tiles of 2 MiB.
  The previous GPU-kernel path can be measured with `--allgather-kernel`.

Two process runs cover every doubling size from 1 KiB through 1 GiB. Run 1
uses ascending sizes and run 2 descending sizes. Each backend gets 20
warmups and three alternating repetitions of ten timed calls at every size.
Every invocation checks the complete output on all eight GPUs. Each timing
is the slowest-rank CUDA-event interval; filling, verification, allocation,
and communicator setup are outside timing. The six sample files contain
**7,560 checked timed calls**:

| Collective | Run 1 | Run 2 |
| --- | --- | --- |
| AllReduce | [samples](ar_run1.csv), [log](ar_run1.log) | [samples](ar_run2.csv), [log](ar_run2.log) |
| AllGather | [samples](ag_run1.csv), [log](ag_run1.log) | [samples](ag_run2.csv), [log](ag_run2.log) |
| ReduceScatter | [samples](rs_run1.csv), [log](rs_run1.log) | [samples](rs_run2.csv), [log](rs_run2.log) |

The [full summary](summary.csv) reports each run's median of repetition
medians and then the median of those two run values. It includes the selected
UCCL-Lite path at each size and every 1 KiB–1 GiB latency result.

| Operation and full tensor size | UCCL-Lite | NCCL SHM | Speedup |
| --- | ---: | ---: | ---: |
| AllReduce, 1 KiB | 101.1 µs | 132.1 µs | 1.31× |
| AllReduce, 128 KiB | 201.7 µs | 215.6 µs | 1.07× |
| AllReduce, 1 GiB | 279.4 ms | 403.7 ms | 1.44× |
| AllGather, 1 KiB | 448.0 µs | 120.6 µs | 0.27× |
| AllGather, 1 GiB | 181.8 ms | 195.1 ms | 1.07× |
| ReduceScatter, 64 KiB | 115.7 µs | 139.5 µs | 1.21× |
| ReduceScatter, 1 GiB | 172.1 ms | 218.0 ms | 1.27× |

All 21 AllReduce and 21 ReduceScatter aggregate medians favor UCCL-Lite.
Copy-engine AllGather beats NCCL SHM at 6 of 21 sizes, beginning at 32 MiB;
its 32 MiB gain is only 0.7% and should be treated as a tie. The 1 KiB
AllGather regression is roughly 3.7× against NCCL SHM and 4.5× against the
previous UCCL-Lite GPU-kernel path. The earlier kernel results remain in
[run 1 samples](ag_kernel_run1.csv), [run 2 samples](ag_kernel_run2.csv),
and their corresponding [logs](ag_kernel_run1.log) and [logs](ag_kernel_run2.log).
The full summary retains every DMA result, including regressions. The 64 KiB
AllReduce point is also effectively tied.

## Reproduce

From `experimental/lite`, with the local NCCL shim described in the
[eight-GPU router record](../cpu_router_8l4_20261110/README.md):

```bash
make -j2 adaptive-collectives CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH="$PWD/build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64" \
  NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
  ./build/uccl_lite_collectives --shm-only --gpus 8 --fallback nccl \
  --op allreduce --in-place --begin 1024 --end 1073741824 \
  --warmup 20 --iterations 10 --repeats 3 \
  --samples build/shm-ar-run1.csv --calibrate build/shm-ar-run1.profile
```

For AllGather and ReduceScatter, use `--op allgather` or
`--op reducescatter` and omit `--in-place`. AllGather selects DMA by default
in this SHM testbed; `--allgather-kernel` selects the archived GPU-kernel
implementation for comparison. Add `--reverse` for the second
run, with distinct sample/profile paths. `scripts/summarize_8l4_shm.py`
regenerates the summary from the archived six CSVs, and
`scripts/plot_8l4_shm.py` regenerates the figures. NCCL's
[environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
documents the transport controls; the initialization log verifies the
transport chosen on this host.
