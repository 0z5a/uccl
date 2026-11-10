# Eight-L4 CPU-router fusion experiment (2026-11-10)

This host has GPUs 0–3 on NUMA node 0 and GPUs 4–7 on node 1. CUDA peer
access is unavailable between the groups. The benchmark uses one process to
drive all eight GPUs and the project CPU router to cross the groups.

## Workloads and controls

- `gemm-ar`: each rank computes its own `A[M,K] × B[K,N]` partial, then
  AllReduce produces `C[M,N]` on every GPU.
- `gemm-rs`: each rank computes the same shape of partial; ReduceScatter
  produces one contiguous `C[M,N]/8` shard per GPU.
- `ag-gemm`: AllGather concatenates eight `A[M/8,K]` row shards, then each
  rank computes `A[M,K] × B[K,N]` with its local `B`.
- `cublas+nccl` is the cuBLAS GEMM plus NCCL 2.28.9 collective control.
- `cublas+router` uses the same cuBLAS GEMM and the CPU router in sequence.
- `fused+router` uses the router's resident CUTLASS BF16 GEMM producer and
  packet transport in one cooperative GPU grid. Its reduction runs on CPU
  workers. Comparisons with cuBLAS also include a change of GEMM kernel, so
  the measured gain cannot be assigned solely to overlap.

Inputs vary by rank, row, and column with exactly representable BF16 values;
this also checks ReduceScatter shard placement. Every output element
on every GPU is checked after every invocation. Each timing is the largest
CUDA-event duration among the eight ranks; validation is outside the timing
interval. Each of two independent process runs uses two warmup rounds and
five measured rounds of ten calls per method. Method order rotates each
round. The figures use the median of the two run-level medians. All 3,900
timed calls are in [run 1](samples_run1.csv) and [run 2](samples_run2.csv).

The `(128,1024,1024)` GEMM + AllReduce shape shows a repeatable fused gain;
the smaller shape is nearly tied in one run and larger shapes regress. For
AllGather + GEMM, replacing NCCL with the CPU router helps the
sequential path, while the resident fused path does not win over that path.
The fused GEMM + ReduceScatter path wins at several shapes; the largest
shape is close to the sequential router path.
The figures retain both findings rather than selecting only favorable shapes.

## Reproduce

From `experimental/lite`, with the local NCCL shim described in the
[eight-GPU router record](../cpu_router_8l4_20261110/README.md):

```bash
make -j2 build/uccl_lite_router_fusion_8l4 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
export LD_LIBRARY_PATH="$PWD/build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
export NCCL_NET=Socket
./build/uccl_lite_router_fusion_8l4 --op gemm-ar \
  --m 128 --n 1024 --k 1024 --warmup 2 --iterations 10 --repeats 5 \
  --samples build/fusion_samples.csv
./build/uccl_lite_router_fusion_8l4 --op ag-gemm \
  --m 128 --n 1024 --k 1024 --warmup 2 --iterations 10 --repeats 5 \
  --samples build/fusion_samples.csv
./build/uccl_lite_router_fusion_8l4 --op gemm-rs \
  --m 128 --n 2048 --k 1024 --warmup 2 --iterations 10 --repeats 5 \
  --samples build/fusion_samples.csv
```

The full shape list and plotting aggregation are in
[plot_router_fusion_8l4.py](../../../scripts/plot_router_fusion_8l4.py).
