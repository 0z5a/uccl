# Four-GPU peer AllGather and ReduceScatter

[peer_ag_rs.cu](../src/collectives/peer_ag_rs.cu) is a standalone float32
implementation and matched NCCL benchmark for the four-GPU PCIe host. It
requires CUDA peer access among all four devices and uses one process. It is
separate from the BF16 CPU router and the historical eight-GPU no-P2P result.

Each source GPU sends its AllGather shard to a distinct output region on every
destination. For ReduceScatter, each source sends the destination's input shard
to that destination's scratch. A destination kernel sums four received arrays
in FP32 and owns every output element. Cross-device CUDA events make each
destination wait for all source transfers. Before reusing ReduceScatter
scratch, sources wait for the preceding destination reduction. The source
tensors remain immutable until completion. The rotating enqueue order spreads
first transfers across peers; CUDA is free to overlap later phases.

The harness validates every output element against a position-sensitive
reference for three changed input generations, including one after timing.
A separate 1008-byte run checks the non-power-of-two tail. The supported byte
count is a multiple of 16, because the four ranks each own an integral number
of float32 elements. Both operations are out of place.

```bash
make peer_ag_rs NCCL_HOME=/path/to/nccl/build
./build/uccl_lite_peer_ag_rs -b 1024 -e 67108864 -w 5 -n 10 -r 3
./build/uccl_lite_peer_ag_rs -b 1008 -e 1008 -w 2 -n 10 -r 3
```

## Measured result

The [complete 17-size sweep](../benchmarks/data/peer_ag_rs_4l4_20261110.txt)
was run November 10, 2026 on four NVIDIA L4s, PCIe Gen4 x16, NUMA pairs
`(0,1)` and `(2,3)`, driver 595.58.03, CUDA 12.9 and NCCL 2.28.7. Each
backend uses the same process, buffers, streams, dtype, size, and timing
method. The order alternates across three repetitions. Each repetition has
five warmups and ten timed calls; the recorded latency is the slowest-rank
CUDA-event interval divided by ten. The table shows medians of repetition
values. Allocation, fill and correctness checking are outside timing. The
complete file also retains each backend's min/max run values. No per-size
variant selection is used.

| Full bytes per rank | AG peer / NCCL | AG speedup | RS peer / NCCL | RS speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 56.42 / 19.43 µs | 0.34× | 79.97 / 19.66 µs | 0.25× |
| 1 MiB | 77.33 / 83.34 µs | 1.08× | 108.95 / 76.49 µs | 0.70× |
| 8 MiB | 371.59 / 397.31 µs | 1.07× | 388.61 / 418.30 µs | 1.08× |
| 16 MiB | 706.57 / 768.90 µs | 1.09× | 729.29 / 820.74 µs | 1.13× |
| 32 MiB | 1383.78 / 1529.04 µs | 1.11× | 1433.29 / 1632.46 µs | 1.14× |
| 64 MiB | 2816.81 / 3037.29 µs | 1.08× | 3111.01 / 3065.24 µs | 0.99× |

AG has a lower median on 7/17 sizes, from 1 through 64 MiB, with
non-overlapping recorded timing ranges. RS has a lower median on 4/17 sizes;
its 4-MiB advantage is only 1.2%, while 8–32 MiB exceed 3% with
non-overlapping ranges. The small-message and 64-MiB RS losses remain in the
data. This benchmark does not install a runtime dispatch policy or establish
performance on another topology, dtype, in-place layout, or node count.
