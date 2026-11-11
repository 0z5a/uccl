# Single-node four-L4 FP32 AllGather with large-message host packets

This is the current main README AllGather figure on **`mibura-sky-test-02`**:
four L4 GPUs in a 2+2 NUMA layout, FP32 out-of-place AllGather, complete
output bytes per GPU. Its UCCL Lite curve retains the earlier direct-GPU
small/medium algorithms and switches to the same host-packet CPU-router
transport geometry measured for BF16 only after that path becomes the fastest
correct UCCL candidate. **Every plotted UCCL point is a UCCL implementation;
NCCL is never a fallback.** The AllReduce and ReduceScatter main README
figures retain their earlier implementations and values.

| Complete output/GPU | Selected UCCL Lite implementation |
| ---: | --- |
| 1 KiB–2 MiB | `push-serial`: each source GPU's single CTA visits four destinations in rotating order |
| 4–16 MiB | `peer`: source-side `cudaMemcpyPeerAsync` copies, as in the earlier FP32 path |
| 32 MiB–1 GiB | `packet-router`: each GPU uploads its shard once to NUMA-local mapped host memory; the four GPUs read other shards through host packets |

The optional `make peer-ag-rs-packet` target links the CPU-router module to
the existing peer AG/RS harness. For **AllGather Copy only**, it treats one
FP32 element as two 16-bit storage units in the existing packet plan; no
BF16 arithmetic or conversion occurs. The original FP32 checker validates
every output element on every invocation, including the two
non-power-of-two tail probes. The packet router uses eight configured CPU
workers per NUMA group, a 248-block transport upper bound (180 actual blocks
at 1 GiB), and a 4-MiB packet cap. `--packet-router-ag` enables this optional
candidate at 4 MiB and above; the default executable remains independent
of the CPU-router library. [Hardware/software versions and binary hashes](hardware_software.txt)
record the exact tested source and NCCL 2.28.7 library.

The [full raw paired sweep](full.log) runs all candidates and NCCL in the
**same harness on this one host**, with 20 warmups, three rotated repetitions
of ten timed calls, complete output checks each time, and slowest-GPU
CUDA-event latency. Input fill and checks are outside the timer. The
[per-call samples](samples.csv) retain all **2,070 timed calls**;
[all candidates](candidates.csv) retain negative results; and
[`selected.csv`](selected.csv) records the UCCL path, run range and paired
NCCL value at every one of 21 sizes. The path is chosen by the fastest
correct UCCL median at each size, without consulting NCCL. All rows match
FP32 dtype, complete output bytes, four devices, out-of-place placement,
host identity, and CUDA-event timing.

| Complete output/GPU | Selected UCCL Lite | UCCL (µs) | Paired NCCL (µs) |
| ---: | --- | ---: | ---: |
| 1 KiB | push-serial | 14.79 | 27.82 |
| 64 KiB | push-serial | 16.13 | 34.27 |
| 1 MiB | push-serial | 73.49 | 100.52 |
| 2 MiB | push-serial | 129.04 | 144.65 |
| 4 MiB | peer | 241.83 | 234.61 |
| 8 MiB | peer | 407.65 | 414.69 |
| 16 MiB | peer | 739.93 | 779.88 |
| 32 MiB | packet-router | 1,356.54 | 1,526.78 |
| 64 MiB | packet-router | 2,536.61 | 3,006.71 |
| 1 GiB | packet-router | **37,283.29** | **48,704.49** |

At 1 GiB, the paired old direct-peer candidate takes **45,548.45 µs**;
the new FP32 packet router takes **37,283.29 µs**, about **1.22× faster**
than that UCCL path and **1.31× faster** than paired NCCL. The packet
candidate loses to peer at 4/8/16 MiB (298.54/479.07/783.23 µs versus
241.83/407.65/739.93 µs), so the selected UCCL crossover is 32 MiB.
Those losses remain in `candidates.csv` rather than being hidden.

`pilot_1g.log`, `tail_1008.log`, `tail_large.log`, and
`default_peer_smoke.log` retain the initial full-output, scalar-tail,
large-tail, and unchanged-default-path checks. All exited successfully.
The [summary script](summarize.py) validates every size, backend, run and
call count, checks the reported summaries, and rebuilds the CSVs. From
`experimental/lite`:

```bash
make -j2 peer-collectives peer-ag-rs-packet
./build/uccl_lite_peer_ag_rs_packet --packet-router-ag \
  -b 1024 -e 1073741824 -w 20 -n 10 -r 3
python3 benchmarks/data/fp32_packet_ag_4l4_20261111/summarize.py
python3 scripts/plot_optimized_collectives.py
```
