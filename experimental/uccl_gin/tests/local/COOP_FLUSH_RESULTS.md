# Cooperative flush validation and speed comparison

**Current measured candidate:** queue partition V2 at
`2abdfdce17ab5747e136701b4fdfd2d0bd5bff43` (production loop from `a3dcc2d`).
The new campaign compares Original, elected-thread V1 and V2 with the same
fixture and settings. The later historical sections retain the earlier V1
results, including GPU3's 0.91×; absolute times from different campaigns are
not mixed into one speed ratio.

Hardware: two RTX 5090s, physical GPUs 0 and 3; CUDA 13.0.88,
driver 580.76.05, NCCL device headers 2.30.4. No P2P, NCCL communicator,
RDMA transport or model is initialized by the local fixture. The production
FIFO's existing NUMA fallback is used because container mempolicy is denied.

## V2: queue partition, actual GPU validation

V1 waits for all queues sequentially on rank 0. V2 retains both collective
barriers and assigns queue indices `r + k * coop.size()` to member `r`.
Every queue receives one QUIET and every assigned queue finishes before the
exit rendezvous. Thread groups retain sequential all-queue completion.
Scalar flush, FIFO/proxy completion and the adapter's acquire guard are unchanged.
The [design and relationship figures](COOP_FLUSH_DESIGN.md) explain the
source-lifetime chain and the distinction between local and network completion.

All three arms use fixture SHA256
`7df01273853b6e8fdb49e8be47303e09eefcfdf9c5a2aa8b5d810531f7e501d6`.
Original is `29e7e7ca868590fb3a70bc96ebf42271983ab9b6` with the same signal-load
compilation prerequisite; V1 is `f77da5123a8e75fb4e9ad5525370c66c140c8056`.
Header, adapter and binary fingerprints are in the
[campaign manifest](results/v2-20261004/source-manifest.json) and
[build fingerprints](results/v2-20261004/build-hashes.json).

| V2 gate | Actual result |
| --- | --- |
| Three arms SM120; V2 SM90; full network microbench SM120/90 | Compile pass; SM90 not run; no network link/run |
| Unsupported CTA cooperation | Expected compilation rejection |
| Thread/warp, adapter/standalone, Q1/3/4/32/33/64 | Pass on both GPUs |
| Private/shared G4/G8, early/late delayed ACK, proxy-major N1/N3/N4 | Pass; complete source and per-FIFO route/count checks |
| Correctness processes | 44 positive processes ×2 rounds; 2 separate expected order traps |
| Two simultaneous GPUs, G64/BS2048, ABC/CBA | 12 processes /84 rounds, all natural exit 0 |
| GPU3 host-stage / Nsight diagnostic | 2 host-stage +2 profile runs and 2 stats commands, all natural exit 0 |

Raw [correctness rows](results/v2-20261004/correctness.jsonl),
[process statuses](results/v2-20261004/run.jsonl) and
[build statuses](results/v2-20261004/build.jsonl) are committed. The evidence
archive was collected and validated before releasing the coordinated GPU/I/O
window; no process was externally signaled.

## V2: high concurrency and large batch speed table

Each GPU runs 64 full warps /2,048 producers, Q32, BS2048, hidden2048,
256 KiB per WRITE, capacity4096, shared queues, lane hint, N1, two iterations.
Both GPU processes start together. The complete HBM-copy and source-overwrite
oracle remains enabled; delays and consumer timing instrumentation are disabled.
Each round checks 4,096 WRITEs /1 GiB D2H per GPU. Original emits 131,072 QUIETs;
V1 and V2 each emit 4,096. Two GPUs total 4,096 producers, not HTTP requests.

Order is Original→V1→V2 (ABC), then V2→V1→Original (CBA). Each process runs seven
rounds and discards its first two: two independent starts and ten repeated warm
observations per arm/GPU. Times below are median /p95 in ms. p95 is descriptive
at this sample size; GPU ratios are not averaged into aggregate throughput.

| GPU | Order | Starts / warm rounds per arm | Original median/p95 | V1 median/p95 | V2 median/p95 | Original/V2 | V1/V2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | pooled ABC/CBA | 2 / 10 | 1602.505 / 1877.001 | 1560.322 / 1716.312 | 1164.814 / 1195.990 | 1.38× | 1.34× |
| 0 | ABC | 1 / 5 | 1623.211 / 1941.367 | 1459.197 / 1733.449 | 1158.544 / 1190.299 | 1.40× | 1.26× |
| 0 | CBA | 1 / 5 | 1581.800 / 1711.961 | 1564.652 / 1567.730 | 1179.587 / 1193.745 | 1.34× | 1.33× |
| 3 | pooled ABC/CBA | 2 / 10 | 1784.976 / 1962.228 | 1700.753 / 2138.367 | 1214.552 / 1525.603 | 1.47× | 1.40× |
| 3 | ABC | 1 / 5 | 1651.180 / 1736.557 | 2134.326 / 2139.975 | 1341.056 / 1564.400 | 1.23× | 1.59× |
| 3 | CBA | 1 / 5 | 1866.218 / 1969.741 | 1532.273 / 1545.942 | 1116.261 / 1284.801 | 1.67× | 1.37× |

| GPU | Original min–max (ms) | V1 min–max (ms) | V2 min–max (ms) |
| --- | --- | --- | --- |
| 0 | 1529.596–1996.841 | 1435.736–1747.158 | 1091.722–1195.998 |
| 3 | 1577.442–1975.751 | 1523.544–2141.261 | 967.928–1595.437 |

V2 improves both execution orders on both GPUs. GPU3 Original/V2 is 1.23× in
ABC and 1.67× in CBA, pooled 1.47×; V1/V2 is 1.59× /1.37×, pooled 1.40×.
V1 itself changes from 0.77× versus Original in ABC to 1.22× in CBA, so its
earlier regression is retained and the variation is not dismissed. This establishes
a local V2 benefit under the tested common fixture, not the cause of every
historical regression or network/model performance.

The [raw seven-round logs](results/v2-20261004/evidence/),
[60 warm rows](results/v2-20261004/samples.jsonl), and
[summary](results/v2-20261004/summary.json) are available. Recompute the tables
with `python3 results/v2-20261004/summarize.py` from this directory.

## V2: separate large-payload diagnosis

The following are single GPU3 runs, one round each. They are separate from the
unprofiled dual-GPU speed campaign and from each other. All use G64/Q32/BS2048,
two iterations, 4,096 D2H copies and the full oracle.

| GPU3 host-stage run (ms) | V1 | V2 |
| --- | ---: | ---: |
| Elapsed | 2000.591 | 1501.581 |
| Callback total | 1984.495 | 1491.761 |
| Copy + event submit | 11.224 | 11.056 |
| Event synchronize | 1884.380 | 1390.556 |
| Full pattern check | 88.540 | 89.785 |
| Between callbacks | 5.934 | 0.777 |

The three WRITE sub-stages are nested within callback total; do not add them to
the total. Between-callback time combines pop/scan/poll/scheduling and omits
the first/last boundary; it is not GPU idle. The [diagnostic JSON](results/v2-20261004/diagnostics.json)
records every field.

| GPU3 separate CUDA profile | V1 | V2 |
| --- | ---: | ---: |
| Kernel, count / total ms | 1 / 2144.190 | 1 / 1320.006 |
| D2H, count / total ms | 4096 / 1550.555 | 4096 / 1077.048 |
| Event synchronize, count / total ms | 4097 / 2018.086 | 4097 / 1198.861 |

The three targeted reports are committed as `profile-*-stats_cuda_*.csv` in
[the evidence directory](results/v2-20261004/). Kernel time, D2H time and API
waiting overlap; their totals must not be added. Event synchronization includes
one initialization call. Each profile is one observation, not a speed benchmark.

The host-stage run's reduced event wait and the separate trace's reduced D2H
duration point toward GPU/copy/FIFO scheduling as the next diagnostic focus.
This is an inference; these aggregate reports do not isolate a causal mechanism
or establish EFA receiver ordering. Host pattern-check work remains about89 ms
in both runs. No oracle was weakened to obtain the benefit.

## Historical V1: command counts and correctness

For a full warp, the actual original adapter emits `32 * Q` QUIET commands;
the cooperative adapter emits `Q`. The identical signal acquire-load compilation
prerequisite is applied to both adapters. The baseline retains its original
scalar `gin.flush()` body. This command reduction is independent of timing.

| Group, 10 iterations, one group | Queues | Original QUIETs | Cooperative QUIETs | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Full warp | 1 | 320 | 10 | 32× |
| Full warp | 4 | 1,280 | 40 | 32× |
| Full warp | 32 | 10,240 | 320 | 32× |
| Thread | 32 | 320 | 320 | 1× |

Both GPUs pass the counts above with real HBM source reads and immediate source
reuse after flush. The host oracle copies each WRITE's source into pinned memory,
waits for the copy event, checks the producer/generation/word pattern, then
acknowledges it through the production FIFO. It rejects missing generations,
incorrect WRITE encoding, and early source overwrite.

| Gate | Coverage | Result |
| --- | --- | --- |
| Actual standalone and adapter APIs | Thread/full warp; Q=1/4/32 | Pass on both GPUs |
| Independent and shared queue arrays | 4/8 groups, Q=4, delayed final acknowledgement | Pass on both GPUs |
| Entry and exit rendezvous | Prior WRITE trace and system-scope return status before final acknowledgement | Pass with private queues on both GPUs |
| Larger synthetic BF16 activations | 32 groups/BS512 and 64 groups/BS2048, hidden width 2048, Q=32 | Pass on both GPUs |
| Concurrent GPU instances | 8 groups/device, Q=32, 10 iterations, 3 rounds | Both processes naturally exit 0 |
| Unsupported CTA group | Real `ncclCoopCta` instantiation | Intended static diagnostic on SM120 and SM90 |
| Unsupported memory order | Relaxed order in a separate process | Device trap observed on both GPUs (CUDA 719) |
| Oracle negative control | Private comparison with flush omitted | Naturally exits 1: source reused before completion |
| Local cooperative fixture and network device gate | Native SM120 and SM90 | Compile pass |
| Entire `microbench.cu` translation unit | EFA flag, real MPI/verbs/NCCL headers, SM120/SM90 | Compile pass; no link or network execution |
| Transport and context compilation | All seven C++ units; isolated rdma-core 64.0 headers | Compile pass; full link remains pending |

The local positive suite completed 82 rounds across 36 sequential processes and
two concurrent instances, plus two isolated expected-trap processes. Test
processes finished without external termination. SM90 binaries were compiled,
not run on SM120.

## Paired wall-clock comparison

The coordinated heavy-I/O window was held for all 48 unprofiled A/B processes.
GPU 0 and GPU 3 were reserved; GPU 1 and GPU 2 remained busy with other tenants
in the before/after snapshots. The lock controls cooperating tasks and does not
make the entire shared host idle.

Both builds use the identical fixture source (SHA256
`68dadfd71567daaff012c7897c4363b273fcb5936e4f4cf2c10649b25333426f`),
compiler settings and queue capacity 4,096. The original adapter has only the
identical signal-load compilation prerequisite; its scalar flush body remains
unchanged. `LOCAL_EXPECT_SCALAR_ADAPTER=1` selects original-count expectations
and omits unsupported standalone instantiations in the baseline. Both arms time
the actual adapter. Each case alternates baseline/candidate as ABBA. Each process
runs seven rounds; the first two are warmup, leaving ten samples per
variant/device/case.
The measured interval covers kernel launch through FIFO completion and consumer
join. Initialization and buffer allocation are outside it.

Cells contain **median / p95 [min–max] in milliseconds**. P95 uses inclusive
linear interpolation. Speedup is baseline median / cooperative median; values
below 1 mean the cooperative version is slower in this fixture.

### Control-only flush

One full warp, 100 iterations, zero-byte payload, source-copy oracle disabled:

| Physical GPU | Queues | Baseline, ms | Cooperative, ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.722 / 0.830 [0.441–0.838] | 0.492 / 0.514 [0.449–0.520] | 1.47× |
| 0 | 4 | 2.558 / 2.919 [2.374–3.076] | 1.288 / 1.351 [1.280–1.354] | 1.99× |
| 0 | 32 | 22.114 / 24.282 [21.260–24.521] | 8.723 / 9.232 [8.677–9.591] | 2.54× |
| 3 | 1 | 0.780 / 0.901 [0.747–0.922] | 0.514 / 0.528 [0.483–0.533] | 1.52× |
| 3 | 4 | 3.047 / 3.445 [2.969–3.706] | 1.426 / 1.467 [1.395–1.468] | 2.14× |
| 3 | 32 | 24.377 / 25.086 [23.241–25.236] | 9.792 / 10.279 [9.741–10.307] | 2.49× |

### Synthetic BF16 payloads

Five iterations with the real HBM source-reuse oracle. Each group uses a
synthetic `BS * 2048 * 2` byte BF16 activation split among 32 producers.
BS2048/64 groups exercises 2,048 producers, 256 KiB each, with a 1 GiB
source/destination allocation per GPU. The width follows
[Qwen3-30B-A3B's configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json).
These fixtures load no model weights and do not measure serving requests or
model inference. Payload timings include HBM-to-host oracle copies and checks.

| Physical GPU | Warp groups | BS | Baseline, ms | Cooperative, ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 128 | 19.953 / 23.043 [19.427–24.121] | 21.219 / 23.534 [19.698–23.571] | 0.94× |
| 0 | 32 | 512 | 224.347 / 261.223 [209.126–267.875] | 297.462 / 306.651 [281.344–307.863] | 0.75× |
| 0 | 64 | 2048 | 3315.384 / 3472.658 [3059.501–3477.019] | 4146.157 / 4324.599 [3920.196–4327.141] | 0.80× |
| 3 | 8 | 128 | 22.562 / 25.107 [20.795–26.505] | 21.024 / 22.851 [20.757–22.957] | 1.07× |
| 3 | 32 | 512 | 269.774 / 312.056 [241.925–316.715] | 378.773 / 432.682 [333.526–434.959] | 0.71× |
| 3 | 64 | 2048 | 4394.541 / 4757.287 [3853.849–4820.229] | 4363.989 / 4408.055 [4313.156–4425.066] | 1.01× |

Control-only flush improves 1.47–2.54×. Payload results are mixed: BS512 regresses
on both GPUs; BS2048 regresses on GPU 0 and is approximately unchanged on GPU 3.
The local payload fixture therefore does not establish a general throughput
improvement. The command-count and group-completion guarantees still pass.

### Two-GPU high-concurrency payloads

Both GPU instances run simultaneously: **64 warp groups and 2,048 producers per
GPU, 4,096 producers total, BS2048, Q=32**, with 2 GiB of combined source/destination
allocation. The same source-reuse oracle is enabled. Baseline/candidate ABBA
completes eight processes and 24 positive rounds, with both exit statuses 0 for
every simultaneous pair. Each process runs two iterations and three rounds;
discarding its first round leaves four samples per variant/device. Compare
within these rows; the preceding payload table uses five iterations.

| Physical GPU | Baseline, ms | Cooperative, ms | Speedup |
| --- | ---: | ---: | ---: |
| 0 | 1643.863 / 1811.548 [1538.166–1830.749] | 1543.112 / 1624.250 [1406.099–1627.160] | 1.07× |
| 3 | 1851.091 / 2009.429 [1624.645–2037.230] | 2024.331 / 2061.415 [1972.746–2063.462] | 0.91× |

The concurrent fixture has no consistent speedup across the two GPUs. This is
GPU communication concurrency, not HTTP request concurrency or model serving.
All GPU jobs naturally exited, and GPU/I/O locks were released before handoff
to the next task.

## Separate CUDA timeline diagnostics

Following NVIDIA's
[Nsight Systems skill](https://github.com/NVIDIA/TensorRT-LLM/blob/fc0876cfd6c5d661f186707857a4e5bfeb12bc6f/.claude/skills/perf-nsight-systems/SKILL.md),
Nsight Systems 2025.3.1 captured one small run per build after the unprofiled
comparison. GPU 0, G=8, BS128, Q=32, two iterations, one round. Collection used
`--trace=cuda --sample=none --cpuctxsw=none --kill=none`. Both targets naturally
exit 0; `cuda_gpu_kern_sum`, `cuda_api_sum` and `cuda_gpu_mem_time_sum` all process
successfully. The following values come directly from their CSV reports:

| Diagnostic metric | Count per build | Baseline, ms | Cooperative, ms |
| --- | ---: | ---: | ---: |
| Producer kernel | 1 | 13.234 | 13.427 |
| Device-to-host source copies | 512 | 1.942 | 1.894 |
| `cudaEventSynchronize` host API time | 513, including warmup | 6.576 | 7.170 |

The trace confirms the real source-copy workload remains present in both builds.
API times and device times overlap and must not be added. These are single
diagnostic captures, excluded from the speed tables, and do not isolate the
cause of the larger-BS regressions.

## Network acceptance

`--only coop-flush` is implemented in the existing EFA microbench: eight complete
warps publish disjoint payloads, flush through the actual adapter, and overwrite
their sources. Independent receiver slots and the received pattern must pass.
The invocation is in [README.md](README.md#efa-gate).

Network execution, EFA source consumption, receiver visibility, and full
NCCL-EP/model dispatch/combine e2e remain **not run**: this container exposes no
RDMA device. The new gate's compile success does not establish those properties.
The wider model integration's window/signal contracts remain a separate scope.
