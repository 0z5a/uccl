# Cooperative flush validation and speed comparison

**Version:** all measured tables below describe the elected-thread implementation
at `f77da5123a8e75fb4e9ad5525370c66c140c8056`. The queue-partition candidate in
the current branch has not yet been compiled or run; those earlier results are
not validation of the new implementation.

Validated on 2026-10-04 against the original adapter at
`29e7e7ca868590fb3a70bc96ebf42271983ab9b6`, stacked on the local SM120 validation
change. Hardware: two RTX 5090s, physical GPUs 0 and 3; CUDA 13.0.88,
driver 580.76.05, NCCL device headers 2.30.4. Neither P2P nor a NCCL communicator
is used by the local fixture. The container denies NUMA policy; the production
FIFO's existing fallback continues.

## Queue-partition iteration

The elected thread waits for each queue in sequence. The new candidate keeps
both group rendezvous but assigns queue indices `r + k * coop.size()` to member
`r`. Thus every queue receives one QUIET, all assigned queues finish before the
exit rendezvous, and thread groups retain sequential all-queue completion.
This preserves the group source-lifetime requirement in the
[NCCL GIN flush contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#ncclGin::flush).
Scalar `flush()` and the adapter's acquire-order guard are unchanged.

Q=3/33/64 extend validation beyond even, single-warp assignments. A new
dual-GPU comparison will use the original adapter, the first cooperative
implementation, and the queue-partition candidate with identical fixture
source and settings. Forward/reverse execution order and seven rounds per
process leave ten warm samples per arm/device after discarding the first two.
The real HBM source-copy and pattern checks remain enabled.

| New candidate gate | Current result |
| --- | --- |
| SM120/SM90 and full microbench compilation | Pending coordinated compile window |
| Actual adapter/standalone, thread/warp, Q=1/3/4/32/33/64 | Not run |
| Private/shared groups and delayed completion | Not run |
| Two-GPU G64/BS2048 comparison against original and first implementation | Not run |
| Large-payload CUDA timeline on GPU 3 | Not run |

The existing RLT and VIME resource windows precede this new iteration. No new
speedup is claimed until the candidate completes its own runs.

## Command counts and correctness

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
