# Cooperative flush validation and speed comparison

Validated on 2026-10-04 against the original adapter at
`29e7e7ca868590fb3a70bc96ebf42271983ab9b6`, stacked on the local SM120 validation
change. Hardware: two RTX 5090s, physical GPUs 0 and 3; CUDA 13.0.88,
driver 580.76.05, NCCL device headers 2.30.4. Neither P2P nor a NCCL communicator
is used by the local fixture. The container denies NUMA policy; the production
FIFO's existing fallback continues.

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

Formal timing is pending the shared host's exclusive heavy-I/O window. The
comparison uses the same fixture source for both builds, alternating baseline
and candidate as ABBA. Each process runs seven rounds; the first two are warmup,
leaving ten samples per variant/device/case. The measured interval covers kernel
launch through production FIFO completion and consumer join. Initialization and
buffer allocation are outside it. Median and p95 are reported in milliseconds;
speedup is baseline median / candidate median.

For payload cases, each group uses a synthetic `BS * 2048 * 2` byte BF16
activation split among 32 producers. BS2048/64 groups exercises 2,048 producers,
256 KiB each, with a 1 GiB source/destination allocation. The width follows
[Qwen3-30B-A3B's configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json).
These fixtures load no model weights and do not measure serving concurrency or
model inference. Payload timings include HBM-to-host oracle copies and checks.

## Network acceptance

`--only coop-flush` is implemented in the existing EFA microbench: eight complete
warps publish disjoint payloads, flush through the actual adapter, and overwrite
their sources. Independent receiver slots and the received pattern must pass.
The invocation is in [README.md](README.md#efa-gate).

Network execution, EFA source consumption, receiver visibility, and full
NCCL-EP/model dispatch/combine e2e remain **not run**: this container exposes no
RDMA device. The new gate's compile success does not establish those properties.
The wider model integration's window/signal contracts remain a separate scope.
