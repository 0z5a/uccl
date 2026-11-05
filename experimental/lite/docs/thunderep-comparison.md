# ThunderEP comparison — in progress

**No result currently establishes that UCCL Lite outperforms ThunderEP.**
Our historical four-L4 fusion speedups are not comparable to its model-level
results. The goal is a matched evaluation, followed by architectural changes
if the measurements require them—not a comparison of unrelated speedup ratios.

## Reference configuration

Source: [ThunderEP, arXiv:2609.40093v1](https://arxiv.org/pdf/2609.40093),
§§4–5, Table 1, Figures 11–14, and Appendix A. Retrieved November 5, 2026;
PDF SHA-256: `977b985bcb74cfb1c886e68805aad93682f44ffb03010e8968099b32bc8d7941`.

| Dimension | Paper configuration |
| :--- | :--- |
| GPU | Six RTX 4090s or six RTX 5090s; no P2P; three host bridges, two GPUs each |
| Host | EPYC 9124, 16 cores; single NUMA node; eight 32-GB DDR5-4800 DIMMs; GENOAD8X-2T |
| Software | Ubuntu 24.04.3, Linux 6.8.0-35, driver 580.65.06, CUDA 13.0; NCCL 2.30.7 |
| Frameworks | vLLM 0.27.1, SGLang 0.5.18, Megatron-Core 0.19.2 |
| Models | Qwen3-30B-A3B: BF16; GPT-OSS-20B/120B: native MXFP4 checkpoints |
| Primitives | AllGather and ReduceScatter; full-array sizes 1 KiB–1 GiB; five-run averages |
| Prefill | Batch 24; prompt lengths 256–4096; 256 output tokens |
| Decode | Prompt length 4096; batches 6–96; 256 output tokens |
| Prompts | Synthetic, ShareGPT, LMSYS-Chat-1M; matched precision and EP/DP settings |

ThunderEP uses shared-host single-step communication, separate DMA directions,
sender-level readiness, and GPU reduction. It uses SM transfers for decode
graphs and overlaps expert computation during prefill. CPU reduction and
routing-aware DMA did not help on its evaluated systems. These findings do not
establish what wins on a different NUMA/PCIe topology.

## Gaps in our current implementation

The local machine has four L4s, two Xeon Silver 4410Y sockets and two NUMA nodes,
CUDA 12.9 and driver 595.58.03. Disabling P2P does not make it the paper's machine.

| Component | Current evidence | Missing for a matched comparison |
| :--- | :--- | :--- |
| EP-MoE fusion | Four-rank CUDA IPC transport; host-generated plan outside timing | No-P2P transport, required model shapes, dynamic routing charged to latency |
| CPU-router | Host-staged BF16 AG/RS, NUMA-aware workers, optional backend policy | Matched no-P2P measurements; overlap and end-to-end benefit |
| Decode | Low-level router graph tests exist; adaptive wrapper is eager-only | Real decode graph with changing inputs/routes and full lifetime validation |
| Baseline | Local NCCL comparison can be reproduced | Author implementation, exact measurement protocol, matched consumer-GPU host |
| End-to-end | Not part of this standalone subtree | Separate evaluation integration, checkpoints, dataset sampling and all settings |

Application integrations must remain **outside** `experimental/lite/`. They are
still necessary for the final model-level evaluation; a standalone kernel win
will not be substituted for that requirement. The removed README result gallery
is not restored by this research work.

Open protocol questions: primitive dtype, bandwidth normalization, placement,
warmup/iteration counts, rank launch model, exact EP/DP factorization, NCCL tuning,
and the six-rank rounding of binary message sizes. Do not silently fill these in
and label the result an exact reproduction. No verified public ThunderEP source
repository was found in the initial paper/GitHub search.

## Local diagnostic, reproducible now

Build the reference NCCL into ignored storage without replacing existing builds:

```bash
git clone --depth 1 --branch v2.30.7-1 https://github.com/NVIDIA/nccl.git build/nccl-2.30.7
make -C build/nccl-2.30.7 -j4 src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_89,code=sm_89'
make -j2 adaptive-collectives BUILD=build/thunderep-nccl23007 \
  NCCL_HOME="$PWD/build/nccl-2.30.7/build"
python3 scripts/compare_thunderep.py \
  --binary build/thunderep-nccl23007/uccl_lite_collectives \
  --output build/thunderep-local-001
python3 -m unittest discover -s tests -p test_compare_thunderep.py -v
```

The script runs paired NCCL/forced CPU-router AG and RS sweeps. It disables NCCL
P2P and GPU RDMA, retains NCCL defaults otherwise, records inherited overrides,
captures hardware/software identity, and checks the loaded NCCL version.
It requires a new output directory and never overwrites a previous experiment.
The reference NCCL tag resolves to commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`; the local build uses CUDA 12.9,
not the paper's CUDA 13.0. No algorithms or protocols are filtered from this build.

Each size has five alternating backend-order repetitions, five warmup calls per
backend, and twenty measured calls per repetition. Every call is checked on all
ranks. Raw samples are the slowest-rank CUDA-event time per invocation; the
summary is the arithmetic mean of the five per-run means, with min/max run means.
The existing calibration policy still uses its original median/range gate.
Both algorithm bandwidth and NCCL-style bus bandwidth are labeled explicitly.
Missing, duplicate, nonfinite, or nonpositive samples fail summary generation.

This is **eager BF16, out-of-place, one process managing four GPUs**. CUDA-event
intervals include idle time while the host submits work and the router publishes
its program; initialization, input fill and verification are outside the measured
interval. It is not a graph-replay latency or an implementation of ThunderEP.
Five repetitions share a process and initialized communicators; the paper's
process-restart policy is unknown. A local NCCL win is only a local NCCL win.
Both candidates coexist in memory, and CPU-router workers remain allocated
while measuring NCCL; isolation and steady-state graph timing need separate
experiments before interpreting small-message overheads as device latency.

## Architecture experiments to test next

The first implemented experiment is
[reduction placement with destination striping](reduction-placement.md): CPU
reduction, flat destination-GPU reduction, and an interleaved destination schedule.
It remains opt-in and does not change the default backend-selection policy.

### First local diagnostic — November 5, 2026

The full 21-size AG/RS sweep completed with **8,400 checked timed calls** and
host-staged NCCL channels verified as `SHM/direct`. Five-run mean latencies:

| Collective | Full-array bytes | NCCL 2.30.7 | CPU-router | NCCL / router |
| :--- | ---: | ---: | ---: | ---: |
| AllGather | 1 KiB | 46.948 µs | 44.648 µs | 1.052× |
| AllGather | 1 MiB | 111.462 µs | 110.141 µs | 1.012× |
| AllGather | 1 GiB | 47.774 ms | 51.459 ms | 0.928× |
| ReduceScatter | 1 KiB | 41.166 µs | 41.604 µs | 0.989× |
| ReduceScatter | 1 MiB | 96.220 µs | 278.332 µs | 0.346× |
| ReduceScatter | 1 GiB | 46.984 ms | 54.098 ms | 0.869× |

CPU-router has a lower mean at 9/21 AG sizes and 0/21 RS sizes. This count is
not a significance test or an automatic deployment recommendation. All sizes,
including unfavorable cases and per-iteration outliers, are retained in the
[summary](../benchmarks/data/thunderep_local_4l4_summary.csv),
[AG samples](../benchmarks/data/thunderep_local_4l4_ag_samples.csv),
[RS samples](../benchmarks/data/thunderep_local_4l4_rs_samples.csv), and
[provenance](../benchmarks/data/thunderep_local_4l4_provenance.json).

These observations prioritize a controlled CPU/GPU/hybrid reduction-placement
ablation, plus a separate submission/graph-timing analysis. They do **not**
identify the cause by themselves and do not compare against ThunderEP. No
superiority claim has yet passed the completion gates. The implementation and
subsequent ablations are documented separately above.

Validation of this measurement tooling: four host-side summary tests pass; the
31 existing policy checks pass; 108 additional in-place calls cover all three
collectives with `off`, `force`, and `auto` sample logging. Existing sample files
are rejected without modification. The first smoke exposed a collective-name
schema mismatch; that parser issue was corrected before the archived full run.
The source and data checksums match, and all 84 summary rows recompute from the
8,400 raw samples. None of these checks validates a new transport architecture.

### Candidate changes

Shared host buffers, dual DMA streams, and sender-ready overlap alone would
reproduce the reference's mechanisms; they are not new contributions.

1. **Topology-aware reduction placement.** Select CPU, destination-GPU, or
   NUMA-local partial reduction using measured host-memory pressure and transfer
   costs. Compare all variants with identical routes and computation, including
   regressions; GPU count alone must never select the backend.
2. **GPU-published routing with bounded host work.** Keep route selection and
   packing on the GPU; publish coarse ready descriptors to a persistent progress
   engine. Test whether skipped destinations actually repay descriptor latency
   and packing overhead, including skew and tiny decode batches. Charge all
   metadata work; a host-precomputed plan is not a valid win.
3. **Graph-stable pipelining.** Investigate fixed-address staging slots and
   explicit ownership/completion epochs to overlap communication with expert
   computation without per-replay host graph edits. Validate slot reuse, delayed
   senders, changing routes and repeated graph replay before timing.

The reduction-placement experiment is now implemented; hybrid NUMA reduction,
GPU-published routing, and graph-stable MoE overlap remain hypotheses. Carry
validated primitive changes through layer and model-level ablations on the
required hardware before making a complete-system claim.

A concrete existing limitation is visible in
[packet_pipeline.cu](../src/cpu_router/packet_pipeline.cu): the `Copy` branch of
`routePackets` waits for every source before publishing `routed`, and the GPU
consumer waits for that publication before reading any peer. Measuring and
removing this all-source gate is a prerequisite for sender-level overlap, not
by itself a claim of novelty over ThunderEP. Buffer ownership and multi-reader
release acknowledgements must remain correct when exposing earlier readiness.

## Completion gates

- Obtain the target consumer-GPU host and the author implementation, or clearly
  identify and validate a faithful reproduction before any head-to-head claim.
- Resolve the open evaluation settings and preserve the original baseline
  kernels, precision, workload and parallelism when changing communication.
- Check correctness under changing routing, numerical tolerances, delayed ranks,
  and graph replay, including the selected fallback paths.
- Run all primitive sizes and all model/workload cases; preserve raw five-run
  data, unfavorable cases, OOMs, environment and source identities.
- If UCCL Lite loses, implement and ablate system-level changes, then repeat the
  same measurements. No selective shape or baseline substitution.
- Publish matched ThunderEP head-to-head bars only after these gates pass.
  Local ablation figures must remain explicitly labeled as local diagnostics.
  Keep primitive, layer, and complete-inference claims separate.
