# Reference measurements

These files retain a compact measurement record for the kernels, including
historical TP-MoE layer measurements whose host harness is not shipped here.
They were collected before the UCCL Lite reorganization. Labels were normalized
to the new project name; numerical values were not changed. Migration-time
validation is recorded separately in [validation.md](../docs/validation.md).

## Testbed and interpretation

- Four NVIDIA L4 GPUs per node, PCIe Gen4, no NVLink; dual Xeon Silver 4410Y.
- NUMA pairs `(0,1)` and `(2,3)`; two-node tests use two local network rails.
- Standalone collectives: float32; dense fusion: FP16; MoE: BF16.
- Speedup is baseline latency divided by compared latency. Above 1× is faster.
- AllReduce bus bandwidth and All-to-All aggregate bandwidth are different
  metrics; do not compare their numerical values as if they were the same.
- A zero-error collective check uses deterministic, position-sensitive inputs.
  It does not prove correctness for all inputs or every floating-point order.

## Included data

| File | Scope and method |
| --- | --- |
| [allreduce_single_node.csv](data/allreduce_single_node.csv) | October 12: 1 KiB–1 GiB, 5 warmups, 10 timed iterations, three runs. NCCL 2.18.3/2.28.7 and TCCL use nccl-tests 2.13.8; UCCL Lite uses its native harness. Fastest correct UCCL Lite variant per size. TCCL uses the corrected 2+2 NUMA map and kernel-only pathfinder policy. |
| [allreduce_two_node.csv](data/allreduce_two_node.csv) | October 12: three runs, slowest of eight ranks. Hierarchical UCCL Lite selects among 12 piece/distance configurations. CPU router uses 1 MiB pieces. Includes TCCL and an explicit memcpy staging control. |
| [cpu_router_4l4_calibration.csv](data/cpu_router_4l4_calibration.csv), [cpu_router_4l4_selection.csv](data/cpu_router_4l4_selection.csv) | New four-L4 BF16 adaptive-module validation against NCCL 2.28.7: three paired repetitions, 5 warmups / 10 iterations, 1–8 MiB AR/AG/RS. All 12 automatic decisions retain the fallback with no router initialization. These are integration calibration results, not the historical peak-throughput sweeps. |
| [primitives_8l4.csv](data/primitives_8l4.csv) | October 12 matched CPU-router primitive rerun after the ReduceScatter fix. Single node, eight L4s, BF16, no GPU P2P; both placements, four backends, 24 sizes, three runs. Used for standalone AG/RS charts and the explicit reference selection profile. [Provenance and port status](../docs/primitive-collectives.md). |
| [collectives.csv](data/collectives.csv) | Earlier compact single-node tables. The AllReduce rows are an older sweep; use the newer single-node CSV for the README figure. |
| [alltoall.txt](data/alltoall.txt), [nccl_alltoall.txt](data/nccl_alltoall.txt) | All-to-All algorithm ladder and the default NCCL transport baseline. |
| [dense_fusion.txt](data/dense_fusion.txt) | September dense shape sweeps, including NCCL, no-overlap, fusion, and Flux controls. Later blocks supersede earlier exploratory blocks. |
| [moe_dispatch.txt](data/moe_dispatch.txt) | Dispatch + GEMM; grouped cuBLAS baseline, settled/interleaved variants. Routing-plan construction excluded. |
| [moe_layer.txt](data/moe_layer.txt) | Dispatch through combine. Twelve rounds × four iterations, min per variant after settling; not the three-run-median protocol used for the collective plots. |
| [tp_moe_prefill.csv](data/tp_moe_prefill.csv) | Complete TP=4 / EP=1 BF16 MoE layer, Qwen and DeepSeek shapes. Medians of three seed-run medians (1801/1901/2001); five rotating rounds × ten graph replays, slowest rank. Baseline searches 15 vLLM configurations. `previous_fused_ms` is whole-stripe readiness; `fused_ms` is shard readiness. |
| [tp_moe_decode.csv](data/tp_moe_decode.csv) | Complete TP-MoE layer at 1/2/4/8/16/32 tokens. Seeds 3201/3301; ten warmups, five rotating rounds × twenty graph replays; baseline tuned in graph mode. `fused-c512` is grouped fusion; `fused-decode` is direct decode. Ranges are paired seed ranges, not confidence intervals. |
| `tccl_ring_run{1,2,3}.txt` | October 13: final 1 GiB host-staged MSCCL++ global-ring runs, eight channels/eight slots, zero relative error. |
| [cutlass_gemm_probe.txt](data/cutlass_gemm_probe.txt), [mma_gemm_occupancy.txt](data/mma_gemm_occupancy.txt) | Diagnostic compute experiments referenced by source comments. |

The CSVs preserve aggregate medians and ranges rather than the complete original
run archive. The new global-ring implementation's 19.21 GB/s result is separate
from the hierarchical `uccl_lite` curve in `allreduce_two_node.csv`.

The dense and MoE benchmarks include their own baseline implementations. Large
fusion speedups are specific to those workloads; small cases can regress. The
L4 power limit changes clocks under sustained compute, so variant interleaving
and thermal settling matter. Cross-binary Flux comparisons have weaker controls.

TP-MoE timings charge router projection, top-k, GPU routing plan, shared
experts, both routed projections, activation, combination, and AllReduce.
Weights and hidden states are synthetic. These CSVs preserve existing layer
measurements, not application throughput results. Importing the data does not
import the application-side benchmark or imply a fresh validation of the headers.

## Figure conventions

The README uses grouped bars with a consistent blue palette: dark navy for the
primary UCCL Lite path, lighter blues for reference or intermediate variants.
All bar axes start at zero. Speedup panels show a dashed 1× baseline and preserve
regressions. Only existing measurements are plotted; no missing series are
interpolated and no uncertainty is invented.

The standalone collective figures select representative message sizes to
keep bars legible. The full numerical sweeps remain in the data files. Dense
fusion uses the later "Current full comparison" block, not the superseded
exploratory numbers. Flux latencies are normalized to the same cuBLAS+NCCL
baseline as UCCL Lite, while retaining the different-harness caveat. The EP-MoE
panels show their full available token ranges, which differ between the two logs.

## Regenerate all README figures

```bash
python3 scripts/plot_results.py
```

Requires Matplotlib and NumPy. The script reads the committed CSV/text data and
regenerates six SVGs under `docs/assets/`: four-GPU AllReduce, eight-GPU
standalone AllGather and ReduceScatter (each showing both placements), All-to-All,
dense fusion, and EP-MoE. It performs no GPU work and runs no benchmarks.
Historical TP-MoE and two-node records remain archived above but are not part of
the README results gallery or figure generation.
Use `python3 scripts/plot_results.py --preview` to also export PNG and PDF copies
under ignored `build/figures/` for inspection or reuse in a technical report.
For fresh measurements, use the commands in [build.md](../docs/build.md), keep
correctness enabled, and save the complete run configuration with each result.
