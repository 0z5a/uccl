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
| [hardware_profile_4l4_20261110_summary.csv](data/hardware_profile_4l4_20261110_summary.csv) and [raw runs](data/hardware_profile_4l4_20261110_run1.txt) | November 10: standalone GPU topology, directed and four-GPU concurrent peer copies, concurrent H2D/D2H, and NUMA CPU copy/BF16 reduction. Three complete runs × three repetitions, useful-byte bandwidth; no NCCL baseline or collective timing. [Method and limits](../docs/hardware-profiler.md). |
| [peer_ag_rs_4l4_20261110.txt](data/peer_ag_rs_4l4_20261110.txt) | November 10: four-L4 float32 direct-peer AG/RS versus NCCL 2.28.7, 1 KiB–64 MiB. Three alternating repetitions; slowest-rank CUDA events; every output checked. [Design and negative results](../docs/peer-ag-rs.md). |
| [multinode_peer_20261110/summary.csv](data/multinode_peer_20261110/summary.csv), [summary_large.csv](data/multinode_peer_20261110/summary_large.csv), [summary_ar.csv](data/multinode_peer_20261110/summary_ar.csv) | November 10: two nodes × four L4, standard-layout float32 AG/RS (out of place) and AR (in place) versus NCCL 2.28.7, up to 1 GiB. Three paired alternating runs, ten calls each, slowest-rank CUDA events; full min/median/max and regressions. [Raw logs and excluded prototypes](data/multinode_peer_20261110/README.md), [design and figure](../docs/multinode.md). |
| [gpu-progress/ablation_provenance.json](data/gpu-progress/ablation_provenance.json) and linked CSVs | November 5: CPU versus GPU packet progress, same GPU reduction and packet layout; 16,800 checked timings, all 21 AG/RS sizes. [Direct NCCL results](../docs/baseline-comparison.md). |
| [reduction-placement/ablation_provenance.json](data/reduction-placement/ablation_provenance.json) and linked CSVs | November 5: default CPU, 8-MiB CPU, flat-GPU and interleaved-GPU paths, same executable; 33,600 checked full-sweep timings plus three 1-GiB packet-size probes. AllGather and ReduceScatter, including every regression. [Architecture, blue bar figures and limitations](../docs/reduction-placement.md). |
| [allreduce_single_node.csv](data/allreduce_single_node.csv) | October 12: 1 KiB–1 GiB, 5 warmups, 10 timed iterations, three runs. NCCL 2.18.3/2.28.7 use nccl-tests 2.13.8; UCCL Lite uses its native harness. Fastest correct UCCL Lite variant per size. |
| [allreduce_two_node.csv](data/allreduce_two_node.csv) | October 12: three runs, slowest of eight ranks. Hierarchical UCCL Lite selects among 12 piece/distance configurations. CPU router uses 1 MiB pieces. |
| [cpu_router_4l4_calibration.csv](data/cpu_router_4l4_calibration.csv), [cpu_router_4l4_selection.csv](data/cpu_router_4l4_selection.csv) | New four-L4 BF16 adaptive-module validation against NCCL 2.28.7: three paired repetitions, 5 warmups / 10 iterations, 1–8 MiB AR/AG/RS. All 12 automatic decisions retain the fallback with no router initialization. These are integration calibration results, not the historical peak-throughput sweeps. |
| [local_fallback_8l4_20261110_v3/README.md](data/local_fallback_8l4_20261110_v3/README.md) | November 10: updated eight-L4 BF16 local fallback and size-specific router packet choice. Both placements, 1 MiB–1 GiB AR/AG/RS, 1,980 checked samples each, exact profiles, fresh automatic replays, and packet-size diagnostics. At 1 MiB AR the router beats the separate NCCL control. |
| [local_fallback_8l4_20261110/README.md](data/local_fallback_8l4_20261110/README.md) | Earlier same-day local v2/router calibration, retained with its small-message regression and complete negative results. Its profiles cannot select the revised code. |
| [cpu_router_8l4_20261110/README.md](data/cpu_router_8l4_20261110/README.md) | Earlier same-day eight-L4 BF16 router versus NCCL 2.28.9 control, both placements. Hardware probe, paired timings and old NCCL-backed profiles remain for external comparison; they cannot select against the new UCCL Lite fallback. |
| [primitives_8l4.csv](data/primitives_8l4.csv) | October 12 matched CPU-router primitive rerun after the ReduceScatter fix. Single node, eight L4s, BF16, no GPU P2P; both placements, two retained backends, 24 sizes, three runs. Used for standalone AG/RS charts, not automatic backend selection. [Provenance and port status](../docs/primitive-collectives.md). |
| [collectives.csv](data/collectives.csv) | Earlier compact single-node tables. The AllReduce rows are an older sweep; use the newer single-node CSV for the README figure. |
| [alltoall.txt](data/alltoall.txt), [nccl_alltoall.txt](data/nccl_alltoall.txt) | All-to-All algorithm ladder and the default NCCL transport baseline. |
| [dense_fusion.txt](data/dense_fusion.txt) | September dense shape sweeps, including NCCL, no-overlap, fusion, and Flux controls. Later blocks supersede earlier exploratory blocks. |
| [moe_dispatch.txt](data/moe_dispatch.txt) | Dispatch + GEMM; grouped cuBLAS baseline, settled/interleaved variants. Routing-plan construction excluded. |
| [moe_layer.txt](data/moe_layer.txt) | Dispatch through combine. Twelve rounds × four iterations, min per variant after settling; not the three-run-median protocol used for the collective plots. |
| [tp_moe_prefill.csv](data/tp_moe_prefill.csv) | Complete TP=4 / EP=1 BF16 MoE layer, Qwen and DeepSeek shapes. Medians of three seed-run medians (1801/1901/2001); five rotating rounds × ten graph replays, slowest rank. Baseline searches 15 vLLM configurations. `previous_fused_ms` is whole-stripe readiness; `fused_ms` is shard readiness. |
| [tp_moe_decode.csv](data/tp_moe_decode.csv) | Complete TP-MoE layer at 1/2/4/8/16/32 tokens. Seeds 3201/3301; ten warmups, five rotating rounds × twenty graph replays; baseline tuned in graph mode. `fused-c512` is grouped fusion; `fused-decode` is direct decode. Ranges are paired seed ranges, not confidence intervals. |
| [cutlass_gemm_probe.txt](data/cutlass_gemm_probe.txt), [mma_gemm_occupancy.txt](data/mma_gemm_occupancy.txt) | Diagnostic compute experiments referenced by source comments. |

The CSVs preserve aggregate medians and ranges rather than the complete original
run archive.

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

Figures now use absolute metrics. Small collective messages use latency in
microseconds; large messages use bus or aggregate bandwidth in GB/s. Lower
latency and higher bandwidth are better. Blue marks UCCL Lite, gray the stated
baseline, and amber marks a fused regression in the detailed dense plot.
The recorded min–max timing ranges remain in the CSVs rather than being
presented as confidence intervals.

Single-node AllReduce, All-to-All, and historical eight-GPU AG/RS figures
split their recorded sweeps between small-message latency and large-message
bandwidth. The dense overview compares complete-operation latency at the large
reference shapes; the [labeled shape plot](../docs/fusion.md) keeps all eight
shapes, including regressions. MoE shows full-layer small-batch latency and
large-batch token throughput, computed as tokens per rank divided by measured
full-layer time. The two-node bandwidth figure selects 64 MiB and 1 GiB, and
the small-latency figure selects readable sizes through 1 MiB. Full AG/RS
sweeps and the 512-KiB NCCL anomaly remain in the CSVs.

## Regenerate figures

```bash
python3 scripts/plot_results.py
python3 scripts/plot_multinode_peer.py
python3 scripts/plot_cpu_router_8l4.py
```

Requires Matplotlib and NumPy. `plot_results.py` reads committed CSV/text data
and regenerates seven SVGs under `docs/assets/`: four-GPU AllReduce and All-to-All,
historical eight-GPU AG/RS, dense-fusion overview and labeled shapes, and EP-MoE. The historical AG/RS
figures now appear in [primitive results](../docs/primitive-collectives.md).
`plot_multinode_peer.py` draws two more SVGs from the validated paired
two-node AG, RS, and AR summaries. `plot_cpu_router_8l4.py` draws one SVG
from the current eight-L4 paired BF16 local/router calibration and automatic
selection replay, with the earlier NCCL run as a separate control, for ten
figures total. These scripts do not run GPU work or benchmarks.
Use `python3 scripts/plot_results.py --preview` to also export PNG and PDF copies
under ignored `build/figures/` for inspection or reuse in a technical report.
`python3 scripts/plot_multinode_peer.py --preview` exports PNG/PDF copies there.
`python3 scripts/plot_cpu_router_8l4.py --preview` does the same for the
eight-L4 CPU-router figure.
For fresh measurements, use the commands in [build.md](../docs/build.md), keep
correctness enabled, and save the complete run configuration with each result.
