# Validation and limitations

## Migration validation — November 5, 2026

The UCCL Lite reorganization was checked on four NVIDIA L4 GPUs, driver
595.58.03, CUDA 12.9, with NUMA pairs `(0,1)` and `(2,3)`. Source changes consist
of relocation, project-owned naming, include/build plumbing, formatting, and
documentation. The collective and fusion algorithms retain their existing
numerical behavior.

| Check | Result |
| --- | --- |
| `make all` | All single-node, fusion, and probe executables build; reusable MoE headers compile |
| `make multinode` | All five two-node executables build with Open MPI and IB-enabled MSCCL++ |
| AllReduce, 1 KiB–16 MiB, doubling sizes | All reported correctness errors are zero across eligible variants |
| All-to-All, 1 KiB–1 MiB, doubling sizes | Zero mismatches, including the host-staging baseline |
| Dense AG/GEMM, GEMM/RS, GEMM/AR small-shape smoke | All 11 reported checks have `bad=0`; maximum absolute difference 0.0078125 |
| MoE dispatch + GEMM, 8 tokens/rank | Checked variants have zero misrouted rows and zero relative error |
| DeepSeek-shaped full MoE layer, 2,048 tokens/rank, `-c 2` | Zero misrouted rows; normalized RMS error 0.00225 against the FP32 reference |
| Qwen-shaped full MoE layer, 8 tokens/rank, `-c 2` | Zero misrouted rows; normalized RMS error 0.00460 against the FP32 reference |

Builds retain unused-variable warnings in diagnostic benchmark variants. These
are not CUDA compilation errors. Smoke timings are not performance claims: they
use short runs with minimal warmup and no thermal settling.

The checked commands are in [build.md](build.md). The DeepSeek prefill check is:

```bash
./build/uccl_lite_fused_moe_layer \
  -b 2048 -e 2048 -w 1 -n 1 -r 1 -S 0 -c 2
```

The new source follows the parent repository's clang-format 14 configuration.
The launcher is checked with `bash -n`; figures are regenerated from the
committed data. Local Markdown links and the staged file inventory are checked
before publication. Generated binaries and validation logs stay in ignored
`build/` storage.

## Figure expansion

The README now includes ten blue bar figures generated from committed
CSV/text measurements. Generation checks for missing series, non-finite or
nonpositive values, and invalid error ranges. The dense figure selects the
current comparison block; TP-MoE adds retained prefill/decode summary data.
All bar axes start at zero and the plotted small-batch regressions are retained.
PNG previews were visually checked for layout; SVG and PDF exports use the
same plots. This is a
documentation/visualization update; no kernels changed and no GPU benchmarks
were rerun for it.

Standalone AllGather and ReduceScatter figures additionally use the complete
matched eight-L4 BF16 CPU-router summary from source revision `9d088c9`.
The CSV checksum, dimensions, bandwidth normalization, and both placements were
checked. Its separate runtime is not included or validated by the current build;
see [primitive results](primitive-collectives.md).

## What this validation does not establish

- The reorganized multi-node binaries have not been executed across two hosts.
  Their performance and correctness tables are historical measurements.
- TP-MoE device headers are compile-checked, not numerically exercised by an
  included host driver. Their former application-side harness is out of scope.
- One smoke pass does not establish race freedom, arbitrary-stream safety,
  behavior across repeated process lifetimes, or portability to other GPUs.
- FP16/BF16 accumulation and reduction order differ from some reference paths.
  Passing tolerance checks is not a claim of bitwise model-output equivalence.
- The opt-in persistent host-hierarchical prototype has known correctness
  failures and remains excluded from the validated transport paths.

## Extending the checks

For communication changes, test small and large sizes, uneven tails where the
algorithm supports them, repeated invocations, and delayed ranks. Inspect data
and flag visibility across GPU, host, and NIC domains separately. Validate both
the default path and any fallback before publishing a performance curve.

For fusion changes, compare complete measured work against an efficient unfused
control, preserve the dtype and rounding boundaries, and include small shapes
that expose launch and scheduling overhead. Use new seeds and representative
routing before making a dispatch policy. Always state whether routing-plan
construction and framework work are inside the timed interval.
