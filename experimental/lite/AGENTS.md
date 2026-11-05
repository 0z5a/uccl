# UCCL Lite

This directory contains standalone CUDA collectives, fusion kernels, benchmarks,
and design notes for PCIe GPU systems. MSCCL++ supplies the external communication
runtime. Keep changes scoped to this directory unless another UCCL component
actually needs updating.

- Keep application integrations, serving hooks, Torch extensions, checkpoints,
  vendored dependency trees, and compiled artifacts out of this project.
- Preserve MSCCL++, CUDA, NCCL, and CUTLASS API names. Project-owned names use
  `uccl_lite` or `UCCL Lite`.
- `src/collectives/` contains standalone collective algorithms and their harnesses.
  `include/uccl_lite/` contains reusable device/fusion headers.
  `benchmarks/` contains fusion harnesses, probes, and reference measurements.
- Use the Makefile. Keep generated binaries and dependency files in `build/`.
- Match the repository's clang-format 14 configuration.
- Build affected CUDA targets. For collective changes, run correctness checks on
  the documented four-GPU topology; multi-node changes require two four-GPU hosts.
  State clearly when validation is limited to compilation.
- Benchmark reports must identify hardware, dtype, baseline, timing scope,
  repetitions, and whether a curve selects the best variant per size.
- Treat historical results as measurements of their recorded configurations,
  not validation of a subsequent source change. Preserve negative results and
  explicit experimental status.
