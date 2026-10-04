# Native SM120 local validation results

Tested on 2026-10-04 (Asia/Shanghai), against `29e7e7ca868590fb3a70bc96ebf42271983ab9b6`.

Two RTX 5090s (physical indices 0 and 3), driver 580.76.05, CUDA 13.0.88, NCCL header mode disabled. Each device reports CC 12.0, compiled architecture 1200, two asynchronous copy engines, and successful mapped-host/FIFO allocation. Both peer-access directions report 0.

| Case | Physical GPU 0 median ms | Physical GPU 3 median ms | Integrity |
| --- | ---: | ---: | --- |
| 100,000 commands, 1 producers, capacity 32, 3 rounds | 60.871 | 60.954 | All commands and one scalar QUIET per round |
| 100,000 commands, 32 producers, capacity 32, 3 rounds | 13.903 | 15.484 | All commands and one scalar QUIET per round |
| 100,000 commands, 32 producers, capacity 512, 3 rounds | 11.768 | 12.676 | All commands and one scalar QUIET per round |
| 100,000 commands, 256 producers, capacity 512, 3 rounds | 12.105 | 11.542 | All commands and one scalar QUIET per round |
| 100,000 commands, 512 producers, capacity 512, 3 rounds | 11.947 | 12.054 | All commands and one scalar QUIET per round |
| Concurrent 500,000 commands/device, 512 producers, 3 rounds | 48.942 | 49.416 | Both processes exit 0 |

Timings are supplementary shared-host observations from the correctness run. This PR adds an independent validation path; it changes no FIFO algorithm and makes no before/after speedup claim.

| Build / capability gate | Result |
| --- | --- |
| Native SM120 and mapped-host GPU write | Pass on both GPUs |
| SM90 compilation in separate object directory | Pass; not executed on SM120 GPUs |
| Runtime linkage | CUDA, libnuma, standard C/C++/pthread; no EFA/MPI/NCCL/PyTorch |
| Signed atomics, reserved bit, all decoded fields, duplicates/missing commands | Pass |
| Repeated wraparound and scalar completion | Pass |
| Full EFA transport build and distributed GIN / model e2e | Not run: this container exposes no RDMA device or EFA/MPI development stack |

The host consumer uses `Fifo::poll()`, the production decoder, and `Fifo::pop()`. Test acknowledgements validate the local command/completion path; they do not establish NIC source consumption or receiver visibility.

Build and run commands are in [README.md](README.md). Raw JSONL logs, compiler output, `cuobjdump`, dependency output, and GPU snapshots are retained with the local execution artifacts. NUMA policy is denied by the container; the existing production fallback warns and continues.
