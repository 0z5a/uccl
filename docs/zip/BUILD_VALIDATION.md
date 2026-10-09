# CUDA build and host protocol validation

Date: 2026-10-09. Local branch: `zip-peer-regions-main`.

The complete source builds in an isolated local Linux environment with CUDA
12.8.1, GNU C++ 13.3.0, Python 3.12.3 and nanobind 2.9.2. The build produces the
Python extension, P2P shared library, DietGPU library and SM90 native decoder
test. Both `USE_DIETGPU=1` and `USE_DIETGPU=0` pass the complete P2P build and
module import. No H100 execution or model E2E result is established by this build.

## Failures found and changes

The standalone production decoder initially fails to compile because
`GpuFloatCodec.h` uses `assert` without including `<cassert>`. The header now
includes its required declaration. The same failure was reproduced separately
against the frozen `70213197` native source. Its minimal successor is
`78687754`, which changes only this include and passes a clean native build.

`getElementCountFromBytes` was declared inline in the public header while its
body existed only in a CUDA translation unit. It is now an ordinary exported
function. The final library exports a strong definition and the consumer
translation units no longer report a missing inline definition.

Loading the extension initially aborts on ARM64 because the shared timer applies
an x86 frequency range to the ARM virtual counter. ARM64 now obtains its counter
frequency from `CNTFRQ_EL0`, following the [Arm Generic Timer
documentation](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Generic%20Timer.pdf?revision=c710e7a7-9f52-4901-8c9d-91b19f44f9c7).
The existing x86 calibration remains in place. A test compares elapsed counter
time with `steady_clock`; the extension subsequently loads successfully.

## Evidence and remaining gates

| Check | Before | Current evidence | Speedup |
|---|---|---|---|
| Standalone production decoder compilation | Missing `assert` declaration | Clean native build PASS | N/A |
| Complete optional DietGPU build | Decoder compilation fails | Python extension and P2P library build PASS | N/A |
| Default build with DietGPU disabled | New API compatibility unverified | Clean P2P build, import and relocations PASS | N/A |
| ARM64 extension import | Timer initialization aborts | Import and receiver-ready method bindings PASS | N/A |
| Counter time conversion | Import abort prevents use | Elapsed-time comparison PASS | N/A |
| P2P shared-library relocations | Not checked | No missing dependency or symbol with SDK driver stubs | N/A |
| ACK generations and errors | Prior host results | Linux production protocol test PASS | N/A |
| Peer regions and slot reuse | Prior host results | Linux production allocator/slot tests PASS | N/A |
| Tracker wrap and concurrent updates | Prior host results | Linux production tracker test PASS; 131,072 transitions | N/A |
| Handshake compatibility | Prepared test | Linux real serialization and legacy extent PASS | N/A |
| Native H100 decoder and memcheck | No new execution result | Pending | Not measured |
| Compressed RDMA destination readiness | Prepared test | Pending | Not measured |
| High-concurrency RL / vLLM PD model E2E | No new execution result | Pending | Not measured |

SDK driver stubs are used only to resolve imports and relocations in this local
build. No endpoint or GPU context is created during the binding check. The build
does not qualify H100 kernels, NIC ordering, x86 runtime behavior or speedup.
The existing nanobind stable-ABI macro redefinition warning remains.

Raw logs and execution metadata remain private. New work stays on local
branches; no new PR, remote push or main update is included in these results.

The optional [verl checkpoint integration](VERL_CHECKPOINT_DESIGN.md) separately
passes four host control/dataflow tests and four consumer-credit tests. These
use a simulated P2P data plane and do not change the pending hardware/E2E rows.
