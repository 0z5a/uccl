# Optional CPU-router module

The opt-in [reduction-placement experiment](reduction-placement.md) adds
destination-GPU arithmetic and interleaved scatter packets to the C++ module.
Existing CPU defaults and measurement-gated fallback behavior are retained.
The subsequent [GPU progress experiment and baseline comparison](baseline-comparison.md)
remove CPU packet-routing workers in an explicit GPU-reduction configuration.

CPU-router is an independent component for routing BF16 collective traffic
through NUMA-local mapped host memory. It is useful when direct peer paths are
unavailable or less efficient for the measured workload. It can be left entirely
uninitialized on systems where the existing backend wins.

## Components and build boundaries

| Component | Responsibility |
| --- | --- |
| `RegisteredMemory`, `MemorySpan` | GPU/NUMA allocation, registration, bounded views, retained lifetime |
| `Channel`, `DeviceChannel` | GPU/host transfers and completion tickets |
| `CpuExecutor` | NUMA-local worker lanes, bounded queues, FP32 accumulation with BF16 rounding |
| `RouterCore` | Reusable `forward`, `process`, and `multicast` operations |
| `PacketPipeline` / `CpuRouter` | AR/AG/RS packet programs, two-slot generation and reader-credit protocol |
| `BackendPolicy` | Pure host selection from capability and validated timing records |
| `CollectiveDispatcher` | Optional router around a caller-supplied existing-backend callback |
| `LocalCollectives` | Project-owned eager BF16 fallback for two four-GPU NUMA groups; no NCCL dependency |

Source lives under `src/cpu_router/`; public headers are under
`include/uccl_lite/cpu_router/`, plus
[backend_policy.hpp](../include/uccl_lite/backend_policy.hpp) and
[collective_dispatch.hpp](../include/uccl_lite/collective_dispatch.hpp).
The core is ported from the primitive implementation associated with source
revision `9d088c9`, including the medium-message ReduceScatter fix. Namespace,
exported C functions, and diagnostic environment variables use UCCL Lite names.
The local fallback is [separate CUDA code](../src/collectives/local_collectives.cu)
with a [public C++ interface](../include/uccl_lite/local_collectives.hpp).

```bash
make -j2 cpu-router CUTLASS_HOME=/path/to/cutlass
# build/cpu_router/libuccl_lite_cpu_router.a
# build/cpu_router/libuccl_lite_cpu_router.so

make local-collectives
# build/libuccl_lite_local_collectives.a

make -j2 adaptive-collectives NCCL_HOME=/path/to/nccl/build
make -j2 router-checks local-collectives-check
```

The module uses Linux/x86-64, AVX2, CUDA, pthreads, and CUTLASS 4.5.0. Its
packet pipeline retains resident-GEMM building blocks, hence the CUTLASS build
dependency. It does not link MSCCL++ or NCCL. The standalone adaptive
benchmark links NCCL only for an explicitly selected external control;
its default fallback is UCCL Lite local. The original `make`,
`single-node`, `fusion`, and their binaries do not link or initialize this module.

No framework patches, application integrations, or NCCL interposition shim are
included. The C wrapper is a direct low-level API, not an NCCL API replacement.

## Selection policy

`auto` is conservative and does not benchmark during a collective invocation.
The selection key includes collective, BF16 dtype, exact byte count, in-place
placement, topology identity, and the identity of the caller's fallback.
No interpolation or best-neighbor size policy is applied.

For AllReduce through 4 MiB and ReduceScatter from 256 KiB through 4 MiB,
the adaptive dispatcher prepares the router with 128 KiB packets by default.
On 8xL4 GPU, 128/256 KiB AllReduce also uses four CPU workers per NUMA node
and 16 transport blocks, based on paired local and NCCL measurements.
At exactly 128 KiB, ReduceScatter uses GPU reduction and GPU progress with
the regular packet limit. Other calls use the 512 KiB packet default and CPU
reduction. These choices happen in `prepare`, outside the timed invocation.
`--small-ar-packet-bytes 0`, `--small-rs-packet-bytes 0`, and
`--no-small-rs-gpu-progress` disable the respective example choices. Packet
geometry and reduction mode are part of the profile identity, so changing
them requires new calibration.

The optional `largeAllGatherPacketBytes` setting selects larger packets for
AllGather outputs of at least 4 MiB. On 8xL4 GPU, the benchmark sets it to
2 MiB and keeps the 512 KiB default below 4 MiB. The benchmark flag is
`--large-ag-packet-bytes 2097152`. This packet setting is also included in
the profile identity.

1. Unsupported dtype/topology/capacity retains the existing backend.
2. Only an exact local calibration record can enable the router. A newer
   matching record showing a regression disables it.
3. A candidate is selected only when it passed correctness, has at least three
   repetitions, median speedup is at least 1.03×, and its slowest run is faster
   than the fallback's fastest run. Overlapping ranges retain the fallback.
4. Unknown configurations and unmeasured sizes retain the existing backend.
   In the eight-GPU example that is UCCL Lite local; load the matching profile
   before relying on automatic AR/RS performance.

The archived eight-L4 profile describes its recorded hardware/software
configuration, but cannot capture a new host's PCIe contention, memory setup,
CPU model, or every NCCL setting. It remains a performance reference and is not
accepted as automatic routing evidence. Recalibrate when CPU, memory, or
workload conditions change.

The four-L4 host cannot match eight-GPU historical records. With no local
evidence, it stays on its existing backend and does not allocate router buffers
or start workers. Local evidence can opt in on any supported topology if it
actually demonstrates a stable gain; four GPUs are not categorically forbidden.

`off` always retains the existing backend. `force` is a diagnostic override;
it still rejects unsupported dtypes, missing capabilities, or insufficient
capacity. It bypasses the performance decision, not correctness requirements.

## Reusable caller interface

One process owns all device ports. A caller provides its existing implementation
as a callback; the policy does not assume that callback is NCCL. For example:

```cpp
#include <uccl_lite/collective_dispatch.hpp>

uccl_lite::BackendPolicy policy;
policy.fallbackIdentity = "my-existing-backend:v1";
policy.measurements = uccl_lite::loadRouterProfile("local.profile");

uccl_lite::CollectiveDispatcher dispatcher(
    uccl_lite::inspectTopology(devices), policy,
    [&](const uccl_lite::CollectiveCall& call) { existingBackend.enqueue(call); });

// Outside graph capture, with valid persistent buffers and idle streams.
auto selection = dispatcher.prepare(call);
dispatcher.enqueue(call);
```

The callback must implement the supplied collective, dtype, placement, and
streams. `fallbackIdentity` must identify that actual implementation and change
when its algorithm/configuration changes. Do not label a custom callback with
an NCCL identity to reuse unrelated measurements.

For the eight-L4 topology, `LocalCollectives` provides that callback. It needs
two NUMA groups of four GPUs with peer access within each group. Its
`prepare()` allocates pinned host and GPU scratch outside timing; its
`enqueue()` is eager and synchronizes its submitted streams before returning.
The local path sends AllGather shards through NUMA-pinned host buffers. For
AllReduce and ReduceScatter through 2 MiB it uses eight pinned input buffers
and CPU reduction workers. Larger reductions sum within each four-GPU peer
group, exchange partials through host memory, then distribute results. This
path does not use NCCL or CPU-router workers. Other topologies need their own
caller-supplied fallback.

```cpp
#include <uccl_lite/local_collectives.hpp>

auto topology = uccl_lite::inspectTopology(devices);
uccl_lite::LocalCollectives local(topology, maxBytes);
uccl_lite::BackendPolicy policy;
policy.fallbackIdentity = uccl_lite::LocalCollectives::identity();
policy.measurements = uccl_lite::loadRouterProfile("caller-local.profile");
uccl_lite::cpu_router::CpuRouterOptions options;
options.maxTensorBytes = maxBytes;
uccl_lite::CollectiveDispatcher dispatcher(
    topology, policy,
    [&](const uccl_lite::CollectiveCall& c) { local.enqueue(c); }, options);

auto selection = dispatcher.prepare(call);  // Outside capture, streams idle.
if (!selection.cpuRouter) local.prepare();
dispatcher.enqueue(call);
```

The profile in this example must be calibrated for the caller's own callback
and topology identity. The standalone benchmark appends its own identity key,
so its profile is not interchangeable with another caller's profile.

`CollectiveCall::count` follows the usual collective API convention: full tensor
elements for AR, contribution elements for AG, output-shard elements for RS.
The policy normalizes these to complete per-rank bytes (AG output / RS input).
In-place aliasing and port device order are checked before dispatch. Unsupported
router types reach the existing callback without conversion.

`prepare` creates the router only for a selected router workload. `enqueue` never
implicitly initializes it. Inspect `decision().reason` and `routerInitialized()`
to audit selection. When introducing a new size that first selects the router,
call `prepare` at an initialization boundary before submitting that work.

The topology snapshot records GPU UUID/PCI/device order, NUMA placement, peer
capabilities, CPU model and permitted affinity, driver/toolkit, and router
geometry. Local profiles must match this identity. Recreate the dispatcher after
changing device visibility, affinity, or execution configuration.

## Paired calibration and the executable example

The included example uses BF16 and supplies `LocalCollectives` as its default
fallback on the eight-GPU topology. `--fallback nccl` retains the previous
external comparison mode. It allocates its own scratch tensors, verifies every
invocation outside timing, alternates backend ordering between repetitions,
and records the slowest-rank CUDA-event latency. Three run-level medians form
the decision's min/median/max range.

```bash
# Local paired measurement. The output must be a new file.
./build/uccl_lite_collectives --gpus 8 --fallback local --op all \
  --begin 1048576 --end 1073741824 --warmup 5 --iterations 10 --repeats 3 \
  --calibrate build/local-out-of-place.profile

# Reuse the measurements. Nonwinning records select the fallback.
./build/uccl_lite_collectives --gpus 8 --fallback local --op all --backend auto \
  --begin 1048576 --end 1073741824 --profile build/local-out-of-place.profile

# In-place is a separate workload key and needs its own measurements.
./build/uccl_lite_collectives --gpus 8 --fallback local --op all --in-place \
  --begin 1048576 --end 1073741824 --calibrate build/local-in-place.profile

# Force is useful for correctness and diagnostics, not a deployment default.
./build/uccl_lite_collectives --gpus 8 --fallback local --backend force --op all
```

`--devices 0,2,1,3,4,6,5,7` selects an explicit rank order. For the older
four-GPU NCCL comparison, pass `--gpus 4 --fallback nccl` explicitly.
`--backend off` measures the selected fallback. Calibration enables both
candidates intentionally; merely loading
a profile in automatic mode does not start CPU workers if all selected sizes lose.

Calibration retains both initialized candidates while comparing them. The CSV's
`dispatcher_router_initialized` field describes the dispatcher for that row;
the fallback row does not imply the separate router candidate has been destroyed.
In a subsequent automatic run, a disabled router is never created.

The example's eager submission/validation schedule differs from the archived
nccl-tests sweeps. Its calibration records apply to the measured fallback and
workload, not to a different process layout or CUDA Graph execution. No new
peak-bandwidth claim is inferred from these smoke/calibration runs.

## Lifetime and graph contract

Adaptive dispatch is currently eager-only on the CPU-router path. It rejects
capture rather than recording GPU work without the required CPU publication.
The low-level pipeline retains explicit `beginCapture/endCapture` and
`beginReplay/endReplay` APIs. Callers using those APIs must publish fresh CPU
jobs, launch all rank graphs, synchronize their streams, and finish replay
before reusing buffers. The supplied graph test exercises queue wrap and mixed
collectives; it is not an implicit graph adapter for the dispatcher.

Calls are serialized on stable per-device streams. Buffers and producer events
must outlive outstanding work. Synchronize before destroying a dispatcher or
changing buffer ownership. Initialization errors and numerical failures propagate;
the component does not silently retry partially executed collectives elsewhere.

The low-level module retains DMA and resident-GEMM experiments, but automatic
selection uses only the validated SM transport. The historical DMA medium-size
graph issue remains outside that path. The current module is single-node and
single-process; it is separate from the existing MPI two-node CPU-router program.

## Checks

```bash
./build/uccl_lite_router_policy_check
./build/uccl_lite_dispatcher_check
./build/uccl_lite_cpu_bf16_check
./build/uccl_lite_router_core_check
./build/uccl_lite_primitives_check
./build/uccl_lite_primitive_example
./build/uccl_lite_router_check --gpus 4
./build/uccl_lite_router_graph_check --gpus 4
./build/uccl_lite_local_collectives_check
```

See [validation](validation.md) for the four-L4 checks and the current eight-L4
rerun. The [paired local/router calibration](../benchmarks/data/local_fallback_8l4_20261110_v3/README.md)
uses this port, a UCCL Lite fallback, and fresh automatic-selection replays.
The [NCCL comparison](../benchmarks/data/cpu_router_8l4_20261110/README.md)
is retained as a separate control; its profiles cannot select a backend against
the new fallback. Both differ from the older no-P2P primitive archive.
