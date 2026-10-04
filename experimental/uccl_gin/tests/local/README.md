# Local device validation

Build native SM120 device code and exercise the production MSCCLPP FIFO without
an RDMA NIC, EFA, MPI, PyTorch, or NCCL. CUDA, a C++17 compiler, pthread, and
libnuma development files are the only dependencies.

```sh
make -C experimental/uccl_gin/tests/local SM=120 CUDA_HOME=/usr/local/cuda -j4
LOCAL_BIN=experimental/uccl_gin/tests/local/build/sm120
CUDA_VISIBLE_DEVICES=0 "$LOCAL_BIN/capabilities" --device 0
CUDA_VISIBLE_DEVICES=0 "$LOCAL_BIN/queue_smoke" --device 0 \
  --producers 32 --capacity 512 --commands 100000 --rounds 3
```

Repeat on the second physical GPU, then run both instances concurrently with
separate logs and exit statuses. Device numbers are relative to each process's
`CUDA_VISIBLE_DEVICES`. Each fixture owns its FIFO and GPU pointers; peer access
is reported as a capability and is never enabled or required.

`capabilities` checks a real mapped-host GPU write and production FIFO allocation.
`queue_smoke` compares every decoded field, rejects duplicate/missing commands,
checks signed atomic values and the FIFO reserved-bit transformation, reuses the
FIFO across rounds, and tests scalar completion. Producers are limited to FIFO
capacity; vary `--producers` and `--capacity` for backpressure and wraparound.
The consumer starts before the producer kernel and has a 60-second deadline.
Wait for the process's own exit and record its status before releasing resources.

For a rootless libnuma prefix, pass `NUMA_INCLUDE_DIR` and `NUMA_LIBRARY_DIR`.
An SM90 compile comparison uses `make SM=90`; architecture-specific directories
prevent stale object reuse. Use `cuobjdump --list-elf` to inspect native code and
`ldd` to verify that no EFA/MPI/NCCL runtime is linked.

These tests cover SM120 compilation and the real GPU-to-host command/completion
path. The consumer supplies test completions. EFA source consumption, receiver
visibility, and distributed NCCL-EP dispatch/combine require the network testbed.

## Cooperative flush

Use real NCCL device headers (validated with 2.30.4); the local executable still
links no NCCL runtime and creates no communicator.

```sh
make -C experimental/uccl_gin/tests/local coop-tests compile-fail \
  SM=120 NCCL_INCLUDE_DIR=/path/to/nccl/include
CUDA_VISIBLE_DEVICES=0 "$LOCAL_BIN/coop_flush" \
  --via adapter --group warp --warps 8 --queues 32 --iterations 100 \
  --shared 1 --check-source-reuse 1
```

Each iteration publishes one WRITE per participant, calls the actual adapter
or standalone cooperative API, then immediately overwrites its source. A host
consumer copies real HBM into pinned memory on a separate nonblocking stream
and waits for its event before acknowledging the command. Streams, buffers,
events, and the worker's CUDA device are initialized before launching producers.
No peer pointer or host-staged production transport is introduced.

With `--shared 0`, each group has separate queues. Per-queue traces check that
participant WRITEs precede each drain marker. System-scope release/acquire status
words verify that no participant returns before any queue's QUIET acknowledgement;
the check runs after any artificial delay and before the consumer pops that slot.
`--delay-us 50 --delay-queue 0` holds an early queue's acknowledgement; the default
delayed queue is Q−1. Both layouts require exactly
`warps * iterations * queues` QUIETs for the fixed warp path.

After the entry rendezvous, member `r` drains queue indices `r + k * coop.size()`.
This covers every resource queue once, including counts larger than one warp;
the exit rendezvous waits for all members' assigned queues. With 32 queues,
a full warp drains one queue per member in parallel. Thread groups retain the
sequential all-queue behavior. Use Q=3/33/64 to test uneven and repeated assignments.
All group members must pass the same resource array and queue count.

`--channel-hint lane` (default) routes by lane index. `--channel-hint warp`
routes a full warp by its group index, matching one aspect of the production
global-channel affinity. `--num-lanes N` selects the production proxy-major
queue layout and must divide Q. It does not add host workers: this fixture
still has one consumer per device. An independently inverted layout checks
every WRITE's destination FIFO and exact per-FIFO WRITE/QUIET counts.
At G64/Q32, either hint assigns 64 writers per queue per iteration: lane hint
uses one member of each of 64 warps; warp hint uses all members of two warps.

Private groups test completion ordering. Changing shared G64/Q32 to private
also increases the worker's scanned FIFO count from 32 to 2,048; it cannot
isolate atomic contention. Use shared queues for the three-arm speed comparison.

Thread groups are supported; CTA groups must fail compilation. The adapter
accepts acquire ordering only: `--invalid-order 1 --bytes 0 --check-source-reuse 0`
expects a device trap in a separate process. Scalar flush is covered by PR1.

For larger synthetic BF16 activations, use `--batch-size 512 --hidden-size 2048`;
each warp splits that activation evenly across its 32 producers. The width
matches [Qwen3-30B-A3B's configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json).
These are communication fixtures, not model inference or HTTP concurrency tests.
`--warps 64 --batch-size 2048 --capacity 4096` exercises 2,048 producers with
256 KiB per producer. Participation must fit FIFO capacity in shared mode.

For the two-GPU high-concurrency case, launch two instances simultaneously with
relative `--device 0` and `--device 1` under `CUDA_VISIBLE_DEVICES=0,3` on the
validated host. Use `--warps 64 --queues 32 --batch-size 2048 --capacity 4096
--iterations 2 --rounds 3 --check-source-reuse 1` for each. Record both exit
statuses and logs; together they exercise 4,096 producers and 2 GiB of
source/destination allocation. This remains communication concurrency.

For a measured original-adapter comparison, compile this same fixture against
the pinned base headers with `EXTRA_DEVFLAGS=-DLOCAL_EXPECT_SCALAR_ADAPTER=1`.
Apply the identical signal acquire-load compilation prerequisite to both
adapters. The baseline keeps the original scalar flush body and must observe
`warps * 32 * iterations * queues` QUIETs. Keep separate baseline build outputs.

Optional CUDA timeline diagnostics follow NVIDIA's
[Nsight Systems skill](https://github.com/NVIDIA/TensorRT-LLM/blob/fc0876cfd6c5d661f186707857a4e5bfeb12bc6f/.claude/skills/perf-nsight-systems/SKILL.md).
Collect these separately from the unprofiled speed comparison:

```sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none --kill=none \
  -o coop_flush_diagnostic "$LOCAL_BIN/coop_flush" \
  --warps 8 --queues 32 --batch-size 128 --iterations 2 --rounds 1
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  coop_flush_diagnostic.nsys-rep
```

For host-stage diagnostics, use a separate run with `--diagnose-consumer 1`.
The JSON records callback wall time, copy/event submission, event synchronization,
full pattern checking, and time between adjacent callbacks. The three WRITE
stages are nested in `callback_ns`; do not add them to it. The inter-callback
time includes FIFO pop, scanning, polling and host scheduling, excludes the first
and last boundary, and is not a measurement of GPU idle. With diagnostics off,
zero fields mean uncollected. Main timing keeps diagnostics off and uses identical
fixture bytes and CLI options across Original, V1 and V2. ABC/CBA with seven rounds
and two discarded rounds gives ten warm observations from two independent
process starts per arm/device; report each wave as well as the pooled median.

The [design document](COOP_FLUSH_DESIGN.md) explains queue ownership and the
source-lifetime proof. Local WRITE slots are popped after copy/check; the real
proxy can pop WRITE on collection and must retire QUIET only after network
completion. A passing local oracle does not validate that production bookkeeping.

## EFA gate

The existing network microbench now accepts `--only coop-flush`. Eight complete
warps publish disjoint payloads with separate receiver completion slots, call
the actual adapter flush, and overwrite sources immediately. Receiver completion
and payload integrity are checked independently of local flush completion.

```sh
mpirun -np 2 --host HOST_A,HOST_B \
  experimental/uccl_gin/build/uccl_gin_microbench \
  --no-nccl --correctness-only --only coop-flush \
  --sizes 4096,65536,1048576,16777216
```

Use the provisioned matching EFA/MPI/CUDA/NCCL build environment. The device
kernel is included in `coop-tests` compilation. Its host/network execution and
full NCCL-EP dispatch/combine remain separate acceptance gates.
