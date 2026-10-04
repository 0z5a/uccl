# Local device validation

Build native SM120 device code and exercise the production MSCCLPP FIFO without
an RDMA NIC, EFA, MPI, PyTorch, or NCCL. CUDA, a C++17 compiler, pthread, and
libnuma development files are the only dependencies.

```sh
make -C experimental/uccl_gin/tests/local SM=120 CUDA_HOME=/usr/local/cuda -j4
LOCAL_BIN=experimental/uccl_gin/tests/local/build/sm120
CUDA_VISIBLE_DEVICES=0 timeout 90s "$LOCAL_BIN/capabilities" --device 0
CUDA_VISIBLE_DEVICES=0 timeout 90s "$LOCAL_BIN/queue_smoke" --device 0 \
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
Use the outer timeout to bound CUDA initialization and teardown as well.

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
CUDA_VISIBLE_DEVICES=0 timeout 90s "$LOCAL_BIN/coop_flush" \
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
words verify that no participant returns before the final QUIET acknowledgement.
`--delay-us 50` holds that acknowledgement. Both layouts require exactly
`warps * iterations * queues` QUIETs for the fixed warp path.

Thread groups are supported; CTA groups must fail compilation. The adapter
accepts acquire ordering only: `--invalid-order 1 --bytes 0 --check-source-reuse 0`
expects a device trap in a separate process. Scalar flush is covered by PR1.

For larger synthetic BF16 activations, use `--batch-size 512 --hidden-size 2048`;
each warp splits that activation evenly across its 32 producers. The width
matches [Qwen3-30B-A3B's configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json).
These are communication fixtures, not model inference or HTTP concurrency tests.
`--warps 64 --batch-size 2048 --capacity 4096` exercises 2,048 producers with
256 KiB per producer. Participation must fit FIFO capacity in shared mode.

For a measured original-adapter comparison, compile this same fixture against
the pinned base headers with `EXTRA_DEVFLAGS=-DLOCAL_EXPECT_SCALAR_ADAPTER=1`.
Apply the identical signal acquire-load compilation prerequisite to both
adapters. The baseline keeps the original scalar flush body and must observe
`warps * 32 * iterations * queues` QUIETs. Keep separate baseline build outputs.

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
