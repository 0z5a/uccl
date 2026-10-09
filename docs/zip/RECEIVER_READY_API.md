# Receiver-ready compressed write API

The model E2E harness must wait for the destination decoder rather than a
sender-local data CQE. The existing `write_async` and `poll_async` behavior is
preserved. This candidate adds a distinct `write_compressed_async` method.

```python
ok, transfer = endpoint.write_compressed_async(
    connection, registration, tensor.data_ptr(), tensor.nbytes, peer_metadata
)
assert ok
while True:
    ok, done = endpoint.poll_async(transfer)
    if done:
        assert ok
        break
```

Admission requires a negotiated compressed RDMA window, an eligible dtype,
matching prepared extent, and enough local/remote archive capacity. Unsupported
requests return failure without silently using raw or IPC transfers. Posting
retains the serialized source-buffer lifetime through local NIC completion;
the transfer handle remains pending until the exact decoder completion token
has been consumed. Decoder failure is returned as `(False, True)`. If another
request fails, outstanding requests still drain their own decoder acknowledgments
before publishing the terminal connection failure, so callers can release their
destination buffers safely.

`compressed_counters(connection)` returns `submitted`, `ready`, `failed`,
`archive_bytes` and `inflight`. Counters accumulate over the connection; an E2E
case compares snapshots after draining all handles. A valid compressed case
requires submitted delta equal to its writes, ready delta equal to submitted,
failure delta zero and inflight zero. The archive byte count records the actual
split/encode output. Counters queried during a post are not a coherent snapshot.

The wrapper returns posting status separately from the 32-bit request ID. A
tracker ID with its high bit set must not be interpreted as a negative posting
error. The per-request completion token is independent of that ID.

| Check | Result | Speedup |
|---|---|---|
| Ordinary raw completion interface | Source semantics preserved | N/A |
| Explicit compressed admission and decoder-ready polling | Full Linux/CUDA build and Python bindings pass; hardware execution pending | Not measured |
| Decoder failure with other outstanding writes | Own ACK drain precedes terminal failure; device regression pending | N/A |
| Model E2E | Still pending; this interface alone is not an E2E result | Not measured |

`p2p/tests/test_zip_receiver_ready.py` prepares a distributed RDMA stage using
equal writer/receiver groups and a connection to every receiver. C worker
threads each post Q distinct BF16 inputs and destinations, with varied raw bits
and no timing sleeps before validation. Zip requires the explicit decoder-ready
method and verifies counters; raw uses the original method in a separate process.
The phase timer includes the final control barrier and records validation time
separately. H100 assignment and unique worker/GPU binding must be checked by the run
controller; the script also rejects other GPU models. Runtime metadata and full
logs remain private. This stage is prepared, has not run on hardware and must
not be reported as model E2E.

For 8 assigned H100 GPUs, the high case is C=128, Q=16, 16 MiB. Its input or
output tensors require 32 GiB per active writer/receiver. A configured 16 GiB
total archive pool plus the existing source/scratch buffers gives a workload
peak near 56 GiB, excluding runtime overhead. Root must approve that budget
before launch. Lower C/Q cases come first; no stage is admitted from a guessed
memory capacity. Large 64/256 MiB requests use smaller C/Q combinations that
fit the same budget. Full RL and vLLM PD model tests still follow the protocol
and barrier correctness gates. [Build evidence](BUILD_VALIDATION.md) is separate
from those pending execution results.
