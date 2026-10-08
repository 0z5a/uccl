# Existing barrier PR: source review and qualification gaps

Date: 2026-10-08. Reviewed head:
`88b9c2e51b6beec7eee8165bfce5b74ea6bb9d81` of
[#1052](https://github.com/uccl-project/uccl/pull/1052).
This review concerns the existing owned barrier PR, not the still-unidentified
Camelia review target. No new hardware result is asserted here.

## Findings

| Priority | Finding | Required correction / evidence |
|---|---|---|
| P0 | Notification increments `barrier_signal_ptrs[rank] + thread_id`, never the peer pointer selected by the participating lane. A local row can become uniform without another rank arriving. | Notify each target rank's row at the sender's column. Freeze a stable per-call target rather than repeatedly reading a changing row. |
| P0 | The regression's `round` depends on `blockIdx.y`, while launch grid Y is always one. It never advances with the loop iteration. Once the first local announcement is visible, later iterations cannot detect early release. | Encode epoch and iteration in peer-visible progress. Intentionally delay one real rank and verify that later iterations cannot pass it. |
| P0 | Every process launches `nranks` blocks and derives rank from block X. The helper is instantiated with `MAX_RANKS=16` even when the runtime world has two or four ranks. This is not one process executing its own actual rank. | Pass `my_rank`, specialize for the real world size, and exchange genuine per-rank device pointers. Correct peer addressing would expose the mismatched pointer-table extent. |
| P1 | The final block rendezvous was removed. The 32-thread test does not cover a block whose additional warps advance before the polling warp. | Restore block-wide exit ordering and test at least 128 threads with payload visibility checks. |
| P1 | Signed counters and ordinary minimum/comparison ordering are not valid across wrap. Inactive lanes use a fixed signed maximum. | Use an explicit bounded modular comparison, neutral inactive-lane votes, and initialize a test near the wrap boundary. |

Pinned source:
[barrier implementation, lines 920–951](https://github.com/uccl-project/uccl/blob/88b9c2e51b6beec7eee8165bfce5b74ea6bb9d81/ep/include/ep_utils.cuh#L920),
[regression rank/round and template, lines 56–80](https://github.com/uccl-project/uccl/blob/88b9c2e51b6beec7eee8165bfce5b74ea6bb9d81/ep/tests/barrier_concurrent.cu#L56),
[launch shape, lines 198–202](https://github.com/uccl-project/uccl/blob/88b9c2e51b6beec7eee8165bfce5b74ea6bb9d81/ep/tests/barrier_concurrent.cu#L198).

## Scope of a corrected regression

One process must execute one actual rank on its assigned GPU. Handle exchange
and peer-visible payload/progress must cover those real ranks. The fixture
must exercise changing iteration tokens, multiple kernel launches, delayed
arrivals, additional warps and counter wrap. The expected calls must count the
actual local rank's work; they must not count virtual copies of every rank in
each process.

Independent epoch cells establish separate-group correctness. They do not
prove overlapping dispatch epochs that share the same cells. Conversely,
anonymous cumulative arrival counters do not identify which logical epoch's
payload has arrived. The shared-cell test and API/caller ordering contract must
be resolved explicitly before claiming the concurrent-epoch issue is closed.

The old published PASS counts are retained as historical outputs. The current
regression cannot use those counts to qualify cross-rank synchronization or the
full inter-node failure reported by #986. A four-GPU, same-node IPC run remains
necessary once the corrected fixture and required peer-atomic topology are
available; device pointers must not be reused as a cross-node transport.
