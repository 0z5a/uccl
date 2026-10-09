# Submission ordering and request tracker review

Date: 2026-10-09. Local follow-up to `44f2fa1c`.

## Reproduced tracker failure

A CPU probe invokes the production `AtomicBitmapPacketTrackerMultiAck`, starting
near `UINT32_MAX`. Requests at the boundary report pending as expected. The next
requests, numbered 0 and 1, report completion before any ACK. This can release
the compressed source lease before the NIC has consumed its bytes.

The cause is ordinary integer ordering in a modulo-32-bit window. The candidate
uses unsigned distance from the base for admission into the active window,
expected counts, packet sizes and enumeration. A retired request lies behind the
base within the modular half range. Sliding continues across the wrap instead
of stopping when the numeric base exceeds the numeric next ID.

Mutation is serialized: resetting a row and advancing its base occur in the
same transition as allocation or acknowledgement. Two callers cannot clear a
row after another transition has already reused it. Completion queries retain
atomic reads. The existing window has 65,536 entries; runs must stay within its
outstanding-request capacity. Unbounded admission is not qualified by this test.

| Case | Prior behavior | Candidate evidence | Speedup |
|---|---|---|---|
| Request 0/1 after 32-bit wrap | Premature completion reproduced without ACK | CPU probe reports pending | N/A |
| Eight requests spanning wrap | Incorrect window/accounting behavior | Exact pending IDs, bytes and two ACKs per request pass | N/A |
| Two-phase expected-count sentinel | Update can be rejected at wrap | No early completion; final third ACK completes | N/A |
| Concurrent ACK / expected-count update | Row reset/base advance can overlap | 16 threads, 131,072 production tracker transitions pass | N/A |
| Host ASan / UBSan / TSan | — | Regression passes | N/A |
| CUDA/RDMA source lifetime and performance | No new hardware result | Execution still pending | Not measured |

## Two-phase publication

Split failure is checked before any data post. Encode must return a nonzero
archive size between the posted prefix and reserved worst-case extent. On
failure, already posted prefix WRs drain before releasing its archive and slot.

After all local data CQEs, the sender marks its own descriptor data-complete and
attempts metadata publication explicitly. Completion of the final CQE can occur
before the expected-count update, so relying on a later CQE to trigger metadata
would leave the first attempt to incidental background progress.

Publishing and the `meta_pushed` transition hold the pending-descriptor lock
together. The posting caller and progress thread cannot both publish the same
descriptor. A failed post keeps the descriptor eligible for retry. One duplicate
descriptor could otherwise cause a second decoder to read an archive after the
first decoder ACK had already allowed it to be reused.

Known compressed transfer handles are issued only after local data completion.
Their receiver-ready status follows the retained descriptor and exact decoder
ACK, independently of a later tracker-window reuse. This remains a source-level
ordering argument until the actual data/control QP test runs.

## Handshake result

The previous handshake response retained `INVALID_PEER_ID`, and its negative
return was assigned to an unsigned variable before comparison. Connection setup
also discarded `build_connect` failure and returned an apparently valid local
connection. The candidate returns the assigned receiver peer ID, keeps the
setup result signed and propagates failure to `Endpoint.connect` before storing
the connection. Actual failure/timeout and successful multi-peer handshakes
still require the Linux/CUDA/RDMA run.

These checks do not complete the full model E2E requirement. The frozen
`70213197` native prerequisite and the high-concurrency model runs remain pending.
