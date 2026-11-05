#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace uccl_lite::host_staged {

// Independent implementation of the communication mechanisms described in
// ThunderEP, arXiv:2609.40093v1, sections 4.2--4.5. Not the authors' source.
enum class Transfer { DmaPrefill, SmDecode };
enum class Collective { AllGather, ReduceScatter };

struct Options {
  Transfer transfer = Transfer::DmaPrefill;
  size_t maxShardElements = 1 << 20;
  // BF16 elements per staging bank. Two banks are allocated per source/target.
  size_t chunkElements = 1 << 18;
};

struct Binding {
  void const* input = nullptr;
  void* output = nullptr;
  cudaStream_t stream = nullptr;
  // DMA only. Events already recorded on this GPU, one for AG or one per
  // destination for RS. They replace (not supplement) input readiness on
  // stream. This lets a destination's partial output depart before other
  // experts finish. Empty means all input is ready in stream order.
  std::vector<cudaEvent_t> inputReady;
};

// Called synchronously by the submitting host, with the receiving GPU current.
// Enqueue computation on the supplied compute stream; do not block the host.
// That stream waits only for this sender's entire shard, not other senders.
// It is separate from all copy streams. The caller can record an event here
// and pass it as the corresponding ReduceScatter inputReady entry.
using GatherConsumer =
    std::function<void(int receiver, int sender, void const* shard,
                       size_t elements, cudaStream_t computeStream)>;
using ScatterConsumer = std::function<void(
    int receiver, void const* shard, size_t elements, cudaStream_t stream)>;

// BF16, one process, 1..16 explicitly ordered GPUs; no peer memory access,
// NCCL, CPU reduction or CPU packet-progress worker. Explicitly constructed:
// this reference never changes the measurement-gated CPU-router policy.
//
// Calls on an instance must be externally serialized. All ranks participate in
// identical collective order. Buffers/streams/events outlive queued work.
// AG input: shardElements; output: ranks*shardElements. RS is the reverse.
// Supports out-of-place and NCCL-style in-place placement (AG own output shard,
// RS own input shard). Other overlapping ranges are rejected.
class EpTransport {
 public:
  EpTransport(std::vector<int> devices, Options options = {});
  ~EpTransport();
  EpTransport(EpTransport const&) = delete;
  EpTransport& operator=(EpTransport const&) = delete;

  // Completion is joined into every Binding.stream. DMA additionally joins
  // callbacks, so subsequent operations on that stream may reuse all inputs.
  // Explicit inputReady can bypass that join for the next combine, enabling
  // dispatch -> per-sender computation -> immediate per-destination return.
  void enqueue(Collective op, size_t shardElements,
               std::vector<Binding> const& bindings,
               GatherConsumer const& onGather = {},
               ScatterConsumer const& onReduced = {});

  // SM only: captures this rank's kernel onto stream. Capture ALL ranks with
  // the same operation sequence, then launch ALL graphs before synchronizing.
  // Ordinary enqueue uses this same kernel. The round/bank is selected on the
  // device: graph replay needs no host publication, graph edit or reset.
  void enqueueRank(int rank, Collective op, size_t shardElements,
                   Binding const& binding);
  void synchronize();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite::host_staged
