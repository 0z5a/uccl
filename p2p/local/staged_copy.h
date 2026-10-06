#pragma once

#include "cuda_utils.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace dietgpu { class StackDeviceMemory; }

namespace uccl::local {

enum class DataType : uint32_t { Bytes = 0, Float16 = 1, BFloat16 = 2, Float32 = 3 };
enum class CopyMode { Raw, Zip };

struct Ticket {
  uint32_t slot = UINT32_MAX;
  uint64_t generation = 0;
  explicit operator bool() const { return slot != UINT32_MAX; }
};

struct Frame {
  uint64_t version = 1;
  uint64_t codec_id = 0;
  uint64_t dtype = 0;
  uint64_t raw_bytes = 0;
  uint64_t payload_bytes = 0;
  uint64_t request = 0;
  uint64_t slot = 0;
  uint64_t generation = 0;
  uint64_t destination_capacity = 0;
  uint64_t logical_destination_offset = 0;
};
static_assert(sizeof(Frame) == 80);

struct CopyStats {
  Frame frame;
  bool encoded = false;
  bool raw_fallback = false;
  size_t d2h_bytes = 0;
  size_t h2d_bytes = 0;
  uint64_t cpu_submit_ns = 0;
  float encode_ms = 0;
  float d2h_ms = 0;
  float h2d_ms = 0;
  float decode_ms = 0;
};

// Internal local-copy primitive. Callers own tensor allocations until source
// completion and consumption; no NIC or peer device-memory IPC is required.
class StagedCopy {
 public:
  StagedCopy(int source, int target, size_t maximum_bytes, unsigned slots,
             size_t chunk_bytes, CopyMode mode);
  ~StagedCopy();
  StagedCopy(const StagedCopy&) = delete;
  StagedCopy& operator=(const StagedCopy&) = delete;

  // A full queue returns an invalid ticket; it never grows its buffers.
  Ticket submit(const void* source, void* target, size_t bytes,
                size_t target_capacity, DataType dtype, cudaStream_t producer);
  bool wait(Ticket ticket, cudaStream_t consumer, CopyStats* stats = nullptr);
  // Record after the consumer has submitted its work, not before consumption.
  bool release(Ticket ticket, cudaStream_t consumer);
  bool source_complete(Ticket ticket);
  size_t pinned_bytes() const { return (capacity_ + sizeof(Metadata)) * slots_.size(); }
  size_t codec_workspace_bytes() const { return workspace_; }
  cudaStream_t source_copy_stream() const { return copy_out_; }
  cudaStream_t target_copy_stream() const { return copy_in_; }

 private:
  struct Metadata { uint32_t bytes = 0, words = 0; uint8_t success = 0; };
  struct Slot {
    void *host = nullptr, *encoded_source = nullptr, *encoded_target = nullptr;
    Metadata *metadata = nullptr, *source_metadata = nullptr, *target_metadata = nullptr;
    cudaEvent_t ready{}, encoded{}, encode_begin{}, encode_end{};
    cudaEvent_t d2h_begin{}, d2h_end{}, h2d_begin{}, h2d_end{};
    cudaEvent_t decode_begin{}, decode_end{}, done{}, consumed{};
    std::vector<cudaEvent_t> chunks;
    uint64_t generation = 0;
    bool issued = false, waited = false, returned = false;
    CopyStats stats;
  };
  Slot* resolve(Ticket ticket);
  int source_, target_;
  size_t maximum_, capacity_, chunk_, workspace_;
  CopyMode mode_;
  cudaStream_t encode_{}, copy_out_{}, copy_in_{};
  std::unique_ptr<dietgpu::StackDeviceMemory> encoder_, decoder_;
  std::vector<Slot> slots_;
  std::mutex mutex_;
  uint64_t next_request_ = 0;
};

}  // namespace uccl::local
