#include "dietgpu_decompress.h"

#ifdef USE_DIETGPU
#include <limits>
#include <vector>

struct DecodeResult {
  uint32_t elements;
  uint8_t success;
};

// Allocate status storage once: per-request cudaMalloc/cudaMallocHost would
// synchronize unrelated work and defeat asynchronous decoder progress.
class DecompressPool {
 public:
  DecompressPool(int device, uint32_t slots) : device(device), events(slots) {
    CodecContextScope context(device);
    GPU_RT_CHECK(gpuMalloc(&device_results, slots * sizeof(DecodeResult)));
    GPU_RT_CHECK(gpuMallocHost(&host_results, slots * sizeof(DecodeResult)));
    free_slots.reserve(slots);
    for (uint32_t i = 0; i < slots; ++i) {
      GPU_RT_CHECK(gpuEventCreateWithFlags(&events[i], gpuEventDisableTiming));
      free_slots.push_back(i);
    }
  }

  ~DecompressPool() {
    CodecContextScope context(device);
    for (auto event : events) GPU_RT_CHECK(gpuEventDestroy(event));
    GPU_RT_CHECK(gpuFree(device_results));
    GPU_RT_CHECK(gpuFreeHost(host_results));
  }

  uint32_t acquire() {
    std::lock_guard<std::mutex> lock(mutex);
    if (free_slots.empty()) return std::numeric_limits<uint32_t>::max();
    auto slot = free_slots.back();
    free_slots.pop_back();
    return slot;
  }

  void release(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mutex);
    free_slots.push_back(slot);
  }

  int device;
  DecodeResult* device_results = nullptr;
  DecodeResult* host_results = nullptr;
  std::vector<gpuEvent_t> events;

 private:
  std::mutex mutex;
  std::vector<uint32_t> free_slots;
};

namespace {
class DietGPUDecompressRequest final : public DecompressRequest {
 public:
  DietGPUDecompressRequest(std::shared_ptr<DecompressPool> pool, uint32_t slot,
                           uint32_t expected)
      : pool_(std::move(pool)), slot_(slot), expected_(expected) {}

  ~DietGPUDecompressRequest() override {
    // Keep the lease on connection shutdown until all decoder/status work is
    // done. A GPU error aborts before the slot can be returned to the pool.
    wait();
    pool_->release(slot_);
  }

  DecodeResult* device_result() { return pool_->device_results + slot_; }

  void record(gpuStream_t stream,
              dietgpu::FloatDecompressStatus const& status) {
    submitted_ok_ = status.error == dietgpu::FloatDecompressError::None;
    GPU_RT_CHECK(gpuMemcpyAsync(pool_->host_results + slot_, device_result(),
                                sizeof(DecodeResult), gpuMemcpyDeviceToHost,
                                stream));
    GPU_RT_CHECK(gpuEventRecord(pool_->events[slot_], stream));
  }

  DecompressStatus poll() override {
    CodecContextScope context(pool_->device);
    auto status = gpuEventQuery(pool_->events[slot_]);
    if (status == gpuErrorNotReady) return DecompressStatus::kPending;
    GPU_RT_CHECK(status);
    return result();
  }

  DecompressStatus wait() override {
    CodecContextScope context(pool_->device);
    GPU_RT_CHECK(gpuEventSynchronize(pool_->events[slot_]));
    return result();
  }

 private:
  DecompressStatus result() const {
    return submitted_ok_ && pool_->host_results[slot_].success &&
                   pool_->host_results[slot_].elements == expected_
               ? DecompressStatus::kSuccess
               : DecompressStatus::kFailed;
  }

  std::shared_ptr<DecompressPool> pool_;
  uint32_t slot_;
  uint32_t expected_;
  bool submitted_ok_ = false;
};
}  // namespace

std::shared_ptr<DecompressPool> make_decompress_pool(int device,
                                                     uint32_t slots) {
  return std::make_shared<DecompressPool>(device, slots);
}

std::unique_ptr<DecompressRequest> make_decompress_request(
    int device, dietgpu::StackDeviceMemory& scratch, gpuStream_t stream,
    std::mutex& codec_mutex, std::shared_ptr<DecompressPool> const& pool,
    void const* input, void* output, dietgpu::FloatType float_type,
    uint32_t elements) {
  if (!input || !output || !stream || elements == 0 ||
      float_type == dietgpu::FloatType::kUndefined)
    return nullptr;
  std::lock_guard<std::mutex> lock(codec_mutex);
  CodecContextScope context(device);
  auto slot = pool->acquire();
  if (slot == std::numeric_limits<uint32_t>::max()) return nullptr;
  auto request =
      std::make_unique<DietGPUDecompressRequest>(pool, slot, elements);
  auto* result = request->device_result();
  GPU_RT_CHECK(gpuMemsetAsync(result, 0, sizeof(DecodeResult), stream));
  dietgpu::FloatDecompressConfig cfg;
  cfg.floatType = float_type;
  cfg.useChecksum = false;
  cfg.is16ByteAligned = true;
  void const* comp_in[1] = {input};
  void* decomp_out[1] = {output};
  uint32_t out_capacity[1] = {elements};
  auto status = dietgpu::floatDecompress(scratch, cfg, 1, comp_in, decomp_out,
                                         out_capacity, &result->success,
                                         &result->elements, stream);
  request->record(stream, status);
  return request;
}
#endif
