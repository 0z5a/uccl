#include "staged_copy.h"
#include "dietgpu/float/GpuFloatCodec.h"
#include <algorithm>
#include <cassert>
#include <chrono>

namespace uccl::local {
namespace {
size_t word_size(DataType dtype) {
  return dtype == DataType::Float32 ? 4 : dtype == DataType::Bytes ? 1 : 2;
}
void event(cudaEvent_t* value, bool timing = false) {
  UCCL_LOCAL_CUDA(cudaEventCreateWithFlags(value, timing ? cudaEventDefault : cudaEventDisableTiming));
}
}  // namespace

StagedCopy::StagedCopy(int source, int target, size_t maximum_bytes,
                       unsigned slots, size_t chunk_bytes, CopyMode mode)
    : source_(source), target_(target), maximum_(maximum_bytes),
      capacity_(maximum_bytes), chunk_(chunk_bytes), workspace_(0), mode_(mode), slots_(slots) {
  assert(maximum_ && maximum_ <= (64u << 20) && slots && chunk_);
  if (mode_ == CopyMode::Zip) {
    for (auto type : {dietgpu::FloatType::kFloat16, dietgpu::FloatType::kBFloat16,
                      dietgpu::FloatType::kFloat32}) {
      const unsigned width = type == dietgpu::FloatType::kFloat32 ? 4 : 2;
      capacity_ = std::max(capacity_, size_t(dietgpu::getMaxFloatCompressedSize(type, (maximum_ + width - 1) / width)));
    }
    workspace_ = maximum_ * 3 + (1 << 20);
  }
  assert((capacity_ + sizeof(Metadata)) * slots <= (256u << 20));
  {
    DeviceScope device(source_);
    UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&encode_, cudaStreamNonBlocking));
    UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&copy_out_, cudaStreamNonBlocking));
    if (mode_ == CopyMode::Zip) encoder_ = std::make_unique<dietgpu::StackDeviceMemory>(source_, workspace_);
    for (auto& slot : slots_) {
      UCCL_LOCAL_CUDA(cudaHostAlloc(&slot.host, capacity_, cudaHostAllocPortable));
      UCCL_LOCAL_CUDA(cudaHostAlloc(&slot.metadata, sizeof(Metadata), cudaHostAllocPortable));
      if (mode_ == CopyMode::Zip) {
        UCCL_LOCAL_CUDA(cudaMalloc(&slot.encoded_source, capacity_));
        UCCL_LOCAL_CUDA(cudaMalloc(&slot.source_metadata, sizeof(Metadata)));
      }
      event(&slot.ready); event(&slot.encoded);
      event(&slot.encode_begin, true); event(&slot.encode_end, true);
      event(&slot.d2h_begin, true); event(&slot.d2h_end, true);
      slot.chunks.resize((capacity_ + chunk_ - 1) / chunk_);
      for (auto& chunk : slot.chunks) event(&chunk);
    }
  }
  {
    DeviceScope device(target_);
    UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&copy_in_, cudaStreamNonBlocking));
    if (mode_ == CopyMode::Zip) decoder_ = std::make_unique<dietgpu::StackDeviceMemory>(target_, workspace_);
    for (auto& slot : slots_) {
      if (mode_ == CopyMode::Zip) {
        UCCL_LOCAL_CUDA(cudaMalloc(&slot.encoded_target, capacity_));
        UCCL_LOCAL_CUDA(cudaMalloc(&slot.target_metadata, sizeof(Metadata)));
      }
      event(&slot.h2d_begin, true); event(&slot.h2d_end, true);
      event(&slot.decode_begin, true); event(&slot.decode_end, true);
      event(&slot.done); event(&slot.consumed);
    }
  }
}

StagedCopy::Slot* StagedCopy::resolve(Ticket ticket) {
  if (ticket.slot >= slots_.size()) return nullptr;
  auto& slot = slots_[ticket.slot];
  return slot.issued && !slot.returned && slot.generation == ticket.generation ? &slot : nullptr;
}

Ticket StagedCopy::submit(const void* source, void* target, size_t bytes,
                          size_t target_capacity, DataType dtype, cudaStream_t producer) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (bytes > maximum_ || bytes > target_capacity ||
      static_cast<uint32_t>(dtype) > static_cast<uint32_t>(DataType::Float32) ||
      (bytes && (!source || !target))) return {};
  unsigned index = 0;
  {
    DeviceScope device(target_);
    for (; index < slots_.size(); ++index) {
      auto& slot = slots_[index];
      if (!slot.issued) break;
      if (slot.returned) {
        const auto status = cudaEventQuery(slot.consumed);
        if (status == cudaSuccess) break;
        if (status != cudaErrorNotReady) UCCL_LOCAL_CUDA(status);
      }
    }
  }
  if (index == slots_.size()) return {};
  auto& slot = slots_[index];
  slot.issued = true; slot.returned = false; slot.waited = false;
  slot.stats = {};
  slot.stats.frame.dtype = static_cast<uint64_t>(dtype);
  slot.stats.frame.raw_bytes = slot.stats.frame.payload_bytes = bytes;
  slot.stats.frame.request = next_request_++;
  slot.stats.frame.slot = index;
  slot.stats.frame.generation = ++slot.generation;
  slot.stats.frame.destination_capacity = target_capacity;
  const auto begin = std::chrono::steady_clock::now();
  const size_t width = word_size(dtype);
  const bool encode = mode_ == CopyMode::Zip && bytes && dtype != DataType::Bytes &&
                      bytes % width == 0 && reinterpret_cast<uintptr_t>(source) % width == 0 &&
                      reinterpret_cast<uintptr_t>(target) % width == 0;
  slot.stats.raw_fallback = mode_ == CopyMode::Zip && !encode;
  const void* payload = source;
  {
    DeviceScope device(source_);
    UCCL_LOCAL_CUDA(cudaEventRecord(slot.ready, producer));
    if (encode) {
      UCCL_LOCAL_CUDA(cudaStreamWaitEvent(encode_, slot.ready, 0));
      dietgpu::FloatCompressConfig config;
      config.floatType = static_cast<dietgpu::FloatType>(dtype);
      config.is16ByteAligned = (reinterpret_cast<uintptr_t>(source) & 15) == 0;
      const uint32_t words = bytes / width;
      const void* inputs[] = {source};
      void* outputs[] = {slot.encoded_source};
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.encode_begin, encode_));
      dietgpu::floatCompress(*encoder_, config, 1, inputs, &words, outputs, &slot.source_metadata->bytes, encode_);
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.encode_end, encode_));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(&slot.metadata->bytes, &slot.source_metadata->bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost, encode_));
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.encoded, encode_));
      UCCL_LOCAL_CUDA(cudaEventSynchronize(slot.encoded));
      assert(slot.metadata->bytes && slot.metadata->bytes <= capacity_);
      slot.stats.encoded = true;
      slot.stats.d2h_bytes += sizeof(uint32_t);
      if (slot.metadata->bytes < bytes) {
        slot.stats.frame.payload_bytes = slot.metadata->bytes;
        payload = slot.encoded_source;
      } else {
        slot.stats.raw_fallback = true;
      }
      UCCL_LOCAL_CUDA(cudaStreamWaitEvent(copy_out_, slot.encoded, 0));
    } else {
      UCCL_LOCAL_CUDA(cudaStreamWaitEvent(copy_out_, slot.ready, 0));
    }
    UCCL_LOCAL_CUDA(cudaEventRecord(slot.d2h_begin, copy_out_));
  }
  const size_t payload_bytes = slot.stats.frame.payload_bytes;
  const bool compressed = slot.stats.encoded && !slot.stats.raw_fallback;
  slot.stats.frame.codec_id = compressed ? 1 : 0;
  void* received = compressed ? slot.encoded_target : target;
  for (size_t offset = 0, chunk = 0; offset < payload_bytes; offset += chunk_, ++chunk) {
    const size_t amount = std::min(chunk_, payload_bytes - offset);
    {
      DeviceScope device(source_);
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(static_cast<char*>(slot.host) + offset,
                                    static_cast<const char*>(payload) + offset, amount,
                                    cudaMemcpyDeviceToHost, copy_out_));
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.chunks[chunk], copy_out_));
    }
    {
      DeviceScope device(target_);
      UCCL_LOCAL_CUDA(cudaStreamWaitEvent(copy_in_, slot.chunks[chunk], 0));
      if (!offset) UCCL_LOCAL_CUDA(cudaEventRecord(slot.h2d_begin, copy_in_));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(static_cast<char*>(received) + offset,
                                    static_cast<char*>(slot.host) + offset, amount,
                                    cudaMemcpyHostToDevice, copy_in_));
    }
  }
  {
    DeviceScope device(source_);
    UCCL_LOCAL_CUDA(cudaEventRecord(slot.d2h_end, copy_out_));
  }
  {
    DeviceScope device(target_);
    if (!payload_bytes) {
      UCCL_LOCAL_CUDA(cudaStreamWaitEvent(copy_in_, slot.d2h_end, 0));
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.h2d_begin, copy_in_));
    }
    UCCL_LOCAL_CUDA(cudaEventRecord(slot.h2d_end, copy_in_));
    if (compressed) {
      dietgpu::FloatDecompressConfig config;
      config.floatType = static_cast<dietgpu::FloatType>(dtype);
      config.is16ByteAligned = (reinterpret_cast<uintptr_t>(target) & 15) == 0;
      const void* inputs[] = {slot.encoded_target};
      void* outputs[] = {target};
      const uint32_t words = bytes / width;
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.decode_begin, copy_in_));
      auto status = dietgpu::floatDecompress(*decoder_, config, 1, inputs, outputs, &words,
                                            &slot.target_metadata->success, &slot.target_metadata->words, copy_in_);
      assert(status.error == dietgpu::FloatDecompressError::None);
      UCCL_LOCAL_CUDA(cudaEventRecord(slot.decode_end, copy_in_));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(&slot.metadata->success, &slot.target_metadata->success, sizeof(uint8_t), cudaMemcpyDeviceToHost, copy_in_));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(&slot.metadata->words, &slot.target_metadata->words, sizeof(uint32_t), cudaMemcpyDeviceToHost, copy_in_));
      slot.stats.d2h_bytes += sizeof(uint8_t) + sizeof(uint32_t);
    }
    UCCL_LOCAL_CUDA(cudaEventRecord(slot.done, copy_in_));
  }
  slot.stats.d2h_bytes += payload_bytes;
  slot.stats.h2d_bytes = payload_bytes;
  slot.stats.cpu_submit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
  return {index, slot.generation};
}

bool StagedCopy::wait(Ticket ticket, cudaStream_t consumer, CopyStats* stats) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* slot = resolve(ticket);
  if (!slot) return false;
  {
    DeviceScope device(target_);
    UCCL_LOCAL_CUDA(cudaEventSynchronize(slot->done));
    if (slot->stats.encoded && !slot->stats.raw_fallback &&
        (slot->metadata->success != 1 || slot->metadata->words != slot->stats.frame.raw_bytes / word_size(static_cast<DataType>(slot->stats.frame.dtype)))) return false;
    UCCL_LOCAL_CUDA(cudaStreamWaitEvent(consumer, slot->done, 0));
    if (stats) {
      UCCL_LOCAL_CUDA(cudaEventElapsedTime(&slot->stats.h2d_ms, slot->h2d_begin, slot->h2d_end));
      if (slot->stats.encoded && !slot->stats.raw_fallback)
        UCCL_LOCAL_CUDA(cudaEventElapsedTime(&slot->stats.decode_ms, slot->decode_begin, slot->decode_end));
    }
  }
  if (stats) {
    DeviceScope device(source_);
    UCCL_LOCAL_CUDA(cudaEventSynchronize(slot->d2h_end));
    UCCL_LOCAL_CUDA(cudaEventElapsedTime(&slot->stats.d2h_ms, slot->d2h_begin, slot->d2h_end));
    if (slot->stats.encoded)
      UCCL_LOCAL_CUDA(cudaEventElapsedTime(&slot->stats.encode_ms, slot->encode_begin, slot->encode_end));
    *stats = slot->stats;
  }
  slot->waited = true;
  return true;
}

bool StagedCopy::release(Ticket ticket, cudaStream_t consumer) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* slot = resolve(ticket);
  if (!slot || !slot->waited) return false;
  DeviceScope device(target_);
  UCCL_LOCAL_CUDA(cudaEventRecord(slot->consumed, consumer));
  slot->returned = true;
  return true;
}

bool StagedCopy::source_complete(Ticket ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* slot = resolve(ticket);
  if (!slot) return false;
  DeviceScope device(source_);
  const auto status = cudaEventQuery(slot->d2h_end);
  if (status != cudaSuccess && status != cudaErrorNotReady) UCCL_LOCAL_CUDA(status);
  return status == cudaSuccess;
}

StagedCopy::~StagedCopy() {
  {
    DeviceScope device(source_);
    UCCL_LOCAL_CUDA(cudaStreamSynchronize(encode_));
    UCCL_LOCAL_CUDA(cudaStreamSynchronize(copy_out_));
  }
  {
    DeviceScope device(target_);
    UCCL_LOCAL_CUDA(cudaStreamSynchronize(copy_in_));
    for (auto& slot : slots_) {
      assert(!slot.issued || slot.returned);
      if (slot.returned) UCCL_LOCAL_CUDA(cudaEventSynchronize(slot.consumed));
      for (auto value : {slot.h2d_begin, slot.h2d_end, slot.decode_begin, slot.decode_end, slot.done, slot.consumed})
        UCCL_LOCAL_CUDA(cudaEventDestroy(value));
      if (slot.encoded_target) UCCL_LOCAL_CUDA(cudaFree(slot.encoded_target));
      if (slot.target_metadata) UCCL_LOCAL_CUDA(cudaFree(slot.target_metadata));
    }
    decoder_.reset();
    UCCL_LOCAL_CUDA(cudaStreamDestroy(copy_in_));
  }
  {
    DeviceScope device(source_);
    for (auto& slot : slots_) {
      for (auto value : {slot.ready, slot.encoded, slot.encode_begin, slot.encode_end, slot.d2h_begin, slot.d2h_end})
        UCCL_LOCAL_CUDA(cudaEventDestroy(value));
      for (auto value : slot.chunks) UCCL_LOCAL_CUDA(cudaEventDestroy(value));
      if (slot.encoded_source) UCCL_LOCAL_CUDA(cudaFree(slot.encoded_source));
      if (slot.source_metadata) UCCL_LOCAL_CUDA(cudaFree(slot.source_metadata));
      UCCL_LOCAL_CUDA(cudaFreeHost(slot.host));
      UCCL_LOCAL_CUDA(cudaFreeHost(slot.metadata));
    }
    encoder_.reset();
    UCCL_LOCAL_CUDA(cudaStreamDestroy(encode_));
    UCCL_LOCAL_CUDA(cudaStreamDestroy(copy_out_));
  }
}

}  // namespace uccl::local
