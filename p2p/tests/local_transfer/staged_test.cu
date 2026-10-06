#include "../../local/staged_copy.h"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace uccl::local;

namespace {
constexpr size_t maximum = 1 << 20;
struct Buffer {
  void *source{}, *target{}, *oracle{};
  std::vector<uint8_t> expected;
  size_t bytes = 0;
  unsigned offset = 0;
  Ticket ticket;
};

void prepare(Buffer& buffer, uint64_t frame) {
  constexpr std::array<size_t, 16> sizes{
      0, 1, 7, 8, 15, 16, 31, 32, 511, 512, 513,
      4095, 4096, 4097, 65536, maximum};
  buffer.bytes = sizes[frame % sizes.size()];
  buffer.offset = frame % 5 == 0 ? 1 : frame % 5 == 1 ? 2 : 0;
  buffer.expected.resize(buffer.bytes);
  const unsigned pattern = (frame / sizes.size()) % 3;
  for (size_t i = 0; i < buffer.bytes; ++i) {
    // Repeated values exercise Zip; varying bits retain incompressible RAW.
    buffer.expected[i] = pattern == 0 ? 0 : pattern == 1 ? (i & 3) * 0x3c
                                                    : (i * 17 + (i >> 8) + frame) & 255;
  }
}

unsigned run(int source, int target, CopyMode mode, unsigned frames) {
  std::array<Buffer, 2> buffers;
  cudaStream_t producer{}, consumer{};
  {
    DeviceScope device(source);
    UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    for (auto& buffer : buffers) UCCL_LOCAL_CUDA(cudaMalloc(&buffer.source, maximum + 16));
  }
  {
    DeviceScope device(target);
    UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    for (auto& buffer : buffers) {
      UCCL_LOCAL_CUDA(cudaMalloc(&buffer.target, maximum + 16));
      UCCL_LOCAL_CUDA(cudaHostAlloc(&buffer.oracle, maximum, cudaHostAllocDefault));
    }
  }
  unsigned compressed = 0, fallback = 0;
  {
    StagedCopy copy(source, target, maximum, 2, 256 << 10, mode);
    const size_t pinned = copy.pinned_bytes();
    assert(!copy.submit(nullptr, nullptr, maximum + 1, maximum + 1,
                        DataType::Bytes, producer));
    for (unsigned base = 0; base < frames; base += buffers.size()) {
      for (unsigned i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        prepare(buffer, base + i);
        const auto dtype = static_cast<DataType>(1 + (base / 16 + i) % 3);
        {
          DeviceScope device(source);
          if (buffer.bytes) UCCL_LOCAL_CUDA(cudaMemcpyAsync(
              static_cast<char*>(buffer.source) + buffer.offset,
              buffer.expected.data(), buffer.bytes, cudaMemcpyHostToDevice, producer));
        }
        const Ticket previous = buffer.ticket;
        buffer.ticket = copy.submit(static_cast<char*>(buffer.source) + buffer.offset,
                                    static_cast<char*>(buffer.target) + buffer.offset,
                                    buffer.bytes, maximum, dtype, producer);
        assert(buffer.ticket);
        if (previous) assert(!copy.wait(previous, consumer));
      }
      // Completed DMA does not return credit until the consumer releases it.
      assert(!copy.submit(buffers[0].source, buffers[0].target, 16, maximum,
                          DataType::BFloat16, producer));
      for (auto& buffer : buffers) {
        {
          DeviceScope device(source);
          // Poison is ordered after D2H, independent of target consumption.
          UCCL_LOCAL_CUDA(cudaMemsetAsync(buffer.source, 0xa5, maximum + 16,
                                        copy.source_copy_stream()));
        }
        CopyStats stats;
        assert(copy.wait(buffer.ticket, consumer, &stats));
        assert(copy.source_complete(buffer.ticket));
        assert(stats.frame.generation == buffer.ticket.generation);
        assert(stats.frame.raw_bytes == buffer.bytes);
        assert(stats.h2d_bytes == stats.frame.payload_bytes);
        assert(stats.d2h_bytes == stats.h2d_bytes + (stats.encoded ? 4 : 0) +
                                   (stats.frame.codec_id ? 5 : 0));
        compressed += stats.frame.codec_id != 0;
        fallback += stats.raw_fallback;
        {
          DeviceScope device(target);
          if (buffer.bytes) UCCL_LOCAL_CUDA(cudaMemcpyAsync(
              buffer.oracle, static_cast<char*>(buffer.target) + buffer.offset,
              buffer.bytes, cudaMemcpyDeviceToHost, consumer));
        }
        assert(copy.release(buffer.ticket, consumer));
        assert(!copy.release(buffer.ticket, consumer));
        assert(!copy.wait(buffer.ticket, consumer));
      }
      {
        DeviceScope device(target);
        UCCL_LOCAL_CUDA(cudaStreamSynchronize(consumer));
      }
      {
        DeviceScope device(source);
        UCCL_LOCAL_CUDA(cudaStreamSynchronize(copy.source_copy_stream()));
      }
      for (const auto& buffer : buffers) {
        assert(!buffer.bytes || !std::memcmp(buffer.expected.data(), buffer.oracle,
                                            buffer.bytes));
      }
      assert(copy.pinned_bytes() == pinned);
    }
    assert(mode == CopyMode::Raw || (compressed && fallback));
    std::cout << "{\"source\":" << source << ",\"target\":" << target
              << ",\"mode\":\"" << (mode == CopyMode::Raw ? "raw" : "zip")
              << "\",\"frames\":" << frames << ",\"compressed_frames\":" << compressed
              << ",\"raw_fallback_frames\":" << fallback << ",\"pinned_bytes\":" << pinned
              << ",\"slots\":2,\"bitwise\":\"PASS\",\"credit\":\"PASS\",\"status\":\"PASS\"}";
  }
  {
    DeviceScope device(source);
    for (auto& buffer : buffers) UCCL_LOCAL_CUDA(cudaFree(buffer.source));
    UCCL_LOCAL_CUDA(cudaStreamDestroy(producer));
  }
  {
    DeviceScope device(target);
    for (auto& buffer : buffers) {
      UCCL_LOCAL_CUDA(cudaFree(buffer.target));
      UCCL_LOCAL_CUDA(cudaFreeHost(buffer.oracle));
    }
    UCCL_LOCAL_CUDA(cudaStreamDestroy(consumer));
  }
  return frames;
}
}  // namespace

int main() {
  int devices = 0;
  UCCL_LOCAL_CUDA(cudaGetDeviceCount(&devices));
  if (devices != 2) return 2;
  unsigned total = 0;
  std::cout << "{\"runs\":[";
  for (int direction = 0; direction < 2; ++direction) {
    for (auto mode : {CopyMode::Raw, CopyMode::Zip}) {
      if (total) std::cout << ',';
      total += run(direction, 1 - direction, mode, 10000);
    }
  }
  std::cout << "],\"total_frames\":" << total << ",\"status\":\"PASS\"}\n";
}
