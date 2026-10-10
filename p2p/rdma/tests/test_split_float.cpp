#include "dietgpu_decompress.h"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/StaticUtils.h"
#include "util/debug.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using dietgpu::FloatType;

struct Format {
  FloatType type;
  uint32_t word_bytes;
  uint32_t symbol_bytes;
};

constexpr std::array<Format, 5> formats{{
    {FloatType::kFloat16, 2, 2},
    {FloatType::kBFloat16, 2, 2},
    {FloatType::kFloat32, 4, 4},
    {FloatType::kFloat8E4M3FN, 1, 2},
    {FloatType::kFloat8E5M2, 1, 2},
}};
constexpr size_t kGuard = 16;
constexpr uint8_t kCanary = 0xa5;

void check_geometry() {
  for (auto const& format : formats) {
    for (uint32_t bytes : {4u, 60u, 16u << 20, (16u << 20) + 20,
                           64u << 20, 256u << 20}) {
      UCCL_CHECK_EQ(dietgpu::getFloatSymbolCountFromBytes(format.type, bytes),
                    bytes / format.symbol_bytes);
    }
  }
  std::cout << "{\"test\":\"split_geometry_host\",\"cases\":30,"
               "\"status\":\"PASS\"}\n";
}

void check_scratch_usage(cudaStream_t stream) {
  dietgpu::StackDeviceMemory scratch(0, 4096);
  auto first = scratch.alloc<uint8_t>(stream, 256);
  UCCL_CHECK_EQ(scratch.getMaxMemoryUsage(), 256u);
  {
    auto second = scratch.alloc<uint8_t>(stream, 512);
    UCCL_CHECK_EQ(scratch.getMaxMemoryUsage(), 768u);
    auto overflow = scratch.alloc<uint8_t>(stream, 8192);
    UCCL_CHECK_EQ(scratch.getMaxMemoryUsage(), 8960u);
  }
  UCCL_CHECK_EQ(scratch.getMaxMemoryUsage(), 8960u);
  std::cout << "{\"test\":\"scratch_peak_gpu\",\"status\":\"PASS\"}\n";
}

void check_guard(uint8_t const* allocation, size_t payload) {
  std::array<uint8_t, kGuard> before, after;
  CUDA_VERIFY(cudaMemcpy(before.data(), allocation, kGuard, cudaMemcpyDeviceToHost));
  CUDA_VERIFY(cudaMemcpy(after.data(), allocation + kGuard + payload, kGuard,
                         cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < kGuard; ++i)
    UCCL_CHECK(before[i] == kCanary && after[i] == kCanary);
}

void check_roundtrip(Format const& format, uint32_t bytes, int repeats = 0) {
  CodecContextScope context(0);
  uint32_t elements = bytes / format.word_bytes;
  uint32_t symbols = bytes / format.symbol_bytes;
  auto capacity = dietgpu::getMaxFloatCompressedSize(format.type, elements);
  std::vector<uint8_t> input(bytes), observed(bytes);
  for (uint32_t i = 0; i < bytes; ++i)
    input[i] = ((i * 1103515245u + 12345u) >> 16) & 255;
  std::array<uint32_t, 8> edges;
  switch (format.type) {
    case FloatType::kFloat16:
      edges = {0, 0x8000, 1, 0x7c00, 0xfc00, 0x7e01, 0x7c01, 0xffff};
      break;
    case FloatType::kBFloat16:
      edges = {0, 0x8000, 1, 0x7f80, 0xff80, 0x7fc1, 0x7f81, 0xffff};
      break;
    case FloatType::kFloat32:
      edges = {0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc00001,
               0x7f800001, 0xffffffff};
      break;
    default:
      edges = {0, 0x80, 1, 0x7c, 0xfc, 0x7f, 0xff, 0x7e};
      break;
  }
  for (size_t i = 0; i < edges.size() && i < elements; ++i)
    std::memcpy(input.data() + i * format.word_bytes, &edges[i], format.word_bytes);
  if (repeats) {
    // The benchmark uses BF16 normal data; correctness also covers raw bits.
    UCCL_CHECK(format.type == FloatType::kBFloat16);
    std::mt19937 generator(0);
    std::normal_distribution<float> normal(0, 1);
    for (uint32_t i = 0; i < elements; ++i) {
      float value = normal(generator);
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      uint16_t word = bits >> 16;
      std::memcpy(input.data() + i * 2, &word, sizeof(word));
    }
  }

  cudaStream_t stream;
  CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::array<cudaEvent_t, 5> events;
  for (auto& event : events) {
    CUDA_VERIFY(cudaEventCreate(&event));
    CUDA_VERIFY(cudaEventRecord(event, stream));
  }
  void* source;
  uint8_t* archive_storage;
  uint8_t* output_storage;
  uint32_t* sizes;
  uint8_t* success;
  CUDA_VERIFY(cudaMalloc(&source, bytes));
  CUDA_VERIFY(cudaMalloc(&archive_storage, capacity + 2 * kGuard));
  CUDA_VERIFY(cudaMalloc(&output_storage, bytes + 2 * kGuard));
  void* archive = archive_storage + kGuard;
  void* output = output_storage + kGuard;
  CUDA_VERIFY(cudaMalloc(&sizes, 2 * sizeof(uint32_t)));
  CUDA_VERIFY(cudaMalloc(&success, 1));
  CUDA_VERIFY(cudaMemsetAsync(archive_storage, kCanary, capacity + 2 * kGuard, stream));
  CUDA_VERIFY(cudaMemsetAsync(output_storage, kCanary, bytes + 2 * kGuard, stream));
  CUDA_VERIFY(cudaMemcpyAsync(source, input.data(), bytes,
                              cudaMemcpyHostToDevice, stream));

  {
    dietgpu::StackDeviceMemory scratch(0, size_t{bytes} * 4 + (1 << 20));
    dietgpu::FloatCompressConfig config;
    config.floatType = format.type;
    config.is16ByteAligned = true;
    dietgpu::FloatCompressSplitContext split(format.type);
    split.maxSize = bytes;
    split.params_dev = scratch.alloc<uintptr_t>(stream, 3);
    uintptr_t params[] = {reinterpret_cast<uintptr_t>(source), symbols,
                          reinterpret_cast<uintptr_t>(archive)};
    CUDA_VERIFY(cudaMemcpyAsync(split.params_dev.data(), params, sizeof(params),
                                cudaMemcpyHostToDevice, stream));
    CUDA_VERIFY(cudaStreamSynchronize(stream));
    for (int sample = repeats ? -3 : 0; sample < std::max(repeats, 1); ++sample) {
      // Poison each destination before the measured call to catch missed writes.
      CUDA_VERIFY(cudaMemsetAsync(output, kCanary, bytes, stream));
      CUDA_VERIFY(cudaMemsetAsync(sizes, 0, 2 * sizeof(uint32_t), stream));
      CUDA_VERIFY(cudaMemsetAsync(success, 0, 1, stream));
      CUDA_VERIFY(cudaStreamSynchronize(stream));
      auto start = std::chrono::steady_clock::now();
      CUDA_VERIFY(cudaEventRecord(events[0], stream));
      dietgpu::floatCompressSplitOneBatch(scratch, config, stream, split);
      CUDA_VERIFY(cudaEventRecord(events[1], stream));
      dietgpu::floatCompressEncodeOneBatch(scratch, config, split, sizes, stream);
      CUDA_VERIFY(cudaEventRecord(events[2], stream));
      UCCL_CHECK_EQ(split.compRowStride, dietgpu::roundUp(symbols, 16u));
      UCCL_CHECK_EQ(split.toComp_dev.num, split.compRowStride);
      split.histogram_dev.release();
      split.toComp_dev.release();

      uint32_t encoded_bytes;
      CUDA_VERIFY(cudaMemcpy(&encoded_bytes, sizes, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
      UCCL_CHECK(encoded_bytes > 0 && encoded_bytes <= capacity);
      void const* inputs[] = {archive};
      void* outputs[] = {output};
      CUDA_VERIFY(cudaEventRecord(events[3], stream));
      auto result = dietgpu::floatDecompress(scratch, config, 1, inputs, outputs,
                                             &elements, success, sizes + 1, stream);
      CUDA_VERIFY(cudaEventRecord(events[4], stream));
      CUDA_VERIFY(cudaEventSynchronize(events[4]));
      double host_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      uint32_t decoded_elements;
      uint8_t decoded_ok;
      CUDA_VERIFY(cudaMemcpy(&decoded_elements, sizes + 1, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
      CUDA_VERIFY(cudaMemcpy(&decoded_ok, success, 1, cudaMemcpyDeviceToHost));
      CUDA_VERIFY(cudaMemcpy(observed.data(), output, bytes, cudaMemcpyDeviceToHost));
      UCCL_CHECK(result.error == dietgpu::FloatDecompressError::None);
      UCCL_CHECK(decoded_ok && decoded_elements == elements);
      UCCL_CHECK(std::memcmp(input.data(), observed.data(), bytes) == 0);
      check_guard(archive_storage, capacity);
      check_guard(output_storage, bytes);
      UCCL_CHECK(scratch.getMaxMemoryUsage() <= scratch.getSizeTotal());
      if (repeats && sample >= 0) {
        std::array<float, 4> ms;
        for (size_t i = 0; i < ms.size(); ++i)
          CUDA_VERIFY(cudaEventElapsedTime(&ms[i], events[i], events[i + 1]));
        std::cout << "{\"test\":\"split_benchmark_gpu\",\"dtype\":\"bf16\","
                  << "\"distribution\":\"normal\",\"sample\":" << sample
                  << ",\"bytes\":" << bytes << ",\"compressed_bytes\":" << encoded_bytes
                  << ",\"split_gpu_ms\":" << ms[0] << ",\"encode_gpu_ms\":" << ms[1]
                  << ",\"size_readback_interval_ms\":" << ms[2]
                  << ",\"decode_gpu_ms\":" << ms[3]
                  << ",\"codec_host_ms\":" << host_ms
                  << ",\"scratch_peak_bytes\":" << scratch.getMaxMemoryUsage()
                  << ",\"exact_bytes\":true,\"guards\":true}\n";
      }
    }
  }

  CUDA_VERIFY(cudaFree(success));
  CUDA_VERIFY(cudaFree(sizes));
  CUDA_VERIFY(cudaFree(output_storage));
  CUDA_VERIFY(cudaFree(archive_storage));
  CUDA_VERIFY(cudaFree(source));
  for (auto event : events) CUDA_VERIFY(cudaEventDestroy(event));
  CUDA_VERIFY(cudaStreamDestroy(stream));
  std::cout << "{\"test\":\"split_roundtrip_gpu\",\"type\":"
            << static_cast<uint32_t>(format.type) << ",\"bytes\":" << bytes
            << ",\"symbols\":" << symbols << ",\"exact_bytes\":true,\"guards\":true}\n";
}

uint32_t parse_positive(char const* text) {
  uint32_t value = 0;
  auto end = text + std::strlen(text);
  auto result = std::from_chars(text, end, value);
  UCCL_CHECK(result.ec == std::errc{} && result.ptr == end && value > 0);
  return value;
}

int main(int argc, char** argv) {
  check_geometry();
  if (argc == 2 && std::strcmp(argv[1], "--geometry") == 0) return 0;
  bool benchmark = argc == 4 && std::strcmp(argv[1], "--benchmark") == 0;
  bool smoke = argc == 2 && std::strcmp(argv[1], "--smoke") == 0;
  UCCL_CHECK(argc == 1 || benchmark || smoke)
      << "usage: --geometry | --smoke | --benchmark BYTES REPEATS";
  uint32_t bytes = benchmark ? parse_positive(argv[2]) : 0;
  uint32_t repeats = benchmark ? parse_positive(argv[3]) : 0;
  UCCL_CHECK(bytes <= (256u << 20) && bytes % 2 == 0 && repeats <= 10000);
  int devices;
  CUDA_VERIFY(cudaGetDeviceCount(&devices));
  UCCL_CHECK_EQ(devices, 1);
  CUDA_VERIFY(cudaSetDevice(0));
  {
    CodecContextScope context(0);
    cudaStream_t stream;
    CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check_scratch_usage(stream);
    CUDA_VERIFY(cudaStreamDestroy(stream));
  }
  if (benchmark) {
    check_roundtrip(formats[1], bytes, static_cast<int>(repeats));
    return 0;
  }
  for (auto const& format : formats) {
    for (uint32_t symbols : {4095u, 4096u, 4097u})
      check_roundtrip(format, symbols * format.symbol_bytes);
    if (smoke) continue;
    for (uint32_t size : {16u << 20, (16u << 20) + 20, 64u << 20, 256u << 20})
      check_roundtrip(format, size);
  }
}
