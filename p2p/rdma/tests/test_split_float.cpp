#include "dietgpu_decompress.h"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/StaticUtils.h"
#include "util/debug.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
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

void check_roundtrip(Format const& format, uint32_t bytes) {
  CodecContextScope context(0);
  uint32_t elements = bytes / format.word_bytes;
  uint32_t symbols = bytes / format.symbol_bytes;
  auto capacity = dietgpu::getMaxFloatCompressedSize(format.type, elements);
  std::vector<uint8_t> input(bytes), observed(bytes);
  for (uint32_t i = 0; i < bytes; ++i)
    input[i] = ((i * 1103515245u + 12345u) >> 16) & 255;

  cudaStream_t stream;
  CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  void* source;
  void* archive;
  void* output;
  uint32_t* sizes;
  uint8_t* success;
  CUDA_VERIFY(cudaMalloc(&source, bytes));
  CUDA_VERIFY(cudaMalloc(&archive, capacity));
  CUDA_VERIFY(cudaMalloc(&output, bytes));
  CUDA_VERIFY(cudaMalloc(&sizes, 2 * sizeof(uint32_t)));
  CUDA_VERIFY(cudaMalloc(&success, 1));
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
    dietgpu::floatCompressSplitOneBatch(scratch, config, stream, split);
    UCCL_CHECK_EQ(split.compRowStride, dietgpu::roundUp(symbols, 16u));
    UCCL_CHECK_EQ(split.toComp_dev.num, split.compRowStride);
    dietgpu::floatCompressEncodeOneBatch(scratch, config, split, sizes, stream);
    split.histogram_dev.release();
    split.toComp_dev.release();

    uint32_t encoded_bytes;
    CUDA_VERIFY(cudaMemcpy(&encoded_bytes, sizes, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));
    UCCL_CHECK(encoded_bytes > 0 && encoded_bytes <= capacity);
    void const* inputs[] = {archive};
    void* outputs[] = {output};
    auto result = dietgpu::floatDecompress(scratch, config, 1, inputs, outputs,
                                           &elements, success, sizes + 1, stream);
    CUDA_VERIFY(cudaStreamSynchronize(stream));
    uint32_t decoded_elements;
    uint8_t decoded_ok;
    CUDA_VERIFY(cudaMemcpy(&decoded_elements, sizes + 1, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));
    CUDA_VERIFY(cudaMemcpy(&decoded_ok, success, 1, cudaMemcpyDeviceToHost));
    CUDA_VERIFY(cudaMemcpy(observed.data(), output, bytes, cudaMemcpyDeviceToHost));
    UCCL_CHECK(result.error == dietgpu::FloatDecompressError::None);
    UCCL_CHECK(decoded_ok && decoded_elements == elements);
    UCCL_CHECK(std::memcmp(input.data(), observed.data(), bytes) == 0);
  }

  CUDA_VERIFY(cudaFree(success));
  CUDA_VERIFY(cudaFree(sizes));
  CUDA_VERIFY(cudaFree(output));
  CUDA_VERIFY(cudaFree(archive));
  CUDA_VERIFY(cudaFree(source));
  CUDA_VERIFY(cudaStreamDestroy(stream));
  std::cout << "{\"test\":\"split_roundtrip_gpu\",\"type\":"
            << static_cast<uint32_t>(format.type) << ",\"bytes\":" << bytes
            << ",\"symbols\":" << symbols << ",\"exact_bytes\":true}\n";
}

int main(int argc, char** argv) {
  check_geometry();
  if (argc == 2 && std::strcmp(argv[1], "--geometry") == 0) return 0;
  int devices;
  CUDA_VERIFY(cudaGetDeviceCount(&devices));
  UCCL_CHECK_EQ(devices, 1);
  CUDA_VERIFY(cudaSetDevice(0));
  for (auto const& format : formats)
    for (uint32_t bytes : {16u << 20, (16u << 20) + 20, 64u << 20, 256u << 20})
      check_roundtrip(format, bytes);
}
