#include "../../local/cuda_utils.h"
#include "dietgpu/float/GpuFloatCodec.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

struct Metadata {
  uint32_t bytes;
  uint32_t words;
  uint8_t success;
};

int main() {
  int devices = 0;
  UCCL_LOCAL_CUDA(cudaGetDeviceCount(&devices));
  if (!devices) return 2;
  constexpr uint32_t maximum = 65536;
  constexpr size_t workspace = 96 << 20;
  const std::array<uint32_t, 17> sizes{1, 7, 8, 15, 16, 31, 32, 63, 64,
                                     511, 512, 513, 4095, 4096, 4097, 65535, maximum};
  const std::array<dietgpu::FloatType, 3> types{
      dietgpu::FloatType::kFloat16, dietgpu::FloatType::kBFloat16,
      dietgpu::FloatType::kFloat32};
  const std::array<uint32_t, 12> fp32_specials{
      0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001,
      0x7f800001, 0xffc0abcd, 0x00000001, 0x007fffff, 0x00800000,
      0x7f7fffff, 0xff7fffff};
  std::cout << "{\"cases\":[";
  unsigned cases = 0;
  for (int device = 0; device < devices; ++device) {
    uccl::local::DeviceScope current(device);
    cudaStream_t stream{};
    UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    dietgpu::StackDeviceMemory scratch(device, workspace);
    Metadata *host = nullptr, *gpu = nullptr;
    UCCL_LOCAL_CUDA(cudaHostAlloc(&host, sizeof(Metadata), cudaHostAllocDefault));
    UCCL_LOCAL_CUDA(cudaMalloc(&gpu, sizeof(Metadata)));
    for (auto type : types) {
      const unsigned width = type == dietgpu::FloatType::kFloat32 ? 4 : 2;
      std::vector<uint8_t> input(maximum * width), output(input.size());
      for (uint32_t i = 0; i < maximum; ++i) {
        if (width == 2) {
          uint16_t bits = i;
          std::memcpy(input.data() + i * width, &bits, width);
        } else {
          uint32_t bits = i < fp32_specials.size() ? fp32_specials[i]
                                                  : i * 0x9e3779b9u + 0x7f4a7c15u;
          std::memcpy(input.data() + i * width, &bits, width);
        }
      }
      void *source = nullptr, *encoded = nullptr, *destination = nullptr;
      const uint32_t capacity = dietgpu::getMaxFloatCompressedSize(type, maximum);
      UCCL_LOCAL_CUDA(cudaMalloc(&source, input.size()));
      UCCL_LOCAL_CUDA(cudaMalloc(&destination, input.size()));
      UCCL_LOCAL_CUDA(cudaMalloc(&encoded, capacity));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(source, input.data(), input.size(), cudaMemcpyHostToDevice, stream));
      dietgpu::FloatCompressConfig config;
      config.floatType = type;
      config.is16ByteAligned = true;
      config.useChecksum = true;
      for (uint32_t words : sizes) {
        const void* sources[] = {source};
        void* compressed[] = {encoded};
        dietgpu::floatCompress(scratch, config, 1, sources, &words, compressed, &gpu->bytes, stream);
        UCCL_LOCAL_CUDA(cudaMemcpyAsync(&host->bytes, &gpu->bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        UCCL_LOCAL_CUDA(cudaStreamSynchronize(stream));
        if (host->bytes > capacity || host->bytes == 0) return 2;
        const void* archives[] = {encoded};
        void* destinations[] = {destination};
        auto status = dietgpu::floatDecompress(scratch, config, 1, archives, destinations,
                                               &words, &gpu->success, &gpu->words, stream);
        UCCL_LOCAL_CUDA(cudaMemcpyAsync(&host->words, &gpu->words, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        UCCL_LOCAL_CUDA(cudaMemcpyAsync(&host->success, &gpu->success, sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        UCCL_LOCAL_CUDA(cudaMemcpyAsync(output.data(), destination, words * width, cudaMemcpyDeviceToHost, stream));
        UCCL_LOCAL_CUDA(cudaStreamSynchronize(stream));
        if (status.error != dietgpu::FloatDecompressError::None || host->success != 1 ||
            host->words != words || std::memcmp(input.data(), output.data(), words * width)) return 3;
        if (cases++) std::cout << ',';
        std::cout << "{\"device\":" << device << ",\"dtype\":" << static_cast<int>(type)
                  << ",\"words\":" << words << ",\"raw_bytes\":" << words * width
                  << ",\"compressed_bytes\":" << host->bytes << ",\"bitwise\":\"PASS\"}";
      }
      UCCL_LOCAL_CUDA(cudaFree(source));
      UCCL_LOCAL_CUDA(cudaFree(encoded));
      UCCL_LOCAL_CUDA(cudaFree(destination));
    }
    UCCL_LOCAL_CUDA(cudaFree(gpu));
    UCCL_LOCAL_CUDA(cudaFreeHost(host));
    UCCL_LOCAL_CUDA(cudaStreamDestroy(stream));
  }
  std::cout << "],\"total_cases\":" << cases
            << ",\"fp16_bf16_all_bit_patterns\":true,\"empty_policy\":\"RAW_BYPASS\",\"status\":\"PASS\"}\n";
}
