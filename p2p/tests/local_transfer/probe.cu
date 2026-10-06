#include "../../local/cuda_utils.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using uccl::local::DeviceScope;

int main() {
  int count = 0;
  UCCL_LOCAL_CUDA(cudaGetDeviceCount(&count));
  if (!count) return 2;
  std::cout << "{\"device_count\":" << count << ",\"devices\":[";
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop{};
    UCCL_LOCAL_CUDA(cudaGetDeviceProperties(&prop, i));
    int registration = 0;
    UCCL_LOCAL_CUDA(cudaDeviceGetAttribute(&registration,
                                         cudaDevAttrHostRegisterSupported, i));
    if (i) std::cout << ',';
    std::cout << "{\"index\":" << i << ",\"name\":\"" << prop.name
              << "\",\"major\":" << prop.major << ",\"minor\":" << prop.minor
              << ",\"async_engine_count\":" << prop.asyncEngineCount
              << ",\"unified_addressing\":" << prop.unifiedAddressing
              << ",\"host_register_supported\":" << registration
              << ",\"memory_bytes\":" << prop.totalGlobalMem << '}';
  }
  std::cout << "],\"peer_access\":[";
  for (int i = 0; i < count; ++i) {
    if (i) std::cout << ',';
    std::cout << '[';
    for (int j = 0; j < count; ++j) {
      if (j) std::cout << ',';
      if (i == j) { std::cout << "null"; continue; }
      int accessible = 0;
      UCCL_LOCAL_CUDA(cudaDeviceCanAccessPeer(&accessible, i, j));
      std::cout << accessible;
    }
    std::cout << ']';
  }
  constexpr size_t bytes = 1 << 20;
  std::vector<uint8_t> input(bytes), output(bytes), poison(bytes, 0xa5);
  for (size_t i = 0; i < bytes; ++i) input[i] = (i * 17 + (i >> 8)) & 255;
  std::cout << "],\"staging\":[";
  for (int source = 0; source < count; ++source) {
    int target = count > 1 ? (source + 1) % count : source;
    void *src = nullptr, *dst = nullptr, *host = nullptr;
    cudaStream_t d2h{}, h2d{};
    cudaEvent_t ready{}, completed{};
    {
      DeviceScope device(source);
      UCCL_LOCAL_CUDA(cudaMalloc(&src, bytes));
      UCCL_LOCAL_CUDA(cudaHostAlloc(&host, bytes, cudaHostAllocPortable));
      UCCL_LOCAL_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));
      UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&d2h, cudaStreamNonBlocking));
      UCCL_LOCAL_CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, d2h));
      UCCL_LOCAL_CUDA(cudaEventRecord(ready, d2h));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(src, poison.data(), bytes, cudaMemcpyHostToDevice, d2h));
    }
    {
      DeviceScope device(target);
      UCCL_LOCAL_CUDA(cudaMalloc(&dst, bytes));
      UCCL_LOCAL_CUDA(cudaStreamCreateWithFlags(&h2d, cudaStreamNonBlocking));
      UCCL_LOCAL_CUDA(cudaEventCreateWithFlags(&completed, cudaEventDisableTiming));
      UCCL_LOCAL_CUDA(cudaStreamWaitEvent(h2d, ready, 0));
      UCCL_LOCAL_CUDA(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, h2d));
      UCCL_LOCAL_CUDA(cudaEventRecord(completed, h2d));
      UCCL_LOCAL_CUDA(cudaEventSynchronize(completed));
      UCCL_LOCAL_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
      if (std::memcmp(input.data(), output.data(), bytes)) return 2;
      UCCL_LOCAL_CUDA(cudaEventDestroy(completed));
      UCCL_LOCAL_CUDA(cudaStreamDestroy(h2d));
      UCCL_LOCAL_CUDA(cudaFree(dst));
    }
    {
      DeviceScope device(source);
      UCCL_LOCAL_CUDA(cudaStreamSynchronize(d2h));
      UCCL_LOCAL_CUDA(cudaEventDestroy(ready));
      UCCL_LOCAL_CUDA(cudaStreamDestroy(d2h));
      UCCL_LOCAL_CUDA(cudaFree(src));
      UCCL_LOCAL_CUDA(cudaFreeHost(host));
    }
    if (source) std::cout << ',';
    std::cout << "{\"source\":" << source << ",\"target\":" << target
              << ",\"bytes\":" << bytes << ",\"pinned_DMA\":\"PASS\""
              << ",\"cross_device_event\":\"PASS\",\"source_reuse\":\"PASS\"}";
  }
  std::cout << "],\"status\":\"PASS\"}\n";
}
