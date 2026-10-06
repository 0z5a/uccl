#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace uccl::local {

inline void check(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA line %d %s: %s (%d)\n", line, expression,
                 cudaGetErrorString(status), static_cast<int>(status));
    std::exit(1);
  }
}

#define UCCL_LOCAL_CUDA(expression) \
  ::uccl::local::check((expression), #expression, __LINE__)

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    UCCL_LOCAL_CUDA(cudaGetDevice(&previous_));
    UCCL_LOCAL_CUDA(cudaSetDevice(device));
  }
  ~DeviceScope() { UCCL_LOCAL_CUDA(cudaSetDevice(previous_)); }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int previous_;
};

}  // namespace uccl::local
