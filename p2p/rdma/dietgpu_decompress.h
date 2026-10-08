#pragma once
#include "decompress_request.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include <memory>
#include <mutex>

#ifdef USE_DIETGPU
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || \
    defined(__HIPCC__)
#include "dietgpu/float/GpuFloatCodec_hip.h"
#include "dietgpu/utils/DeviceUtils_hip.h"
#include "dietgpu/utils/GreenContext_hip.h"
#else
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/GreenContext.h"
#endif

class CodecContextScope {
 public:
  explicit CodecContextScope(int device)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || \
    defined(__HIPCC__)
      : device_scope_(device) {
#else
  {
    UCCL_CHECK(cuCtxGetCurrent(&previous_) == CUDA_SUCCESS);
    GPU_RT_CHECK(gpuSetDevice(device));
#endif
    dietgpu::initGreenContextIfNeeded(device);
  }

  ~CodecContextScope() {
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__) && \
    !defined(__HIPCC__)
    UCCL_CHECK(cuCtxSetCurrent(previous_) == CUDA_SUCCESS);
#endif
  }

 private:
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || \
    defined(__HIPCC__)
  dietgpu::DeviceScope device_scope_;
#else
  CUcontext previous_ = nullptr;
#endif
};

class DecompressPool;

std::shared_ptr<DecompressPool> make_decompress_pool(int device,
                                                     uint32_t slots);

std::unique_ptr<DecompressRequest> make_decompress_request(
    int device, dietgpu::StackDeviceMemory& scratch, gpuStream_t stream,
    std::mutex& codec_mutex, std::shared_ptr<DecompressPool> const& pool,
    void const* input, void* output, dietgpu::FloatType float_type,
    uint32_t elements);
#endif
