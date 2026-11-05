#pragma once
#include "signal.hpp"
#include <cuda_runtime.h>
namespace uccl_lite::cpu_router {
struct DeviceSignalRef {
  GenerationSignal* word;
  __device__ Generation acquire() const {
    Generation g;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(g)
                 : "l"(&word->value)
                 : "memory");
    return g;
  }
  __device__ void publish(Generation g) const {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(&word->value),
                 "r"(g)
                 : "memory");
  }
  __device__ void wait(Generation g) const {
    while (acquire() < g) __nanosleep(64);
  }
};
struct DevicePacketOwnership {
  DeviceSignalRef ready, released;
  GenerationSignal* readers;
  __device__ void acquire(Generation previous) const {
    if (previous) released.wait(previous);
  }
  __device__ void publish(Generation g) const { ready.publish(g); }
  __device__ void consume(unsigned reader, Generation g) const {
    DeviceSignalRef{readers + reader}.publish(g);
  }
};
// Device-side channel operations compose with signaling and caller CTA
// barriers.
struct DeviceChannel {
  __device__ static uint4 get16(void const* p) {
    return *static_cast<uint4 const*>(p);
  }
  __device__ static uint16_t get2(void const* p) {
    uint16_t v;
    asm volatile("ld.global.u16 %0,[%1];" : "=h"(v) : "l"(p) : "memory");
    return v;
  }
  __device__ static void put16(void* p, uint4 value) {
    *static_cast<uint4*>(p) = value;
  }
  __device__ static void put2(void* p, uint16_t value) {
    asm volatile("st.global.u16 [%0],%1;" ::"l"(p), "h"(value) : "memory");
  }
  __device__ static void put(void* dst, void const* src, size_t bytes,
                             size_t lane, size_t stride) {
    if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) &
         15) == 0) {
      for (size_t i = lane; i < bytes / 16; i += stride)
        put16(static_cast<char*>(dst) + i * 16,
              get16(static_cast<char const*>(src) + i * 16));
      for (size_t i = bytes / 16 * 16 + lane; i < bytes; i += stride)
        static_cast<char*>(dst)[i] = static_cast<char const*>(src)[i];
    } else
      for (size_t i = lane; i < bytes; i += stride)
        static_cast<char*>(dst)[i] = static_cast<char const*>(src)[i];
  }
  __device__ static void get(void* dst, void const* src, size_t bytes,
                             size_t lane, size_t stride) {
    put(dst, src, bytes, lane, stride);
  }
};
}  // namespace uccl_lite::cpu_router
