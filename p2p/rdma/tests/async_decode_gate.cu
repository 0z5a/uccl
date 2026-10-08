#include "async_decode_gate.h"

namespace {
__global__ void wait_for_release(DecodeGate* gate, uint64_t timeout_cycles) {
  atomicExch_system(&gate->entered, 1u);
  auto begin = clock64();
  for (;;) {
    uint32_t release;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(release)
                 : "l"(&gate->release)
                 : "memory");
    if (release) return;
    if (clock64() - begin > timeout_cycles) {
      atomicExch_system(&gate->expired, 1u);
      return;
    }
    __nanosleep(256);
  }
}
}  // namespace

void launch_decode_gate(cudaStream_t stream, DecodeGate* gate,
                        uint64_t timeout_cycles) {
  wait_for_release<<<1, 1, 0, stream>>>(gate, timeout_cycles);
}
