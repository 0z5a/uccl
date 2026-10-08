#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct DecodeGate {
  uint32_t entered;
  uint32_t release;
  uint32_t expired;
};

void launch_decode_gate(cudaStream_t stream, DecodeGate* gate,
                        uint64_t timeout_cycles);
