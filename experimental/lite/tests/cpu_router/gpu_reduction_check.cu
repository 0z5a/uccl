// Independent host-double oracle for the opt-in GPU reduction arithmetic.
#include "cpu_router.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>

using namespace uccl_lite::cpu_router;
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int ranks = 4;
    bool interleaveScatter = false;
    size_t packetBytes = 512UL << 10, customElements = 0;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--gpus" && i + 1 < argc)
        ranks = std::stoi(argv[++i]);
      else if (std::string(argv[i]) == "--interleave-scatter")
        interleaveScatter = true;
      else if (std::string(argv[i]) == "--packet-bytes" && i + 1 < argc)
        packetBytes = std::stoull(argv[++i]);
      else if (std::string(argv[i]) == "--elements" && i + 1 < argc) {
        customElements = std::stoull(argv[++i]);
        if (!customElements || customElements > (1UL << 24))
          throw std::invalid_argument("invalid element count");
      } else
        throw std::invalid_argument("expected --gpus N");
    }
    if (ranks < 1 || ranks > 4)
      throw std::invalid_argument(
          "oracle uses 1..4 GPUs for exactly representable FP32 sums");
    std::vector<size_t> counts{1UL,   3UL,    8UL,     17UL,
                               257UL, 4096UL, 65537UL, 262144UL};
    if (customElements) counts = {customElements};
    size_t maxCount = *std::max_element(counts.begin(), counts.end()) * ranks,
           guards = 8;
    std::vector<uint16_t*> input(ranks), output(ranks);
    std::vector<cudaStream_t> streams(ranks);
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      check(cudaMalloc(&input[r], (maxCount + guards * 2) * 2));
      check(cudaMalloc(&output[r], (maxCount + guards * 2) * 2));
      check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    }
    {
      CpuRouterOptions options;
      options.maxTensorBytes = maxCount * 2;
      options.reduction = ReductionPlacement::Gpu;
      options.interleaveScatter = interleaveScatter;
      options.packetBytes = packetBytes;
      CpuRouter router(options);
      std::mt19937 rng(260940093);
      size_t checked = 0;
      int cases = 0;
      for (size_t count : counts)
        for (bool scatter : {false, true})
          for (bool inPlace : {false, true})
            for (bool masked : {false, true}) {
              if (masked && ranks == 1) continue;
              auto plan = scatter ? PacketProgram::reduceScatter(count, ranks)
                                  : PacketProgram::allReduce(count, ranks);
              int root = ranks / 2;
              if (masked) {
                plan.sources &= ~(1U << root);
                plan.destinations = 1U << root;
              }
              // Alternate aligned vector and deliberately unaligned scalar
              // paths.
              size_t offset = count % 8 ? 3 : guards;
              size_t extent = plan.inputElements + guards * 2;
              std::vector<std::vector<uint16_t>> host(
                  ranks, std::vector<uint16_t>(extent));
              std::vector<PacketPort> ports;
              for (int r = 0; r < ranks; ++r) {
                for (auto& bits : host[r]) {
                  unsigned v = rng();
                  // Bounded finite BF16 values with cancellation and varied
                  // exponents. Sum of <=4 terms is exactly representable in
                  // FP32, allowing bitwise comparison to an independent
                  // host-double oracle.
                  bits =
                      uint16_t(((v & 1) << 15) | ((120 + (v >> 1) % 15) << 7) |
                               ((v >> 16) & 127));
                }
                check(cudaSetDevice(r));
                check(cudaMemcpyAsync(input[r], host[r].data(), extent * 2,
                                      cudaMemcpyHostToDevice, streams[r]));
                check(cudaMemsetAsync(output[r], 0xa5, extent * 2, streams[r]));
                auto* out = (inPlace ? input[r] : output[r]) + offset;
                if (inPlace && scatter) out += r * count;
                ports.push_back(
                    {plan.sources & (1U << r) ? input[r] + offset : nullptr,
                     plan.destinations & (1U << r) ? out : nullptr, streams[r],
                     r});
              }
              router.enqueue(plan, ports);
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                check(cudaStreamSynchronize(streams[r]));
                std::vector<uint16_t> actual(extent),
                    expected = inPlace ? host[r]
                                       : std::vector<uint16_t>(extent, 0xa5a5);
                if (plan.destinations & (1U << r)) {
                  for (size_t i = 0; i < count; ++i) {
                    size_t index = offset + (scatter ? r * count : 0) + i;
                    double sum = 0;
                    for (int peer = 0; peer < ranks; ++peer)
                      if (plan.sources & (1U << peer))
                        sum += double(__bfloat162float(
                            __ushort_as_bfloat16(host[peer][index])));
                    size_t target =
                        offset + (inPlace && scatter ? r * count : 0) + i;
                    expected[target] =
                        __bfloat16_as_ushort(__float2bfloat16_rn(float(sum)));
                  }
                }
                check(cudaMemcpy(actual.data(), inPlace ? input[r] : output[r],
                                 extent * 2, cudaMemcpyDeviceToHost));
                if (actual != expected)
                  throw std::runtime_error(
                      "GPU reduction differs from host-double oracle or "
                      "overwrote a guard");
                checked += extent;
              }
              ++cases;
            }
      std::printf(
          "PASS: %d randomized GPU reduction cases, %zu scalar/guard checks; "
          "ranks=%d\n",
          cases, checked, ranks);
    }
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamDestroy(streams[r]));
      check(cudaFree(input[r]));
      check(cudaFree(output[r]));
    }
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
