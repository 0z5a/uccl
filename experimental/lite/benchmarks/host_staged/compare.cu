// Matched eager diagnostic: independent paper reimplementation, UCCL Lite,
// NCCL.
#include "../cpu_router/nccl_identity.hpp"
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include "uccl_lite/host_staged/ep_transport.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <nccl.h>

namespace hs = uccl_lite::host_staged;
namespace cr = uccl_lite::cpu_router;
extern char** environ;
namespace {
using E = __nv_bfloat16;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void nc(ncclResult_t s) {
  if (s != ncclSuccess) throw std::runtime_error(ncclGetErrorString(s));
}
__global__ void fill(E* p, size_t count, int r) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    p[i] = __float2bfloat16_rn(float((r + 1) * (1 + i % 7)) / 64.f);
}
__global__ void verify(E const* p, size_t count, size_t shard, int r, int ranks,
                       bool gather, int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    float expected =
        gather
            ? float((i / shard + 1) * (1 + (i % shard) % 7)) / 64.f
            : float(ranks * (ranks + 1) / 2 * (1 + (r * shard + i) % 7)) / 64.f;
    if (__bfloat162float(p[i]) != expected) atomicAdd(errors, 1);
  }
}
struct Rank {
  E *input = nullptr, *output = nullptr;
  int* errors = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
};
}  // namespace
int main(int argc, char** argv) {
  try {
    int ranks = 4, repeats = 5, iterations = 20, warmup = 5;
    size_t begin = 1024, end = 1UL << 30, chunkBytes = 8UL << 20;
    std::string samples;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (i + 1 >= argc) throw std::invalid_argument("missing option value");
      std::string v = argv[++i];
      if (arg == "--gpus")
        ranks = std::stoi(v);
      else if (arg == "--begin")
        begin = std::stoull(v);
      else if (arg == "--end")
        end = std::stoull(v);
      else if (arg == "--chunk-bytes")
        chunkBytes = std::stoull(v);
      else if (arg == "--repeats")
        repeats = std::stoi(v);
      else if (arg == "--iterations")
        iterations = std::stoi(v);
      else if (arg == "--warmup")
        warmup = std::stoi(v);
      else if (arg == "--samples")
        samples = v;
      else
        throw std::invalid_argument("unknown option: " + arg);
    }
    if (ranks < 2 || ranks > 8 || begin < 1024 || end < begin ||
        end > (1UL << 30) || begin % (ranks * 2) || end % (ranks * 2) ||
        chunkBytes < 2 || chunkBytes % 2 || chunkBytes > (32UL << 20) ||
        iterations < 1 || repeats < 1 || warmup < 1 || samples.empty() ||
        std::filesystem::exists(samples))
      throw std::invalid_argument(
          "invalid settings or sample file already exists");
    std::ofstream out(samples);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.precision(12);
    out << "collective,bytes,in_place,backend,repeat,iteration,slowest_rank_"
           "us\n";
    int version;
    nc(ncclGetVersion(&version));
    auto identity = uccl_lite::benchmark::ncclFallbackIdentity(
        version, uccl_lite::benchmark::ncclConfiguration(environ));
    std::printf("# NCCL=%d identity=%s ranks=%d BF16 eager out-of-place\n",
                version, identity.c_str(), ranks);
    std::printf(
        "# independent ThunderEP communication reimplementation, NOT author "
        "code or paper hardware\n");
    std::printf(
        "# fixed reference chunk=%zu bytes; UCCL Lite gpu-progress/striped "
        "packet=8388608 bytes\n",
        chunkBytes);
    std::vector<int> devices(ranks);
    std::iota(devices.begin(), devices.end(), 0);
    std::vector<Rank> rank(ranks);
    std::vector<ncclComm_t> comm(ranks);
    nc(ncclCommInitAll(comm.data(), ranks, devices.data()));
    for (int r = 0; r < ranks; ++r) {
      auto& p = rank[r];
      check(cudaSetDevice(r));
      check(cudaMalloc(&p.input, end));
      check(cudaMalloc(&p.output, end));
      check(cudaMalloc(&p.errors, sizeof(int)));
      check(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
      check(cudaEventCreate(&p.start));
      check(cudaEventCreate(&p.stop));
    }
    {
      hs::Options opt;
      opt.maxShardElements = end / 2 / ranks;
      opt.chunkElements = std::min(opt.maxShardElements, chunkBytes / 2);
      hs::EpTransport dma(devices, opt);
      opt.transfer = hs::Transfer::SmDecode;
      hs::EpTransport sm(devices, opt);
      cr::PacketPipelineOptions ro;
      ro.maxTensorBytes = end;
      ro.packetBytes = 8UL << 20;
      ro.reduction = cr::ReductionPlacement::Gpu;
      ro.gpuProgress = true;
      ro.interleaveScatter = true;
      cr::PacketPipeline router(ro);
      std::array<char const*, 4> names{"nccl", "thunderep-reimpl-dma",
                                       "thunderep-reimpl-sm",
                                       "uccl-lite-gpu-progress"};
      for (bool gather : {true, false})
        for (size_t bytes = begin; bytes <= end; bytes *= 2) {
          size_t shard = bytes / 2 / ranks;
          size_t inCount = shard * (gather ? 1 : ranks),
                 outCount = shard * (gather ? ranks : 1);
          std::vector<hs::Binding> bindings;
          std::vector<cr::PacketPort> ports;
          for (int r = 0; r < ranks; ++r) {
            auto const& p = rank[r];
            bindings.push_back({p.input, p.output, p.stream});
            ports.push_back({p.input, p.output, p.stream, r});
          }
          router.prepare(ports);
          auto plan = gather ? cr::PacketProgram::allGather(shard, ranks)
                             : cr::PacketProgram::reduceScatter(shard, ranks);
          auto op = gather ? hs::Collective::AllGather
                           : hs::Collective::ReduceScatter;
          auto run = [&](int backend, int repeat, int count) {
            for (int it = 0; it < count; ++it) {
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                auto& p = rank[r];
                fill<<<32, 256, 0, p.stream>>>(p.input, inCount, r);
                check(cudaMemsetAsync(p.errors, 0, sizeof(int), p.stream));
                check(cudaEventRecord(p.start, p.stream));
              }
              if (backend == 0) {
                nc(ncclGroupStart());
                for (int r = 0; r < ranks; ++r) {
                  check(cudaSetDevice(r));
                  auto& p = rank[r];
                  if (gather)
                    nc(ncclAllGather(p.input, p.output, shard, ncclBfloat16,
                                     comm[r], p.stream));
                  else
                    nc(ncclReduceScatter(p.input, p.output, shard, ncclBfloat16,
                                         ncclSum, comm[r], p.stream));
                }
                nc(ncclGroupEnd());
              } else if (backend == 1)
                dma.enqueue(op, shard, bindings);
              else if (backend == 2)
                sm.enqueue(op, shard, bindings);
              else
                router.enqueue(plan, ports);
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                auto& p = rank[r];
                check(cudaEventRecord(p.stop, p.stream));
                verify<<<32, 256, 0, p.stream>>>(p.output, outCount, shard, r,
                                                 ranks, gather, p.errors);
                check(cudaGetLastError());
              }
              double worst = 0;
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                auto& p = rank[r];
                check(cudaStreamSynchronize(p.stream));
                int errors;
                float ms;
                check(cudaMemcpy(&errors, p.errors, sizeof(int),
                                 cudaMemcpyDeviceToHost));
                if (errors)
                  throw std::runtime_error(std::string(names[backend]) +
                                           " correctness failure");
                check(cudaEventElapsedTime(&ms, p.start, p.stop));
                worst = std::max(worst, double(ms) * 1000);
              }
              if (repeat >= 0)
                out << (gather ? "all_gather" : "reduce_scatter") << ','
                    << bytes << ",0," << names[backend] << ',' << repeat << ','
                    << it << ',' << worst << '\n';
            }
          };
          for (int b = 0; b < 4; ++b) run(b, -1, warmup);
          for (int repeat = 0; repeat < repeats; ++repeat)
            for (int position = 0; position < 4; ++position)
              run((position + repeat) % 4, repeat, iterations);
          out.flush();
          std::printf("# checked %s bytes=%zu all four backends\n",
                      gather ? "allgather" : "reducescatter", bytes);
          std::fflush(stdout);
        }
    }
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      auto& p = rank[r];
      nc(ncclCommDestroy(comm[r]));
      check(cudaFree(p.input));
      check(cudaFree(p.output));
      check(cudaFree(p.errors));
      check(cudaEventDestroy(p.start));
      check(cudaEventDestroy(p.stop));
      check(cudaStreamDestroy(p.stream));
    }
    std::puts("# correctness: PASS (every invocation, all ranks)");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
