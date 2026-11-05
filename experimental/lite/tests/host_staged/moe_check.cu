// Framework-free routed MoE correctness test. Intentionally tiny scalar GEMMs:
// validates transport/dependency semantics, not model throughput or kernel
// speed.
#include "uccl_lite/host_staged/ep_transport.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>

using namespace uccl_lite::host_staged;
namespace {
using E = __nv_bfloat16;
constexpr int T = 17, H = 8, I = 16, Local = 2, K = 2, Row = H + 2 * K;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
__host__ __device__ float weight(int expert, int i, int j, int kind) {
  return float(((expert + 1) * 7 + i * 3 + j * 5 + kind * 11) % 13 - 6) / 32.f;
}
__host__ __device__ float activation(int source, int token, int h,
                                     int iteration) {
  return float((source * 7 + token * 3 + h + iteration * 5) % 17 - 8) / 16.f;
}
__host__ __device__ int route(int source, int token, int iteration, int experts,
                              int k) {
  // Alternate balanced and heavily skewed routing, with top-2 weights fixed.
  int first = iteration % 3 == 0 ? 0 : (source + token + iteration) % experts;
  return (first + k) % experts;
}
__global__ void makeTokens(E* x, int source, int iteration, int ranks) {
  int t = threadIdx.x;
  if (t >= T) return;
  for (int h = 0; h < H; ++h)
    x[t * Row + h] = __float2bfloat16_rn(activation(source, t, h, iteration));
  for (int k = 0; k < K; ++k) {
    x[t * Row + H + k] = __float2bfloat16_rn(
        float(route(source, t, iteration, Local * ranks, k)));
    x[t * Row + H + K + k] = __float2bfloat16_rn(k == 0 ? .75f : .25f);
  }
}
struct Work {
  int count[Local], token[Local * T];
  float gateWeight[Local * T], packed[Local * T * H];
  float up[Local * T * I], gate[Local * T * I], hidden[Local * T * I];
  float output[T * H];
};
__global__ void permute(E const* tokens, Work* w, int rank) {
  int t = threadIdx.x;
  if (t >= T) return;
  for (int k = 0; k < K; ++k) {
    int expert = int(__bfloat162float(tokens[t * Row + H + k]));
    if (expert / Local != rank) continue;
    int local = expert % Local, slot = atomicAdd(&w->count[local], 1);
    int row = local * T + slot;
    w->token[row] = t;
    w->gateWeight[row] = __bfloat162float(tokens[t * Row + H + K + k]);
    for (int h = 0; h < H; ++h)
      w->packed[row * H + h] = __bfloat162float(tokens[t * Row + h]);
  }
}
__global__ void upGemm(Work* w, int rank) {
  int local = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
  if (row >= w->count[local] || i >= I) return;
  int index = local * T + row, expert = rank * Local + local;
  float up = 0, gate = 0;
  for (int h = 0; h < H; ++h) {
    up += w->packed[index * H + h] * weight(expert, h, i, 0);
    gate += w->packed[index * H + h] * weight(expert, h, i, 1);
  }
  w->up[index * I + i] = up;
  w->gate[index * I + i] = gate;
}
__global__ void swiglu(Work* w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Local * T * I) {
    float gate = w->gate[i];
    w->hidden[i] = w->up[i] * gate / (1.f + expf(-gate));
  }
}
__global__ void downUnpermute(Work* w, int rank) {
  int local = blockIdx.x, row = blockIdx.y, h = threadIdx.x;
  if (row >= w->count[local] || h >= H) return;
  int index = local * T + row, expert = rank * Local + local;
  float sum = 0;
  for (int i = 0; i < I; ++i)
    sum += w->hidden[index * I + i] * weight(expert, i, h, 2);
  atomicAdd(&w->output[w->token[index] * H + h], sum * w->gateWeight[index]);
}
__global__ void finish(Work const* w, E* output) {
  int i = threadIdx.x;
  if (i < T * H) output[i] = __float2bfloat16_rn(w->output[i]);
}
void compute(E const* tokens, E* output, Work* work, int rank,
             cudaStream_t stream) {
  check(cudaMemsetAsync(work, 0, sizeof(Work), stream));
  permute<<<1, 32, 0, stream>>>(tokens, work, rank);
  upGemm<<<dim3(Local, T), 32, 0, stream>>>(work, rank);
  swiglu<<<(Local * T * I + 127) / 128, 128, 0, stream>>>(work);
  downUnpermute<<<dim3(Local, T), 32, 0, stream>>>(work, rank);
  finish<<<1, 256, 0, stream>>>(work, output);
  check(cudaGetLastError());
}
float rounded(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
std::vector<float> oracle(int source, int iteration, int ranks) {
  std::vector<float> expected(T * H, 0);
  for (int r = 0; r < ranks; ++r) {
    std::vector<float> partial(T * H, 0);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < K; ++k) {
        int expert = route(source, t, iteration, Local * ranks, k);
        if (expert / Local != r) continue;
        double hidden[I];
        for (int i = 0; i < I; ++i) {
          double up = 0, gate = 0;
          for (int h = 0; h < H; ++h) {
            double x = activation(source, t, h, iteration);
            up += x * weight(expert, h, i, 0);
            gate += x * weight(expert, h, i, 1);
          }
          hidden[i] = up * gate / (1. + std::exp(-gate));
        }
        for (int h = 0; h < H; ++h) {
          double sum = 0;
          for (int i = 0; i < I; ++i)
            sum += hidden[i] * weight(expert, i, h, 2);
          partial[t * H + h] += float(sum * (k == 0 ? .75 : .25));
        }
      }
    for (int i = 0; i < T * H; ++i) expected[i] += rounded(partial[i]);
  }
  for (auto& x : expected) x = rounded(x);
  return expected;
}
void run(std::vector<int> const& devices, Transfer transfer) {
  int ranks = devices.size();
  std::vector<E*> local(ranks), gathered(ranks), partial(ranks), output(ranks);
  std::vector<Work*> workspace(ranks);
  std::vector<cudaStream_t> stream(ranks);
  std::vector<cudaEvent_t> produced(ranks * ranks);
  std::vector<cudaGraph_t> graph(ranks);
  std::vector<cudaGraphExec_t> exec(ranks);
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    check(cudaStreamCreateWithFlags(&stream[r], cudaStreamNonBlocking));
    check(cudaMalloc(&local[r], T * Row * sizeof(E)));
    check(cudaMalloc(&gathered[r], ranks * T * Row * sizeof(E)));
    check(cudaMalloc(&partial[r], ranks * T * H * sizeof(E)));
    check(cudaMalloc(&output[r], T * H * sizeof(E)));
    check(cudaMalloc(&workspace[r], ranks * sizeof(Work)));
    for (int s = 0; s < ranks; ++s) {
      check(cudaEventCreateWithFlags(&produced[r * ranks + s],
                                     cudaEventDisableTiming));
      // Callback kernels must be loaded on every GPU before inter-rank waits.
      makeTokens<<<1, 32, 0, stream[r]>>>(local[r], r, 0, ranks);
      compute(local[r], partial[r] + s * T * H, workspace[r] + s, r, stream[r]);
    }
    check(cudaStreamSynchronize(stream[r]));
  }
  {
    EpTransport ep(devices, {transfer, T * Row, 73});
    std::vector<Binding> ag(ranks), rs(ranks);
    for (int r = 0; r < ranks; ++r) {
      ag[r] = {local[r], gathered[r], stream[r]};
      rs[r] = {partial[r], output[r], stream[r]};
      if (transfer == Transfer::SmDecode) {
        check(cudaSetDevice(devices[r]));
        check(cudaStreamBeginCapture(stream[r],
                                     cudaStreamCaptureModeThreadLocal));
        ep.enqueueRank(r, Collective::AllGather, T * Row, ag[r]);
        for (int s = 0; s < ranks; ++s)
          compute(gathered[r] + s * T * Row, partial[r] + s * T * H,
                  workspace[r] + s, r, stream[r]);
        ep.enqueueRank(r, Collective::ReduceScatter, T * H, rs[r]);
        check(cudaStreamEndCapture(stream[r], &graph[r]));
        check(cudaGraphInstantiate(&exec[r], graph[r], nullptr, nullptr, 0));
      }
    }
    for (int iteration = 0; iteration < 24; ++iteration) {
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        makeTokens<<<1, 32, 0, stream[r]>>>(local[r], r, iteration, ranks);
        if (transfer == Transfer::SmDecode)
          check(cudaGraphLaunch(exec[r], stream[r]));
      }
      if (transfer == Transfer::DmaPrefill) {
        ep.enqueue(
            Collective::AllGather, T * Row, ag,
            [&](int r, int s, void const* shard, size_t,
                cudaStream_t computeStream) {
              compute(static_cast<E const*>(shard), partial[r] + s * T * H,
                      workspace[r] + s, r, computeStream);
              check(cudaEventRecord(produced[r * ranks + s], computeStream));
            });
        for (int r = 0; r < ranks; ++r)
          rs[r].inputReady.assign(produced.begin() + r * ranks,
                                  produced.begin() + (r + 1) * ranks);
        ep.enqueue(Collective::ReduceScatter, T * H, rs);
      }
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        check(cudaStreamSynchronize(stream[r]));
        auto expected = oracle(r, iteration, ranks);
        std::vector<E> actual(T * H);
        check(cudaMemcpy(actual.data(), output[r], actual.size() * sizeof(E),
                         cudaMemcpyDeviceToHost));
        for (int i = 0; i < T * H; ++i) {
          float got = __bfloat162float(actual[i]);
          if (!std::isfinite(got) || std::abs(got - expected[i]) >
                                         2e-6f + .01f * std::abs(expected[i]))
            throw std::runtime_error(
                "routed MoE differs from independent CPU oracle");
        }
      }
    }
    if (transfer == Transfer::SmDecode)
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        check(cudaGraphExecDestroy(exec[r]));
        check(cudaGraphDestroy(graph[r]));
      }
  }
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    check(cudaFree(local[r]));
    check(cudaFree(gathered[r]));
    check(cudaFree(partial[r]));
    check(cudaFree(output[r]));
    check(cudaFree(workspace[r]));
    for (int s = 0; s < ranks; ++s)
      check(cudaEventDestroy(produced[r * ranks + s]));
    check(cudaStreamDestroy(stream[r]));
  }
  std::printf(
      "PASS: %s routed top-2 MoE, permute/up+gate GEMM/SwiGLU/down/unpermute, "
      "24 changing/skewed routes, ranks=%d\n",
      transfer == Transfer::DmaPrefill ? "DMA overlapped"
                                       : "SM unchanged graph",
      ranks);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    int ranks = argc == 2 ? std::stoi(argv[1]) : 4;
    if (ranks < 1 || ranks > 4)
      throw std::invalid_argument("expected GPU count 1..4");
    std::vector<int> devices(ranks);
    std::iota(devices.begin(), devices.end(), 0);
    run(devices, Transfer::DmaPrefill);
    run(devices, Transfer::SmDecode);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
