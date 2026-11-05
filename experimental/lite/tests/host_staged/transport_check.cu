#include "uccl_lite/host_staged/ep_transport.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_bf16.h>

using namespace uccl_lite::host_staged;
namespace {
using Elem = __nv_bfloat16;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void require(bool v, char const* message) {
  if (!v) throw std::runtime_error(message);
}
struct Buffers {
  std::vector<int> devices;
  std::vector<Elem*> input, output, partial;
  std::vector<cudaStream_t> streams;
  size_t capacity;
  Buffers(std::vector<int> d, size_t n)
      : devices(d),
        input(d.size()),
        output(d.size()),
        partial(d.size()),
        streams(d.size()),
        capacity(n * d.size() + 16) {
    for (size_t r = 0; r < d.size(); ++r) {
      check(cudaSetDevice(d[r]));
      check(cudaMalloc(&input[r], capacity * 2));
      check(cudaMalloc(&output[r], capacity * 2));
      check(cudaMalloc(&partial[r], capacity * 2));
      check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    }
  }
  ~Buffers() {
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaStreamSynchronize(streams[r]);
      cudaFree(input[r]);
      cudaFree(output[r]);
      cudaFree(partial[r]);
      cudaStreamDestroy(streams[r]);
    }
  }
  void sync() {
    for (size_t r = 0; r < devices.size(); ++r) {
      check(cudaSetDevice(devices[r]));
      check(cudaStreamSynchronize(streams[r]));
    }
  }
};
std::vector<uint16_t> read(Elem* p, size_t n) {
  std::vector<uint16_t> result(n);
  check(cudaMemcpy(result.data(), p, n * 2, cudaMemcpyDeviceToHost));
  return result;
}
uint16_t bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float value(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }

size_t eager(std::vector<int> const& devices, Transfer transfer) {
  size_t const maxShard = 65539, guard = 3;
  Buffers b(devices, maxShard);
  int ranks = devices.size();
  EpTransport ep(devices, {transfer, maxShard, 257});
  std::mt19937 rng(40093);
  size_t cases = 0;
  for (size_t n : {size_t(1), size_t(3), size_t(17), size_t(256), size_t(257),
                   size_t(258), size_t(777), size_t(4099), maxShard}) {
    for (auto op : {Collective::AllGather, Collective::ReduceScatter})
      for (bool inplace : {false, true}) {
        bool gather = op == Collective::AllGather;
        size_t inCount = n * (gather ? 1 : ranks);
        size_t outCount = n * (gather ? ranks : 1);
        std::vector<std::vector<uint16_t>> host(
            ranks, std::vector<uint16_t>(b.capacity, 0x5a5a));
        std::vector<Binding> bindings(ranks);
        for (int r = 0; r < ranks; ++r) {
          size_t inputOffset = guard + (inplace && gather ? r * n : 0);
          for (size_t i = 0; i < inCount; ++i) {
            unsigned x = rng();
            host[r][inputOffset + i] =
                uint16_t(((x & 1) << 15) | ((120 + ((x >> 1) % 12)) << 7) |
                         ((x >> 16) & 127));
          }
          check(cudaSetDevice(devices[r]));
          check(cudaMemcpyAsync(b.input[r], host[r].data(), b.capacity * 2,
                                cudaMemcpyHostToDevice, b.streams[r]));
          check(
              cudaMemsetAsync(b.output[r], 0xa5, b.capacity * 2, b.streams[r]));
          bindings[r] = {b.input[r] + inputOffset,
                         (inplace ? b.input[r] : b.output[r]) + guard +
                             (inplace && !gather ? r * n : 0),
                         b.streams[r]};
        }
        ep.enqueue(op, n, bindings);
        b.sync();
        for (int r = 0; r < ranks; ++r) {
          check(cudaSetDevice(devices[r]));
          auto expected =
              inplace ? host[r] : std::vector<uint16_t>(b.capacity, 0xa5a5);
          size_t outputOffset = guard + (inplace && !gather ? r * n : 0);
          for (size_t i = 0; i < outCount; ++i) {
            if (gather) {
              int s = i / n;
              expected[outputOffset + i] =
                  host[s][guard + (inplace ? s * n : 0) + i % n];
            } else {
              double sum = 0;
              for (int s = 0; s < ranks; ++s)
                sum += value(host[s][guard + r * n + i]);
              expected[outputOffset + i] = bits(float(sum));
            }
          }
          require(
              read(inplace ? b.input[r] : b.output[r], b.capacity) == expected,
              "eager result/guard differs from independent host oracle");
          if (!inplace)
            require(read(b.input[r], b.capacity) == host[r], "input modified");
        }
        ++cases;
      }
  }
  std::printf("PASS: %zu eager %s cases, tails, in-place, guards, ranks=%d\n",
              cases, transfer == Transfer::DmaPrefill ? "DMA" : "SM", ranks);
  return cases;
}

__global__ void fill(Elem* p, size_t n, int rank, int iteration) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    p[i] = __float2bfloat16_rn(float((int(i % 11) - 5) + rank + iteration % 7));
}
__global__ void lag() {
  auto start = clock64();
  while (clock64() - start < 100000) __nanosleep(128);
}
void graphs(std::vector<int> const& devices) {
  int ranks = devices.size();
  size_t n = 777;
  Buffers b(devices, n);
  EpTransport ep(devices, {Transfer::SmDecode, n, 127});
  std::vector<cudaGraph_t> graph(ranks);
  std::vector<cudaGraphExec_t> exec(ranks);
  // Mixed AG/RS, alternating banks across operation boundaries. The same graph
  // is replayed with changing contents; no beginReplay or host epoch update.
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    cudaFuncAttributes attributes;
    check(cudaFuncGetAttributes(&attributes, lag));
    check(
        cudaStreamBeginCapture(b.streams[r], cudaStreamCaptureModeThreadLocal));
    for (int pair = 0; pair < 3; ++pair) {
      ep.enqueueRank(r, Collective::AllGather, n,
                     {b.input[r], b.output[r], b.streams[r]});
      ep.enqueueRank(r, Collective::ReduceScatter, n,
                     {b.output[r], b.partial[r], b.streams[r]});
    }
    check(cudaStreamEndCapture(b.streams[r], &graph[r]));
    check(cudaGraphInstantiate(&exec[r], graph[r], nullptr, nullptr, 0));
  }
  for (int iteration = 0; iteration < 137; ++iteration) {
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      fill<<<4, 128, 0, b.streams[r]>>>(b.input[r], n, r, iteration);
      if (iteration % 17 == 0 && r == iteration % ranks)
        lag<<<1, 1, 0, b.streams[r]>>>();
      check(cudaGraphLaunch(exec[r], b.streams[r]));
    }
    b.sync();
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      auto actual = read(b.partial[r], n);
      for (size_t i = 0; i < n; ++i)
        require(
            actual[i] ==
                bits(float(ranks * ((int(i % 11) - 5) + r + iteration % 7))),
            "mixed graph replay differs from oracle");
    }
  }
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    check(cudaGraphExecDestroy(exec[r]));
    check(cudaGraphDestroy(graph[r]));
  }
  std::printf(
      "PASS: 137 unchanged graph replays x 6 mixed collectives, changing "
      "inputs, ranks=%d\n",
      ranks);
}

__global__ void hold(unsigned* release) {
  unsigned v;
  do {
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(release)
                 : "memory");
    __nanosleep(128);
  } while (!v);
}
__global__ void transform(Elem const* input, Elem* output, size_t n,
                          int expertRank) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    output[i] =
        __float2bfloat16_rn(__bfloat162float(input[i]) * (expertRank + 1));
}
void overlap(std::vector<int> const& devices) {
  int ranks = devices.size();
  if (ranks < 2) return;
  size_t n = 777;
  Buffers b(devices, n);
  EpTransport ep(devices, {Transfer::DmaPrefill, n, 127});
  std::vector<cudaEvent_t> produced(ranks * ranks), returned(ranks),
      sourceReady(ranks);
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    fill<<<4, 128, 0, b.streams[r]>>>(b.input[r], n, r, 0);
    // Load the callback kernel before deliberately holding a GPU. CUDA lazy
    // module loading may require quiescence and is not communication blocking.
    transform<<<4, 128, 0, b.streams[r]>>>(b.input[r], b.partial[r], n, r);
    for (int s = 0; s < ranks; ++s)
      check(cudaEventCreateWithFlags(&produced[r * ranks + s],
                                     cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&returned[r], cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&sourceReady[r], cudaEventDisableTiming));
  }
  b.sync();
  unsigned* release;
  check(cudaHostAlloc(&release, sizeof(unsigned),
                      cudaHostAllocMapped | cudaHostAllocPortable));
  *release = 0;
  int delayed = ranks - 1;
  check(cudaSetDevice(devices[delayed]));
  unsigned* mapped;
  check(cudaHostGetDevicePointer(&mapped, release, 0));
  hold<<<1, 1, 0, b.streams[delayed]>>>(mapped);
  std::vector<Binding> gather(ranks), scatter(ranks);
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    check(cudaEventRecord(sourceReady[r], b.streams[r]));
    gather[r] = {b.input[r], b.output[r], b.streams[r], {sourceReady[r]}};
    scatter[r] = {b.partial[r], b.input[r], b.streams[r]};
  }
  bool progressed = false;
  try {
    ep.enqueue(Collective::AllGather, n, gather,
               [&](int r, int s, void const* shard, size_t count,
                   cudaStream_t stream) {
                 transform<<<4, 128, 0, stream>>>(
                     static_cast<Elem const*>(shard), b.partial[r] + s * n,
                     count, r);
                 check(cudaEventRecord(produced[r * ranks + s], stream));
               });
    for (int r = 0; r < ranks; ++r)
      scatter[r].inputReady.assign(produced.begin() + r * ranks,
                                   produced.begin() + (r + 1) * ranks);
    ep.enqueue(Collective::ReduceScatter, n, scatter, {},
               [&](int r, void const*, size_t, cudaStream_t stream) {
                 check(cudaEventRecord(returned[r], stream));
               });
    // Rank 0's complete dispatch -> compute -> combine must finish while the
    // last rank's dispatch is deliberately held indefinitely. Release on every
    // error path so a failing test cannot strand device work in destruction.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    check(cudaSetDevice(devices[0]));
    while (std::chrono::steady_clock::now() < deadline) {
      auto status = cudaEventQuery(returned[0]);
      if (status == cudaSuccess) {
        progressed = true;
        break;
      }
      if (status != cudaErrorNotReady) check(status);
      std::this_thread::yield();
    }
  } catch (...) {
    __atomic_store_n(release, 1U, __ATOMIC_RELEASE);
    b.sync();
    cudaFreeHost(release);
    throw;
  }
  if (!progressed) {
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      std::fprintf(stderr, "held-source progress rank %d: produced", r);
      for (int s = 0; s < ranks; ++s)
        std::fprintf(stderr, " %d",
                     int(cudaEventQuery(produced[r * ranks + s])));
      std::fprintf(stderr, "; returned=%d\n", int(cudaEventQuery(returned[r])));
    }
  }
  __atomic_store_n(release, 1U, __ATOMIC_RELEASE);
  b.sync();
  check(cudaFreeHost(release));
  require(progressed,
          "ready sender could not complete its round trip while another sender "
          "was held");
  for (int r = 0; r < ranks; ++r) {
    check(cudaSetDevice(devices[r]));
    auto actual = read(b.input[r], n);
    for (size_t i = 0; i < n; ++i)
      require(actual[i] == bits(float(ranks * (ranks + 1) / 2 *
                                      ((int(i % 11) - 5) + r))),
              "overlapped round trip differs from oracle");
    for (int s = 0; s < ranks; ++s)
      check(cudaEventDestroy(produced[r * ranks + s]));
    check(cudaEventDestroy(returned[r]));
    check(cudaEventDestroy(sourceReady[r]));
  }
  std::printf(
      "PASS: ready sender completes dispatch/compute/combine before delayed "
      "sender is released, ranks=%d\n",
      ranks);
}

void invalid(std::vector<int> const& devices) {
  int rejected = 0;
  auto rejects = [&](auto fn) {
    try {
      fn();
    } catch (std::invalid_argument const&) {
      ++rejected;
      return;
    }
    throw std::runtime_error("invalid configuration accepted");
  };
  rejects([&] { EpTransport x({}, {}); });
  rejects([&] { EpTransport x({devices[0], devices[0]}, {}); });
  rejects([&] { EpTransport x(devices, {Transfer::SmDecode, 16, 17}); });
  Buffers b(devices, 16);
  EpTransport x(devices, {Transfer::SmDecode, 16, 8});
  rejects([&] { x.enqueue(Collective::AllGather, 16, {}); });
  rejects([&] { x.enqueueRank(-1, Collective::AllGather, 16, {}); });
  rejects([&] {
    x.enqueueRank(0, Collective::AllGather, 0,
                  {b.input[0], b.output[0], b.streams[0]});
  });
  if (devices.size() > 1)
    rejects([&] {
      x.enqueueRank(0, Collective::AllGather, 16,
                    {b.input[0] + 1, b.input[0], b.streams[0]});
    });
  rejects([&] {
    x.enqueueRank(0, Collective::AllGather, 16,
                  {b.input[0], b.output[0], b.streams[0], {nullptr}});
  });
  rejects([&] {
    x.enqueueRank(0, static_cast<Collective>(99), 16,
                  {b.input[0], b.output[0], b.streams[0]});
  });
  std::vector<Binding> bindings(devices.size());
  for (size_t r = 0; r < devices.size(); ++r)
    bindings[r] = {b.input[r], b.output[r], b.streams[r]};
  bindings.back().output = nullptr;
  // A bad LAST rank must be rejected before the first rank is launched.
  rejects([&] { x.enqueue(Collective::AllGather, 16, bindings); });
  if (devices.size() > 1) {
    rejects([&] {
      x.enqueueRank(0, Collective::AllGather, 16,
                    {b.input[1], b.output[0], b.streams[0]});
    });
    rejects([&] {
      x.enqueueRank(0, Collective::AllGather, 16,
                    {b.input[0], b.output[0], b.streams[1]});
    });
  }
  EpTransport dma(devices, {Transfer::DmaPrefill, 16, 8});
  rejects([&] {
    dma.enqueueRank(0, Collective::AllGather, 16,
                    {b.input[0], b.output[0], b.streams[0]});
  });
  std::printf("PASS: %d invalid options/bindings rejected before submission\n",
              rejected);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int ranks = 4;
    bool reverse = false;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--gpus" && i + 1 < argc)
        ranks = std::stoi(argv[++i]);
      else if (arg == "--reverse")
        reverse = true;
      else
        throw std::invalid_argument("expected --gpus N [--reverse]");
    }
    require(ranks >= 1 && ranks <= 4, "host oracle supports 1..4 GPUs");
    std::vector<int> devices(ranks);
    std::iota(devices.begin(), devices.end(), 0);
    if (reverse) std::reverse(devices.begin(), devices.end());
    invalid(devices);
    eager(devices, Transfer::DmaPrefill);
    eager(devices, Transfer::SmDecode);
    graphs(devices);
    overlap(devices);
    std::puts("PASS: host-staged reference transport checks");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
