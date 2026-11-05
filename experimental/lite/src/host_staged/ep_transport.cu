#include "uccl_lite/host_staged/ep_transport.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace uccl_lite::host_staged {
namespace {
constexpr int kMaxRanks = 16;
constexpr int kThreads = 256;
using Elem = __nv_bfloat16;

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
// Restore the caller's current device, including on exceptions.
struct DeviceGuard {
  int device;
  DeviceGuard() { check(cudaGetDevice(&device)); }
  ~DeviceGuard() { cudaSetDevice(device); }
};
struct alignas(128) Signal {
  unsigned long long value;
};
struct Wire {
  Signal ready[2][kMaxRanks][kMaxRanks];
  Signal credit[2][kMaxRanks][kMaxRanks];
};
struct State {
  unsigned long long next = 1;
};
struct DeviceView {
  Elem* staging;
  Elem* scratch;
  Wire* wire;
  State* state;
  size_t chunk, maxShard;
  int ranks, rank;
};
__host__ __device__ size_t bankOffset(int slot, int source, int destination,
                                      int ranks, size_t chunk) {
  return ((size_t(slot) * ranks + source) * ranks + destination) * chunk;
}
// System-scoped acquire/release loads/stores, not read-modify-write atomics.
// Each ready/credit word has exactly one writer. Payload and flags are
// separate.
__device__ unsigned long long load(Signal const* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(&p->value)
               : "memory");
  return v;
}
__device__ void publish(Signal* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(&p->value), "l"(v)
               : "memory");
}
__device__ void wait(Signal const* p, unsigned long long v) {
  while (load(p) < v) __nanosleep(128);
}

// A cooperative grid guarantees that both directional contexts are resident.
// Only a CTA leader polls each remote sender. Its other threads wait in a
// GPU-local barrier. Producers and receivers independently loop over chunks;
// there is no all-sender gate between chunks or before a peer's receive.
__global__ void smCollective(DeviceView v, Collective op, size_t count,
                             Elem const* input, Elem* output) {
  auto grid = cooperative_groups::this_grid();
  unsigned long long first = v.state->next;
  size_t tiles = (count + v.chunk - 1) / v.chunk;
  if (first > ULLONG_MAX - tiles) asm volatile("trap;");
  int role = blockIdx.x;
  bool gather = op == Collective::AllGather;
  // First ranks CTAs produce; second ranks CTAs consume. Local rank is skipped
  // for RS, and only producer rank 0 is used for shared AG upload.
  if (role < v.ranks && (gather ? role == 0 : role != v.rank)) {
    int destination = gather ? 0 : role;
    for (size_t t = 0; t < tiles; ++t) {
      auto seq = first + t;
      int slot = seq & 1;
      if (seq > 2) {
        if (gather) {
          if (threadIdx.x < v.ranks && threadIdx.x != v.rank)
            wait(&v.wire->credit[slot][v.rank][threadIdx.x], seq - 2);
        } else if (threadIdx.x == 0)
          wait(&v.wire->credit[slot][v.rank][destination], seq - 2);
      }
      __syncthreads();
      size_t begin = t * v.chunk, n = min(v.chunk, count - begin);
      Elem volatile* dst =
          v.staging + bankOffset(slot, v.rank, destination, v.ranks, v.chunk);
      auto* src = input + (gather ? 0 : destination * count) + begin;
      for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        reinterpret_cast<unsigned short volatile*>(dst)[i] =
            reinterpret_cast<unsigned short const*>(src)[i];
      // Collect all threads' system-visible writes before the single publisher.
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0)
        publish(&v.wire->ready[slot][v.rank][destination], seq);
      __syncthreads();
    }
  } else if (role >= v.ranks) {
    int source = role - v.ranks;
    if (source != v.rank) {
      for (size_t t = 0; t < tiles; ++t) {
        auto seq = first + t;
        int slot = seq & 1, destination = gather ? 0 : v.rank;
        if (threadIdx.x == 0)
          wait(&v.wire->ready[slot][source][destination], seq);
        __syncthreads();
        size_t begin = t * v.chunk, n = min(v.chunk, count - begin);
        auto* src =
            v.staging + bankOffset(slot, source, destination, v.ranks, v.chunk);
        auto* dst = gather ? output + source * count + begin
                           : v.scratch + source * v.maxShard + begin;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x)
          reinterpret_cast<unsigned short*>(dst)[i] =
              reinterpret_cast<unsigned short const volatile*>(src)[i];
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0)
          publish(&v.wire->credit[slot][source][v.rank], seq);
        __syncthreads();
      }
    } else if (gather && output + source * count != input) {
      for (size_t i = threadIdx.x; i < count; i += blockDim.x)
        output[source * count + i] = input[i];
    }
  }
  grid.sync();
  if (!gather) {
    for (size_t i = grid.thread_rank(); i < count; i += grid.size()) {
      float sum = 0;
      for (int source = 0; source < v.ranks; ++source)
        sum += __bfloat162float(source == v.rank
                                    ? input[v.rank * count + i]
                                    : v.scratch[source * v.maxShard + i]);
      output[i] = __float2bfloat16_rn(sum);
    }
  }
  grid.sync();
  if (grid.thread_rank() == 0) v.state->next = first + tiles;
}

__global__ void reduce(Elem const* input, Elem const* scratch, Elem* output,
                       size_t count, size_t maxShard, int rank, int ranks) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    float sum = 0;
    for (int s = 0; s < ranks; ++s)
      sum += __bfloat162float(s == rank ? input[rank * count + i]
                                        : scratch[s * maxShard + i]);
    output[i] = __float2bfloat16_rn(sum);
  }
}

// Copy engines move only the payload. Tiny stream-ordered control kernels
// publish completion after D2H, wait for a single source before H2D, and return
// reader credits after H2D. No CPU polling and no cross-device CUDA event ABI.
__global__ void dmaAcquire(Wire* wire, unsigned long long seq, int source,
                           int destination, int ranks, bool gather) {
  if (seq <= 2) return;
  int reader = gather ? int(threadIdx.x) : destination;
  if ((gather ? reader < ranks : threadIdx.x == 0) && reader != source)
    wait(&wire->credit[seq & 1][source][reader], seq - 2);
}
__global__ void dmaSignal(Signal* signal, unsigned long long seq,
                          bool awaiting) {
  if (awaiting)
    wait(signal, seq);
  else
    publish(signal, seq);
}

struct Rank {
  int device = -1;
  std::vector<cudaStream_t> upload, download, combineDownload, compute;
  std::vector<cudaEvent_t> arrival, computed, uploadDone;
  cudaStream_t reduction = nullptr;
  cudaStream_t gatherUpload = nullptr;
  cudaEvent_t entry = nullptr, reduced = nullptr;
  Elem* scratch = nullptr;
  State* state = nullptr;
  DeviceView view{};
  Wire* gatherWire = nullptr;
  bool reductionRecorded = false;
};
}  // namespace

struct EpTransport::Impl {
  Options options;
  int ranks;
  std::vector<Rank> rank;
  Elem* staging = nullptr;
  Elem* gatherStaging = nullptr;
  Wire* wire = nullptr;
  Wire* gatherWire = nullptr;
  unsigned long long dmaNext = 1;
  unsigned long long gatherNext = 1;

  Impl(std::vector<int> const& devices, Options opt)
      : options(opt), ranks(devices.size()), rank(ranks) {
    if (!ranks || ranks > kMaxRanks || !opt.chunkElements ||
        !opt.maxShardElements || opt.chunkElements > opt.maxShardElements ||
        (opt.transfer != Transfer::DmaPrefill &&
         opt.transfer != Transfer::SmDecode))
      throw std::invalid_argument("invalid host-staged options");
    size_t multiplier = size_t(2) * ranks * ranks * sizeof(Elem);
    if (opt.chunkElements > SIZE_MAX / multiplier ||
        opt.maxShardElements > SIZE_MAX / (sizeof(Elem) * ranks))
      throw std::invalid_argument("host-staged extent overflow");
    auto sorted = devices;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      throw std::invalid_argument(
          "device list must be distinct and nonnegative");
    DeviceGuard guard;
    try {
      check(cudaHostAlloc(&staging, opt.chunkElements * multiplier,
                          cudaHostAllocPortable | cudaHostAllocMapped));
      if (opt.transfer == Transfer::DmaPrefill)
        check(cudaHostAlloc(&gatherStaging,
                            opt.chunkElements * sizeof(Elem) * 2 * ranks,
                            cudaHostAllocPortable | cudaHostAllocMapped));
      check(cudaHostAlloc(&wire, sizeof(Wire),
                          cudaHostAllocPortable | cudaHostAllocMapped));
      *wire = {};
      if (opt.transfer == Transfer::DmaPrefill) {
        check(cudaHostAlloc(&gatherWire, sizeof(Wire),
                            cudaHostAllocPortable | cudaHostAllocMapped));
        *gatherWire = {};
      }
      for (int r = 0; r < ranks; ++r) {
        auto& p = rank[r];
        p.device = devices[r];
        check(cudaSetDevice(p.device));
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, p.device));
        if (!prop.canMapHostMemory || prop.major < 7)
          throw std::invalid_argument("requires mapped host memory and sm_70+");
        check(cudaMalloc(&p.scratch,
                         opt.maxShardElements * ranks * sizeof(Elem)));
        if (opt.transfer == Transfer::SmDecode) {
          if (!prop.cooperativeLaunch)
            throw std::invalid_argument("SM path requires cooperative launch");
          int occupancy = 0;
          check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &occupancy, smCollective, kThreads, 0));
          if (occupancy * prop.multiProcessorCount < 2 * ranks)
            throw std::invalid_argument("cooperative grid cannot be resident");
          check(cudaMalloc(&p.state, sizeof(State)));
          State initial;
          check(cudaMemcpy(p.state, &initial, sizeof(initial),
                           cudaMemcpyHostToDevice));
        }
        Elem* mapped;
        Wire* flags;
        check(cudaHostGetDevicePointer(&mapped, staging, 0));
        check(cudaHostGetDevicePointer(&flags, wire, 0));
        p.view = {mapped,
                  p.scratch,
                  flags,
                  p.state,
                  opt.chunkElements,
                  opt.maxShardElements,
                  ranks,
                  r};
        if (opt.transfer == Transfer::SmDecode) continue;
        check(cudaHostGetDevicePointer(&p.gatherWire, gatherWire, 0));
        // Force-load transport kernels before any rank can enter a wait. Users
        // must likewise warm their callback kernels (or use eager CUDA
        // loading).
        cudaFuncAttributes attributes;
        check(cudaFuncGetAttributes(&attributes, dmaAcquire));
        check(cudaFuncGetAttributes(&attributes, dmaSignal));
        check(cudaFuncGetAttributes(&attributes, reduce));
        p.upload.resize(ranks);
        p.download.resize(ranks);
        p.combineDownload.resize(ranks);
        p.compute.resize(ranks);
        p.arrival.resize(ranks);
        p.computed.resize(ranks);
        p.uploadDone.resize(ranks);
        for (int s = 0; s < ranks; ++s) {
          check(cudaStreamCreateWithFlags(&p.upload[s], cudaStreamNonBlocking));
          check(
              cudaStreamCreateWithFlags(&p.download[s], cudaStreamNonBlocking));
          check(cudaStreamCreateWithFlags(&p.combineDownload[s],
                                          cudaStreamNonBlocking));
          check(
              cudaStreamCreateWithFlags(&p.compute[s], cudaStreamNonBlocking));
          check(
              cudaEventCreateWithFlags(&p.arrival[s], cudaEventDisableTiming));
          check(
              cudaEventCreateWithFlags(&p.computed[s], cudaEventDisableTiming));
          check(cudaEventCreateWithFlags(&p.uploadDone[s],
                                         cudaEventDisableTiming));
        }
        check(cudaStreamCreateWithFlags(&p.reduction, cudaStreamNonBlocking));
        check(
            cudaStreamCreateWithFlags(&p.gatherUpload, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&p.entry, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&p.reduced, cudaEventDisableTiming));
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void synchronize() {
    DeviceGuard guard;
    for (auto const& p : rank)
      if (p.device >= 0) {
        check(cudaSetDevice(p.device));
        check(cudaDeviceSynchronize());
      }
  }
  void cleanup() noexcept {
    int original = 0;
    cudaGetDevice(&original);
    for (auto const& p : rank)
      if (p.device >= 0) {
        cudaSetDevice(p.device);
        cudaDeviceSynchronize();
      }
    for (int r = 0; r < ranks; ++r) {
      auto const& p = rank[r];
      if (p.device < 0) continue;
      cudaSetDevice(p.device);
      for (auto const* streams :
           {&p.upload, &p.download, &p.combineDownload, &p.compute})
        for (auto s : *streams)
          if (s) cudaStreamDestroy(s);
      if (p.reduction) cudaStreamDestroy(p.reduction);
      if (p.gatherUpload) cudaStreamDestroy(p.gatherUpload);
      for (auto const* events : {&p.arrival, &p.computed, &p.uploadDone})
        for (auto e : *events)
          if (e) cudaEventDestroy(e);
      if (p.entry) cudaEventDestroy(p.entry);
      if (p.reduced) cudaEventDestroy(p.reduced);
      if (p.scratch) cudaFree(p.scratch);
      if (p.state) cudaFree(p.state);
    }
    if (staging) cudaFreeHost(staging);
    if (gatherStaging) cudaFreeHost(gatherStaging);
    if (wire) cudaFreeHost(wire);
    if (gatherWire) cudaFreeHost(gatherWire);
    cudaSetDevice(original);
  }
  ~Impl() { cleanup(); }

  void validate(int r, Collective op, size_t n, Binding const& b) const {
    if (r < 0 || r >= ranks || !n || n > options.maxShardElements || !b.input ||
        !b.output ||
        (op != Collective::AllGather && op != Collective::ReduceScatter))
      throw std::invalid_argument("invalid host-staged collective binding");
    bool gather = op == Collective::AllGather;
    size_t inBytes = n * sizeof(Elem) * (gather ? 1 : ranks);
    size_t outBytes = n * sizeof(Elem) * (gather ? ranks : 1);
    auto in = reinterpret_cast<uintptr_t>(b.input);
    auto out = reinterpret_cast<uintptr_t>(b.output);
    if ((in | out) % alignof(Elem) || in > UINTPTR_MAX - inBytes ||
        out > UINTPTR_MAX - outBytes)
      throw std::invalid_argument("misaligned or overflowing buffer range");
    bool overlap = in < out + outBytes && out < in + inBytes;
    bool inplace = gather ? in == out + r * n * sizeof(Elem)
                          : out == in + r * n * sizeof(Elem);
    if (overlap && !inplace)
      throw std::invalid_argument("unsupported overlapping buffers");
    if (!b.inputReady.empty()) {
      if (options.transfer != Transfer::DmaPrefill ||
          b.inputReady.size() != size_t(gather ? 1 : ranks) ||
          std::any_of(b.inputReady.begin(), b.inputReady.end(),
                      [](cudaEvent_t e) { return !e; }))
        throw std::invalid_argument("invalid per-destination readiness events");
    }
    DeviceGuard guard;
    check(cudaSetDevice(rank[r].device));
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(b.stream, &capture));
    // CUDA 12.9 disallows some allocation/device queries during capture.
    // Validate addresses in an eager warmup before capturing the fixed binding.
    if (capture != cudaStreamCaptureStatusNone) return;
    int streamDevice = -1;
    check(cudaStreamGetDevice(b.stream, &streamDevice));
    if (streamDevice != rank[r].device)
      throw std::invalid_argument("binding stream belongs to another device");
    for (void const* pointer : {b.input, static_cast<void const*>(b.output)}) {
      cudaPointerAttributes attributes;
      check(cudaPointerGetAttributes(&attributes, pointer));
      if (attributes.type != cudaMemoryTypeDevice ||
          attributes.device != rank[r].device)
        throw std::invalid_argument("binding must use local device memory");
    }
  }
  void sm(int r, Collective op, size_t n, Binding const& b) {
    validate(r, op, n, b);
    DeviceGuard guard;
    check(cudaSetDevice(rank[r].device));
    auto view = rank[r].view;
    auto* input = static_cast<Elem const*>(b.input);
    auto* output = static_cast<Elem*>(b.output);
    void* args[] = {&view, &op, &n, &input, &output};
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(smCollective),
                                      dim3(2 * ranks), dim3(kThreads), args, 0,
                                      b.stream));
  }
  void dma(Collective op, size_t n, std::vector<Binding> const& bindings,
           GatherConsumer const& consumer,
           ScatterConsumer const& reducedConsumer) {
    DeviceGuard guard;
    bool gather = op == Collective::AllGather;
    auto& next = gather ? gatherNext : dmaNext;
    size_t tiles = (n + options.chunkElements - 1) / options.chunkElements;
    if (next > ULLONG_MAX - tiles)
      throw std::overflow_error("DMA epoch exhausted");
    // Per-peer queues prevent an unready sender from blocking ready peers on a
    // shared stream. These are independent upload/download contexts, not peer
    // transfers. The paper does not publish its precise queue assignment.
    for (int r = 0; r < ranks; ++r) {
      auto& p = rank[r];
      auto const& b = bindings[r];
      check(cudaSetDevice(p.device));
      if (b.inputReady.empty()) check(cudaEventRecord(p.entry, b.stream));
      for (int s = 0; s < ranks; ++s) {
        auto ready =
            b.inputReady.empty() ? p.entry : b.inputReady[gather ? 0 : s];
        if (!gather)
          check(cudaStreamWaitEvent(p.upload[s], ready, 0));
        else if (s == 0)
          check(cudaStreamWaitEvent(p.gatherUpload, ready, 0));
        auto incoming = gather ? p.download[s] : p.combineDownload[s];
        if (b.inputReady.empty())
          check(cudaStreamWaitEvent(incoming, p.entry, 0));
        // Scratch must not be overwritten until the preceding reduction read
        // it.
        if (!gather && p.reductionRecorded)
          check(cudaStreamWaitEvent(incoming, p.reduced, 0));
      }
      auto ownReady =
          b.inputReady.empty() ? p.entry : b.inputReady[gather ? 0 : r];
      if (!gather) check(cudaStreamWaitEvent(p.reduction, ownReady, 0));
      if (gather) {
        check(cudaStreamWaitEvent(p.download[r], ownReady, 0));
        auto* local = static_cast<Elem*>(b.output) + r * n;
        if (local != b.input)
          check(cudaMemcpyAsync(local, b.input, n * sizeof(Elem),
                                cudaMemcpyDeviceToDevice, p.download[r]));
      }
    }
    for (size_t t = 0; t < tiles; ++t) {
      auto seq = next + t;
      int slot = seq & 1;
      size_t begin = t * options.chunkElements;
      size_t count = std::min(options.chunkElements, n - begin);
      for (int s = 0; s < ranks; ++s) {
        auto& p = rank[s];
        check(cudaSetDevice(p.device));
        for (int d = 0; d < (gather ? 1 : ranks); ++d) {
          if (!gather && s == d) continue;
          auto stream = gather ? p.gatherUpload : p.upload[d];
          auto* flags = gather ? p.gatherWire : p.view.wire;
          dmaAcquire<<<1, 32, 0, stream>>>(flags, seq, s, d, ranks, gather);
          auto* dst = gather ? gatherStaging + (size_t(slot) * ranks + s) *
                                                   options.chunkElements
                             : staging + bankOffset(slot, s, d, ranks,
                                                    options.chunkElements);
          auto* src = static_cast<Elem const*>(bindings[s].input) +
                      (gather ? 0 : d * n) + begin;
          check(cudaMemcpyAsync(dst, src, count * sizeof(Elem),
                                cudaMemcpyDeviceToHost, stream));
          dmaSignal<<<1, 1, 0, stream>>>(&flags->ready[slot][s][d], seq, false);
          check(cudaGetLastError());
        }
      }
      for (int d = 0; d < ranks; ++d) {
        auto& p = rank[d];
        check(cudaSetDevice(p.device));
        for (int s = 0; s < ranks; ++s) {
          if (s == d) continue;
          auto stream = gather ? p.download[s] : p.combineDownload[s];
          auto* flags = gather ? p.gatherWire : p.view.wire;
          dmaSignal<<<1, 1, 0, stream>>>(&flags->ready[slot][s][gather ? 0 : d],
                                         seq, true);
          auto* src = gather ? gatherStaging + (size_t(slot) * ranks + s) *
                                                   options.chunkElements
                             : staging + bankOffset(slot, s, d, ranks,
                                                    options.chunkElements);
          auto* dst =
              gather ? static_cast<Elem*>(bindings[d].output) + s * n + begin
                     : p.scratch + s * options.maxShardElements + begin;
          check(cudaMemcpyAsync(dst, src, count * sizeof(Elem),
                                cudaMemcpyHostToDevice, stream));
          dmaSignal<<<1, 1, 0, stream>>>(&flags->credit[slot][s][d], seq,
                                         false);
          check(cudaGetLastError());
        }
      }
    }
    for (int r = 0; r < ranks; ++r) {
      auto& p = rank[r];
      auto const& b = bindings[r];
      check(cudaSetDevice(p.device));
      for (int s = 0; s < ranks; ++s) {
        check(cudaEventRecord(p.arrival[s],
                              gather ? p.download[s] : p.combineDownload[s]));
        if (gather && consumer) {
          check(cudaStreamWaitEvent(p.compute[s], p.arrival[s], 0));
          consumer(r, s, static_cast<Elem const*>(b.output) + s * n, n,
                   p.compute[s]);
          check(cudaEventRecord(p.computed[s], p.compute[s]));
          check(cudaStreamWaitEvent(b.stream, p.computed[s], 0));
        } else
          check(cudaStreamWaitEvent(b.stream, p.arrival[s], 0));
        if (!gather) check(cudaStreamWaitEvent(p.reduction, p.arrival[s], 0));
        check(cudaEventRecord(p.uploadDone[s],
                              gather ? p.gatherUpload : p.upload[s]));
        check(cudaStreamWaitEvent(b.stream, p.uploadDone[s], 0));
      }
      if (!gather) {
        reduce<<<std::min(size_t(128), (n + 255) / 256), 256, 0, p.reduction>>>(
            static_cast<Elem const*>(b.input), p.scratch,
            static_cast<Elem*>(b.output), n, options.maxShardElements, r,
            ranks);
        check(cudaGetLastError());
        if (reducedConsumer) reducedConsumer(r, b.output, n, p.reduction);
        check(cudaEventRecord(p.reduced, p.reduction));
        check(cudaStreamWaitEvent(b.stream, p.reduced, 0));
        p.reductionRecorded = true;
      }
    }
    next += tiles;
  }
};

EpTransport::EpTransport(std::vector<int> devices, Options options)
    : impl_(std::make_unique<Impl>(devices, options)) {}
EpTransport::~EpTransport() = default;
void EpTransport::synchronize() { impl_->synchronize(); }
void EpTransport::enqueueRank(int r, Collective op, size_t n,
                              Binding const& b) {
  if (impl_->options.transfer != Transfer::SmDecode)
    throw std::invalid_argument(
        "rank-local capture is supported only in SM mode");
  impl_->sm(r, op, n, b);
}
void EpTransport::enqueue(Collective op, size_t n,
                          std::vector<Binding> const& bindings,
                          GatherConsumer const& consumer,
                          ScatterConsumer const& reducedConsumer) {
  if (bindings.size() != size_t(impl_->ranks) ||
      (consumer && (op != Collective::AllGather ||
                    impl_->options.transfer != Transfer::DmaPrefill)) ||
      (reducedConsumer && (op != Collective::ReduceScatter ||
                           impl_->options.transfer != Transfer::DmaPrefill)))
    throw std::invalid_argument("invalid rank count or gather consumer");
  // Validate ALL ranks before launching any communication that can wait on
  // them.
  for (int r = 0; r < impl_->ranks; ++r) {
    impl_->validate(r, op, n, bindings[r]);
    if (impl_->options.transfer == Transfer::DmaPrefill) {
      DeviceGuard guard;
      check(cudaSetDevice(impl_->rank[r].device));
      cudaStreamCaptureStatus status;
      check(cudaStreamIsCapturing(bindings[r].stream, &status));
      if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument(
            "DMA graph capture is unsupported; use SM decode");
    }
  }
  if (impl_->options.transfer == Transfer::DmaPrefill)
    impl_->dma(op, n, bindings, consumer, reducedConsumer);
  else
    for (int r = 0; r < impl_->ranks; ++r) impl_->sm(r, op, n, bindings[r]);
}
}  // namespace uccl_lite::host_staged
