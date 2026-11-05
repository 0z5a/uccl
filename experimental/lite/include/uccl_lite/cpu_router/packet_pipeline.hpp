#pragma once
#include "channel.hpp"
#include "router_core.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace uccl_lite::cpu_router {
// Rank masks use uint32_t; capacities are bounds, not the active world size.
inline constexpr int kMaxRanks = 32;
// CUDA NUMA hint, PCI sysfs fallback, then the first online host node.
int gpuNumaNode(int device);
struct PacketPort {
  void const* input;
  void* output;
  cudaStream_t stream;
  int device;
};
enum class PacketTransform : uint32_t { Sum, Copy };
enum class PacketLayout : uint32_t { Replicate, Concatenate, Scatter };

#ifdef __CUDACC__
#define UCCL_LITE_HD __host__ __device__
#else
#define UCCL_LITE_HD
#endif
struct PacketUpload {
  size_t inputOffset = 0, packetOffset = 0, elements = 0;
};
struct PacketRoute {
  size_t sourceOffset = 0, outputOffset = 0, elements = 0, retainedOffset = 0;
  int sourceRank = -1;
  bool local = false, addRetained = false;
};
struct PacketUploads {
  int count = 0;
  PacketUpload ranges[2];
};
struct PacketFrame {
  size_t begin = 0, elements = 0;
  int uploads = 0, routes = 0;
  PacketUpload upload[2];
  PacketRoute route[kMaxRanks];
};
// BF16 packets. The transport is independent of the reduction/copy operator
// and output placement. Masks permit reduce-to-root and broadcast plans too.
struct PacketProgram {
  PacketTransform transform;
  PacketLayout layout;
  size_t inputElements;
  uint32_t sources = 255;
  uint32_t destinations = 255;
  int ranks = 8;
  static PacketProgram allReduce(size_t elements, int ranks = 8);
  static PacketProgram allGather(size_t contributionElements, int ranks = 8);
  static PacketProgram reduceScatter(size_t shardElements, int ranks = 8);
  static PacketProgram broadcast(size_t elements, int root, int ranks = 8);
  static PacketProgram reduce(size_t elements, int root, int ranks = 8);
  // These scalar routes are shared by host DMA emission and device channels.
  UCCL_LITE_HD PacketUploads uploadRanges(int rank, size_t tile,
                                          size_t extent) const {
    PacketUploads u;
    size_t begin = tile * extent,
           n = extent < inputElements - begin ? extent : inputElements - begin,
           end = begin + n;
    if (!(sources & (1U << rank))) return u;
    if (layout != PacketLayout::Scatter)
      u.ranges[u.count++] = {begin, 0, n};
    else {
      size_t shard = inputElements / ranks, ownLo = size_t(rank) * shard,
             ownHi = ownLo + shard;
      size_t before = end < ownLo ? end : ownLo;
      if (before > begin) u.ranges[u.count++] = {begin, 0, before - begin};
      size_t after = begin > ownHi ? begin : ownHi;
      if (after < end)
        u.ranges[u.count++] = {after, after - begin, end - after};
    }
    return u;
  }
  UCCL_LITE_HD PacketRoute outputRoute(int rank, size_t tile, size_t extent,
                                       int source = 0) const {
    PacketRoute r;
    size_t begin = tile * extent,
           n = extent < inputElements - begin ? extent : inputElements - begin,
           end = begin + n;
    if (!(destinations & (1U << rank))) return r;
    if (layout == PacketLayout::Concatenate)
      return {source == rank ? begin : 0,
              size_t(source) * inputElements + begin,
              n,
              0,
              source,
              source == rank,
              false};
    if (layout == PacketLayout::Scatter) {
      size_t shard = inputElements / ranks,
             lo = begin > size_t(rank) * shard ? begin : size_t(rank) * shard,
             hi = end < size_t(rank + 1) * shard ? end
                                                 : size_t(rank + 1) * shard;
      if (lo < hi)
        return {lo - begin, lo - size_t(rank) * shard,    hi - lo, lo, -1,
                false,      (sources & (1U << rank)) != 0};
      return r;
    }
    if (transform == PacketTransform::Copy) {
      source = 0;
      while (!(sources & (1U << source))) ++source;
      return {source == rank ? begin : 0,
              begin,
              n,
              0,
              source,
              source == rank,
              false};
    }
    return {0, begin, n, 0, -1, false, false};
  }
  UCCL_LITE_HD PacketFrame frame(int rank, size_t tile, size_t extent) const {
    PacketFrame f;
    f.begin = tile * extent;
    f.elements =
        extent < inputElements - f.begin ? extent : inputElements - f.begin;
    auto ranges = uploadRanges(rank, tile, extent);
    f.uploads = ranges.count;
    for (int i = 0; i < ranges.count; ++i) f.upload[i] = ranges.ranges[i];
    int count = layout == PacketLayout::Concatenate ? ranks : 1;
    for (int i = 0; i < count; ++i) {
      auto route = outputRoute(rank, tile, extent, i);
      if (route.elements) f.route[f.routes++] = route;
    }
    return f;
  }
};
struct PacketPipelineOptions {
  TransportBackend backend = TransportBackend::GpuKernel;
  size_t maxTensorBytes = 1UL << 30;
  size_t packetBytes = 512UL << 10;
  size_t minPacketBytes = 128UL << 10;
  int transportBlocks = 64;
  int transportThreads = 256;
  size_t transportGrainBytes = 16UL << 10;
  int workersPerNode = 23;
  size_t grainElements = 8192;
  // Optional group-major affinity, groups ordered by first device occurrence.
  // Empty discovers NUMA-local online CPUs and reserves one CPU per group.
  std::vector<int> workerCpus;
  // Optional physical NUMA node per rank. Empty queries CUDA then PCI sysfs.
  std::vector<int> numaNodes;
  // -1 selects an unused NUMA-local CPU where possible.
  int replayCpu = -1;
  // Publish <=128 captured commands before launching their GPU graphs. Larger
  // captures use the replay helper so publication cannot exhaust GPU credits.
  bool inlineReplay = false;
  // Low-occupancy persistent transport alongside an independent producer.
  // Caller must establish input readiness with CUDA-visible dependencies.
  bool concurrentProducer = false;
  // Independent GEMM tile override: (0,0) chooses by shape; explicit pairs are
  // (32,128), (64,128), (128,128), (128,256), (256,128).
  // 32x128 needs K/stages=64/2; 128x128 needs K/stages=32/3.
  // packetBytes remains independent.
  int gemmTileM = 0;
  int gemmTileN = 0;
  // Optional matched K/stage pipeline: (0,0) or (32,4), and (64,2).
  // The 128x128 tile instead uses (32,3) for two-CTA residency.
  // Both use the same shared-memory footprint for a selected M/N tile.
  int gemmTileK = 0;
  int gemmStages = 0;
  // 0 retains the cache-size policy; a positive value groups adjacent M tiles.
  int gemmMGroup = 0;
  // ReduceScatter final results are owned by destinations. Cap active CPU lanes
  // separately from the executor pool to avoid waking every AllReduce worker.
  // Zero uses the full workersPerNode limit; positive values are upper bounds.
  int scatterWorkersPerNode = 16;
};

// One process, 1..kMaxRanks ports in arbitrary device/NUMA order. All GPU
// transfers go through mapped pinned packets. No NCCL collective implementation
// is called by this core.
class PacketPipeline : public RouterCore {
 public:
  explicit PacketPipeline(PacketPipelineOptions options = {});
  ~PacketPipeline();
  PacketPipeline(PacketPipeline const&) = delete;
  PacketPipeline& operator=(PacketPipeline const&) = delete;
  void prepare(std::vector<PacketPort> const& ports);
  void enqueue(PacketProgram const& plan, std::vector<PacketPort> const& ports);
  struct Binding {
    explicit Binding(size_t ranks = 8)
        : inputs(ranks),
          outputs(ranks),
          streams(ranks),
          devices(ranks),
          producerEvents(ranks) {}
    std::vector<MemorySpan> inputs, outputs;
    std::vector<cudaStream_t> streams;
    std::vector<int> devices;
    // Optional event per contiguous input stripe. Each persistent epoch only
    // touches packets whose intersecting stripes have completed. Events outlive
    // captured graphs and this binding. One CPU job spans all epochs.
    std::vector<std::vector<cudaEvent_t>> producerEvents;
    size_t producerElements = 0;
  };
  void enqueue(PacketProgram const& program, Binding const& binding);
  // Resident CUTLASS producer and CPU-router transport in one cooperative grid.
  // A fixed shape, one set per rank of BF16 A[M,K], B[N,K], partial[M,N]
  // operands.
  void prepareGemm(int m, int n, int k, std::vector<MemorySpan> const& a,
                   std::vector<MemorySpan> const& b,
                   std::vector<MemorySpan> const& partial,
                   PacketLayout layout = PacketLayout::Replicate);
  // AllGather local A[localM,K] into gatheredA[localM*ranks,K], with
  // resident workers consuming completed input row tiles in the same grid.
  void prepareAllGatherGemm(int localM, int n, int k,
                            std::vector<MemorySpan> const& localA,
                            std::vector<MemorySpan> const& gatheredA,
                            std::vector<MemorySpan> const& b,
                            std::vector<MemorySpan> const& output);
  void enqueueGemm(PacketProgram const& program, Binding const& binding);
  void enqueueGemmProducerOnly(std::vector<cudaStream_t> const& streams);
  // Source compatibility for callers with fixed std::array port sets.
  template <size_t N>
  void prepare(std::array<PacketPort, N> const& ports) {
    prepare(std::vector<PacketPort>(ports.begin(), ports.end()));
  }
  template <size_t N>
  void enqueue(PacketProgram const& plan,
               std::array<PacketPort, N> const& ports) {
    enqueue(plan, std::vector<PacketPort>(ports.begin(), ports.end()));
  }
  template <size_t N>
  void prepareGemm(int m, int n, int k, std::array<MemorySpan, N> const& a,
                   std::array<MemorySpan, N> const& b,
                   std::array<MemorySpan, N> const& partial) {
    prepareGemm(m, n, k, std::vector<MemorySpan>(a.begin(), a.end()),
                std::vector<MemorySpan>(b.begin(), b.end()),
                std::vector<MemorySpan>(partial.begin(), partial.end()));
  }
  template <size_t N>
  void enqueueGemmProducerOnly(std::array<cudaStream_t, N> const& streams) {
    enqueueGemmProducerOnly(
        std::vector<cudaStream_t>(streams.begin(), streams.end()));
  }
  // Record an externally captured program. GPU graphs belong to the caller.
  // beginReplay publishes fresh jobs asynchronously; launch all GPU graphs,
  // synchronize their streams, then call endReplay before reusing the program.
  void beginCapture();
  void endCapture();
  void beginReplay();
  void endReplay();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
using RouterPort = PacketPort;
using RouterTransform = PacketTransform;
using RouterLayout = PacketLayout;
using RouterPlan = PacketProgram;
using CpuRouterOptions = PacketPipelineOptions;
using CpuRouter = PacketPipeline;
}  // namespace uccl_lite::cpu_router
