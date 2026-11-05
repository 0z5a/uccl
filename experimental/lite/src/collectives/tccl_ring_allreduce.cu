// TCCL's two-node x four-GPU AllReduce schedule implemented with MSCCL++.
//
// Unlike multinode_allreduce.cu, this is not hierarchical.  It uses
// the exact global ring selected by the TCCL path database:
//
//   N0.G3 -> N0.G2 -> N0.G0 -> N0.G1 ->
//   N1.G0 -> N1.G1 -> N1.G3 -> N1.G2 -> N0.G3
//
// Each logical channel runs the ordinary eight-rank ring algorithm: seven
// ReduceScatter steps followed by seven AllGather steps.  Local edges use
// CUDA IPC and the two inter-node edges use their NUMA-local IB rail.  Four
// independent MSCCL++ connections preserve TCCL's four-channel concurrency.

#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sched.h>

constexpr int kWorld = 8;
constexpr int kLocalRanks = 4;
constexpr int kMaxChannels = 8;

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e_ = (cmd);                                              \
    if (e_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e_));                              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                    \
  } while (0)

#define MPICHECK(cmd)                                                         \
  do {                                                                        \
    int e_ = (cmd);                                                           \
    if (e_ != MPI_SUCCESS) {                                                  \
      char s_[MPI_MAX_ERROR_STRING];                                          \
      int n_ = 0;                                                             \
      MPI_Error_string(e_, s_, &n_);                                          \
      std::fprintf(stderr, "%s:%d MPI error: %.*s\n", __FILE__, __LINE__, n_, \
                   s_);                                                       \
      MPI_Abort(MPI_COMM_WORLD, e_);                                          \
    }                                                                         \
  } while (0)

using PortChan = mscclpp::PortChannelDeviceHandle;

struct DeviceChannels {
  PortChan send[kMaxChannels];
  PortChan recv[kMaxChannels];
};

__device__ __forceinline__ void addInPlace(float* dst, float const* src,
                                           size_t count) {
  int tid = threadIdx.x;
  int nth = blockDim.x;
  if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) &
       15) != 0) {
    for (size_t i = tid; i < count; i += nth) dst[i] += src[i];
    return;
  }
  size_t n4 = count / 4;
  float4* dst4 = reinterpret_cast<float4*>(dst);
  float4 const* src4 = reinterpret_cast<float4 const*>(src);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = dst4[i];
    float4 y = src4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    dst4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < count; i += nth) dst[i] += src[i];
}

__device__ __forceinline__ void copyToMappedHost(float* dst, float const* src,
                                                 size_t count) {
  int tid = threadIdx.x;
  int nth = blockDim.x;
  if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) &
       15) == 0) {
    size_t n4 = count / 4;
    float4* dst4 = reinterpret_cast<float4*>(dst);
    float4 const* src4 = reinterpret_cast<float4 const*>(src);
    for (size_t i = tid; i < n4; i += nth) dst4[i] = src4[i];
    for (size_t i = n4 * 4 + tid; i < count; i += nth) dst[i] = src[i];
  } else {
    for (size_t i = tid; i < count; i += nth) dst[i] = src[i];
  }
}

// One block owns one TCCL channel.  Every ring step has a distinct inbox,
// matching TCCL/NCCL's step-slot discipline and preventing a fast producer
// from reusing storage while the receiver is still reducing it.
__global__ void tcclRingAllReduce(DeviceChannels chans, float* data,
                                  float* scratch, size_t count,
                                  size_t scratchSlotBytes,
                                  size_t remoteScratchBase, float* hostStage,
                                  bool stageInterSend, int hostStageSlots,
                                  int ringPosition, int channels) {
  int channel = blockIdx.x;
  if (channel >= channels) return;

  size_t channelBegin = count * (size_t)channel / channels;
  size_t channelEnd = count * (size_t)(channel + 1) / channels;
  size_t channelCount = channelEnd - channelBegin;
  size_t chunk = (channelCount + kWorld - 1) / kWorld;
  size_t scratchSlotElems = scratchSlotBytes / sizeof(float);
  float* inbox = scratch + (size_t)channel * kWorld * scratchSlotElems;

  auto chunkInfo = [&](int index, size_t& offset, size_t& length) {
    offset = channelBegin + (size_t)index * chunk;
    length = offset >= channelEnd ? 0 : min(chunk, channelEnd - offset);
  };

  // ReduceScatter.  Rank p finishes with chunk p+1, matching NCCL/TCCL ring.
  for (int step = 0; step < kWorld - 1; step++) {
    int sendChunk = (ringPosition - step + kWorld) % kWorld;
    int recvChunk = (sendChunk - 1 + kWorld) % kWorld;
    size_t sendOffset, sendLength, recvOffset, recvLength;
    chunkInfo(sendChunk, sendOffset, sendLength);
    chunkInfo(recvChunk, recvOffset, recvLength);
    size_t slotOffset = (size_t)step * scratchSlotElems;
    size_t remoteSlotBytes =
        remoteScratchBase +
        ((size_t)channel * kWorld + step) * scratchSlotBytes;

    size_t hostSlot =
        ((size_t)channel * hostStageSlots + step % hostStageSlots) *
        scratchSlotElems;
    if (stageInterSend && step >= hostStageSlots) {
      if (threadIdx.x == 0) chans.send[channel].flush(-1);
      __syncthreads();
    }
    if (stageInterSend) {
      copyToMappedHost(hostStage + hostSlot, data + sendOffset, sendLength);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      if (stageInterSend)
        chans.send[channel].putWithSignal(remoteSlotBytes,
                                          hostSlot * sizeof(float),
                                          sendLength * sizeof(float));
      else
        chans.send[channel].putWithSignal(remoteSlotBytes,
                                          sendOffset * sizeof(float),
                                          sendLength * sizeof(float));
      chans.recv[channel].wait(-1);
    }
    __syncthreads();
    if (recvLength)
      addInPlace(data + recvOffset, inbox + slotOffset, recvLength);
    __syncthreads();
  }

  // Drain outstanding reads before the same data locations become AG targets.
  if (threadIdx.x == 0) chans.send[channel].flush(-1);
  __syncthreads();

  // AllGather forwards the fully reduced chunk received in the preceding step.
  for (int step = 0; step < kWorld - 1; step++) {
    int sendChunk = (ringPosition + 1 - step + kWorld) % kWorld;
    size_t offset, length;
    chunkInfo(sendChunk, offset, length);
    size_t hostSlot =
        ((size_t)channel * hostStageSlots + step % hostStageSlots) *
        scratchSlotElems;
    if (stageInterSend && step >= hostStageSlots) {
      if (threadIdx.x == 0) chans.send[channel].flush(-1);
      __syncthreads();
    }
    if (stageInterSend) {
      copyToMappedHost(hostStage + hostSlot, data + offset, length);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      if (stageInterSend)
        chans.send[channel].putWithSignal(offset * sizeof(float),
                                          hostSlot * sizeof(float),
                                          length * sizeof(float));
      else
        chans.send[channel].putWithSignal(offset * sizeof(float),
                                          offset * sizeof(float),
                                          length * sizeof(float));
      chans.recv[channel].wait(-1);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) chans.send[channel].flush(-1);
}

struct Config {
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  int warmup = 5;
  int iters = 10;
  int channels = 4;
  int threads = 512;
  bool hostStageInter = true;
  int hostStageSlots = 4;
  bool check = true;
};

static mscclpp::Transport ibTransport(int index) {
  return static_cast<mscclpp::Transport>(
      static_cast<int>(mscclpp::Transport::IB0) + index);
}

// Global ranks are assigned by mpirun as N0.G0..G3,N1.G0..G3.
static constexpr std::array<int, kWorld> kRing{3, 2, 0, 1, 4, 5, 7, 6};

static int ringPosition(int rank) {
  for (int i = 0; i < kWorld; i++)
    if (kRing[i] == rank) return i;
  return -1;
}

static int edgeRail(int src, int dst) {
  if (src == 1 && dst == 4) return 0;
  if (src == 6 && dst == 3) return 1;
  return -1;
}

static mscclpp::Transport edgeTransport(int src, int dst) {
  int rail = edgeRail(src, dst);
  return rail < 0 ? mscclpp::Transport::CudaIpc : ibTransport(rail);
}

static void bindToRailLocalCpuBeforeMpi() {
  char const* value = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int localRank = value ? std::atoi(value) : 0;
  int numa = localRank < 2 ? 0 : 1;
  int cpu = numa == 0 ? localRank * 2 : 1 + (localRank - 2) * 2;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  CPU_SET(cpu + 8, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

int main(int argc, char** argv) {
  bindToRailLocalCpuBeforeMpi();
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (world != kWorld) {
    if (rank == 0)
      std::fprintf(stderr, "requires exactly two nodes x four GPUs\n");
    MPI_Finalize();
    return 2;
  }

  Config cfg;
  try {
    for (int i = 1; i < argc; i++) {
      std::string key = argv[i];
      auto next = [&]() -> char const* {
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        return argv[i];
      };
      if (key == "-b")
        cfg.minBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-e")
        cfg.maxBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-w")
        cfg.warmup = std::atoi(next());
      else if (key == "-n")
        cfg.iters = std::atoi(next());
      else if (key == "-C")
        cfg.channels = std::atoi(next());
      else if (key == "-T")
        cfg.threads = std::atoi(next());
      else if (key == "-H")
        cfg.hostStageInter = std::atoi(next()) != 0;
      else if (key == "-S")
        cfg.hostStageSlots = std::atoi(next());
      else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else
        throw std::runtime_error("unknown argument " + key);
    }
    if (cfg.channels < 1 || cfg.channels > kMaxChannels)
      throw std::runtime_error("channels must be in [1,8]");
    if (cfg.hostStageSlots < 1 || cfg.hostStageSlots > kWorld)
      throw std::runtime_error("host stage slots must be in [1,8]");
  } catch (std::exception const& e) {
    if (rank == 0) std::fprintf(stderr, "%s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0, localSize = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_size(localComm, &localSize));
  if (localSize != kLocalRanks) MPI_Abort(MPI_COMM_WORLD, 2);
  CUDACHECK(cudaSetDevice(localRank));

  int position = ringPosition(rank);
  int prev = kRing[(position + kWorld - 1) % kWorld];
  int next = kRing[(position + 1) % kWorld];
  mscclpp::Transport sendTransport = edgeTransport(rank, next);
  mscclpp::Transport recvTransport = edgeTransport(prev, rank);

  for (mscclpp::Transport transport : {sendTransport, recvTransport}) {
    int value = static_cast<int>(transport);
    int ib0 = static_cast<int>(mscclpp::Transport::IB0);
    if (value >= ib0 && value < ib0 + 8 &&
        value - ib0 >= mscclpp::getIBDeviceCount()) {
      std::fprintf(stderr,
                   "rank %d: requested IB rail %d but MSCCL++ sees %d\n", rank,
                   value - ib0, mscclpp::getIBDeviceCount());
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id{};
  if (rank == 0) id = mscclpp::TcpBootstrap::createUniqueId();
  MPICHECK(MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD));
  bootstrap->initialize(id, 120);
  mscclpp::Communicator comm(bootstrap);

  // Data plus two inboxes per channel.  All ranks use an identical layout so
  // a sender can address its successor without exchanging offsets.
  size_t dataRegionBytes = (cfg.maxBytes + 255) & ~(size_t)255;
  size_t scratchSlotBytes =
      (dataRegionBytes + cfg.channels * kWorld - 1) / (cfg.channels * kWorld);
  scratchSlotBytes = (scratchSlotBytes + 255) & ~(size_t)255;
  size_t scratchBytes = (size_t)cfg.channels * kWorld * scratchSlotBytes;
  size_t bufferBytes = dataRegionBytes + scratchBytes;
  if (bufferBytes >= (1ULL << 32)) {
    if (rank == 0) std::fprintf(stderr, "PortChannel offsets exceed 32 bits\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  char* buffer = nullptr;
  CUDACHECK(cudaMalloc(&buffer, bufferBytes));
  CUDACHECK(cudaMemset(buffer, 0, bufferBytes));

  mscclpp::TransportFlags transports = mscclpp::Transport::CudaIpc;
  if (edgeRail(rank, next) >= 0) transports |= sendTransport;
  if (edgeRail(prev, rank) >= 0) transports |= recvTransport;
  auto localMemory = comm.registerMemory(buffer, bufferBytes, transports);

  bool stageInterSend = cfg.hostStageInter && edgeRail(rank, next) >= 0;
  float* hostStageCpu = nullptr;
  float* hostStageGpu = nullptr;
  size_t hostStageBytes =
      (size_t)cfg.channels * cfg.hostStageSlots * scratchSlotBytes;
  mscclpp::RegisteredMemory hostStageMemory;
  if (stageInterSend) {
    CUDACHECK(cudaHostAlloc(&hostStageCpu, hostStageBytes,
                            cudaHostAllocMapped | cudaHostAllocPortable));
    CUDACHECK(cudaHostGetDevicePointer(&hostStageGpu, hostStageCpu, 0));
    hostStageMemory =
        comm.registerMemory(hostStageCpu, hostStageBytes, sendTransport);
  }

  std::vector<std::shared_future<mscclpp::Connection>> sendConnF(cfg.channels),
      recvConnF(cfg.channels);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> nextMemF(
      cfg.channels),
      prevMemF(cfg.channels);
  for (int c = 0; c < cfg.channels; c++) {
    int connTag = 1000 + c;
    int memTag = 2000 + c;
    sendConnF[c] = comm.connect(sendTransport, next, connTag);
    recvConnF[c] = comm.connect(recvTransport, prev, connTag);
    comm.sendMemory(localMemory, next, memTag);
    nextMemF[c] = comm.recvMemory(next, memTag);
    comm.sendMemory(localMemory, prev, memTag);
    prevMemF[c] = comm.recvMemory(prev, memTag);
  }

  std::vector<mscclpp::Connection> sendConnections, recvConnections;
  std::vector<mscclpp::RegisteredMemory> nextMemory, prevMemory;
  for (int c = 0; c < cfg.channels; c++) {
    sendConnections.push_back(sendConnF[c].get());
    recvConnections.push_back(recvConnF[c].get());
    nextMemory.push_back(nextMemF[c].get());
    prevMemory.push_back(prevMemF[c].get());
  }

  auto sendProxy = std::make_shared<mscclpp::ProxyService>();
  auto recvProxy = std::make_shared<mscclpp::ProxyService>();
  auto sendLocalId =
      sendProxy->addMemory(stageInterSend ? hostStageMemory : localMemory);
  auto recvLocalId = recvProxy->addMemory(localMemory);
  std::vector<std::shared_future<mscclpp::Semaphore>> sendSemF(cfg.channels),
      recvSemF(cfg.channels);
  for (int c = 0; c < cfg.channels; c++) {
    sendSemF[c] = comm.buildSemaphore(sendConnections[c], next, 3000 + c);
    recvSemF[c] = comm.buildSemaphore(recvConnections[c], prev, 3000 + c);
  }
  std::vector<mscclpp::PortChannel> sendObjects, recvObjects;
  sendObjects.reserve(cfg.channels);
  recvObjects.reserve(cfg.channels);
  DeviceChannels deviceChannels{};
  for (int c = 0; c < cfg.channels; c++) {
    auto sendSem = sendProxy->addSemaphore(sendSemF[c].get());
    auto recvSem = recvProxy->addSemaphore(recvSemF[c].get());
    auto nextId = sendProxy->addMemory(nextMemory[c]);
    auto prevId = recvProxy->addMemory(prevMemory[c]);
    sendObjects.emplace_back(
        sendProxy->portChannel(sendSem, nextId, sendLocalId));
    recvObjects.emplace_back(
        recvProxy->portChannel(recvSem, prevId, recvLocalId));
    deviceChannels.send[c] = sendObjects.back().deviceHandle();
    deviceChannels.recv[c] = recvObjects.back().deviceHandle();
  }
  sendProxy->startProxy();
  recvProxy->startProxy();

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::vector<float> hostIn, hostOut;
  auto fill = [&](size_t count) {
    hostIn.resize(count);
    for (size_t i = 0; i < count; i++)
      hostIn[i] = (float)(rank + 1) * (float)(1 + i % 97);
    CUDACHECK(cudaMemcpyAsync(buffer, hostIn.data(), count * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  };
  auto launch = [&](size_t count) {
    tcclRingAllReduce<<<cfg.channels, cfg.threads, 0, stream>>>(
        deviceChannels, reinterpret_cast<float*>(buffer),
        reinterpret_cast<float*>(buffer + dataRegionBytes), count,
        scratchSlotBytes, dataRegionBytes, hostStageGpu, stageInterSend,
        cfg.hostStageSlots, position, cfg.channels);
  };
  auto verify = [&](size_t count) {
    hostOut.resize(count);
    CUDACHECK(cudaMemcpy(hostOut.data(), buffer, count * sizeof(float),
                         cudaMemcpyDeviceToHost));
    constexpr double rankSum = (double)kWorld * (kWorld + 1) / 2.0;
    double worst = 0.0;
    for (size_t i = 0; i < count; i++) {
      double want = rankSum * (double)(1 + i % 97);
      worst = std::max(worst, std::fabs((double)hostOut[i] - want) / want);
    }
    return worst;
  };

  if (rank == 0) {
    std::printf(
        "# MSCCL++ implementation of TCCL global-ring AllReduce, float32\n");
    std::printf("# ring N0:3->2->0->1 -> N1:0->1->3->2 -> N0:3; channels %d\n",
                cfg.channels);
    std::printf(
        "# inter-node send path: %s\n",
        cfg.hostStageInter
            ? "GPU SM -> mapped host -> IB -> remote GPU (CPU_GPU_INTER)"
            : "GPU memory -> GDR -> remote GPU (GPU_GPU_INTER)");
    if (cfg.hostStageInter)
      std::printf("# host staging step slots: %d\n", cfg.hostStageSlots);
    std::printf("# %12s %12s %10s %10s %10s %10s\n", "size(B)", "count",
                "time(us)", "algbw", "busbw", "maxrelerr");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes = (bytes + 511) & ~(size_t)511;
    size_t count = bytes / sizeof(float);
    double error = -1.0;
    if (cfg.check) {
      fill(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      launch(count);
      CUDACHECK(cudaStreamSynchronize(stream));
      error = verify(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
    fill(count);
    for (int i = 0; i < cfg.warmup; i++) launch(count);
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; i++) launch(count);
    CUDACHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - begin).count() /
                cfg.iters;
    double worstUs = 0.0, worstError = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&error, &worstError, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    if (rank == 0) {
      double algbw = (double)bytes / (worstUs * 1e-6) / 1e9;
      std::printf("  %12zu %12zu %10.2f %10.2f %10.2f %10.1e\n", bytes, count,
                  worstUs, algbw, algbw * 2.0 * (kWorld - 1) / kWorld,
                  worstError);
      std::fflush(stdout);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  recvProxy->stopProxy();
  sendProxy->stopProxy();
  CUDACHECK(cudaStreamDestroy(stream));
  if (hostStageCpu) CUDACHECK(cudaFreeHost(hostStageCpu));
  CUDACHECK(cudaFree(buffer));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
