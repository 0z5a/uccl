// Two-node, four-GPU-per-node hierarchical AllReduce for PCIe-only GPUs.
//
// Schedule (H1):
//   1. topology-aware four-GPU ring ReduceScatter inside each node;
//   2. four parallel, rail-local IB exchanges between equal local ranks;
//   3. reduce the remote shard and ring AllGather inside each node.
//
// H2 pipelines the three stages by splitting each rank shard into pieces. While
// the IB channel moves piece p, the local copy engines work on other pieces. It
// is intended for medium/large messages; H1 avoids the extra triggers at small
// sizes.  Both algorithms move the minimum N bytes in each direction between
// two nodes, spread over the two HCAs without routing GPU traffic over UPI.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>

constexpr int kNodes = 2;
constexpr int kLocalRanks = 4;
constexpr int kWorld = kNodes * kLocalRanks;

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
__device__ mscclpp::DeviceSyncer gMultiNodeSyncer;

struct LocalAgChans {
  PortChan c[kLocalRanks];
};

struct Layout {
  size_t data;
  size_t localScratch;
  size_t ibSend;
  size_t ibScratch;
};

__device__ __forceinline__ void addOut(float* dst, float const* a,
                                       float const* b, size_t n, int tid,
                                       int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* a4 = reinterpret_cast<float4 const*>(a);
  float4 const* b4 = reinterpret_cast<float4 const*>(b);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = a4[i], y = b4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] = a[i] + b[i];
}

__device__ __forceinline__ void addInto(float* dst, float const* src, size_t n,
                                        int tid, int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* s4 = reinterpret_cast<float4 const*>(src);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = d4[i], y = s4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] += src[i];
}

// DeviceSyncer provides the device release/acquire.  The proxy FIFO push is a
// system-scope release, which makes the preceding GPU stores visible to DMA.
__device__ __forceinline__ void publish() { gMultiNodeSyncer.sync(gridDim.x); }

// Local ring order is 0 -> 1 -> 3 -> 2 -> 0.  It crosses the NUMA boundary once
// in each direction, so the UPI is never oversubscribed by two same-direction
// flows.  pos is the rank's position in this ring.
__device__ void localReduceScatterPiece(
    PortChan send, PortChan recv, float* data, float* scratch, float* ibSend,
    size_t nelem, int pos, size_t pieceOff, size_t pieceLen, size_t chunkElems,
    size_t scratchByteOffset, int tid, int nth) {
  const size_t chunkBytes = chunkElems * sizeof(float);
  for (int step = 0; step < kLocalRanks - 1; step++) {
    int cSend = (pos - step + kLocalRanks) % kLocalRanks;
    int cRecv = (cSend - 1 + kLocalRanks) % kLocalRanks;
    size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
    size_t recvOff = (size_t)cRecv * chunkElems + pieceOff;
    size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);
    size_t recvLen = recvOff >= nelem ? 0 : min(pieceLen, nelem - recvOff);

    publish();
    if (tid == 0)
      send.putWithSignal(scratchByteOffset + (size_t)step * chunkBytes +
                             pieceOff * sizeof(float),
                         sendOff * sizeof(float), sendLen * sizeof(float));
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
    if (recvLen) {
      if (step == kLocalRanks - 2)
        addOut(ibSend + pieceOff, data + recvOff,
               scratch + (size_t)step * chunkElems + pieceOff, recvLen, tid,
               nth);
      else
        addInto(data + recvOff, scratch + (size_t)step * chunkElems + pieceOff,
                recvLen, tid, nth);
    }
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

__device__ void localAllGatherPiece(PortChan send, PortChan recv, float* data,
                                    size_t nelem, int pos, size_t pieceOff,
                                    size_t pieceLen, size_t chunkElems,
                                    int tid) {
  for (int step = 0; step < kLocalRanks - 1; step++) {
    int cSend = (pos + 1 - step + kLocalRanks) % kLocalRanks;
    size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
    size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);

    // Only the first AllGather hop sends bytes produced by SM stores. Later
    // hops forward bytes that arrived through the same CUDA-IPC proxy and are
    // already ordered by wait(); the end-of-step sync aligns every block.
    if (step == 0) publish();
    if (tid == 0)
      send.putWithSignal(sendOff * sizeof(float), sendOff * sizeof(float),
                         sendLen * sizeof(float));
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

__device__ void localFlatAllGatherPiece(LocalAgChans chans, float* data,
                                        size_t ownBase, size_t pieceOff,
                                        size_t pieceLen, int localRank,
                                        int tid) {
  publish();
  if (tid == 0) {
    size_t byteOff = (ownBase + pieceOff) * sizeof(float);
    size_t bytes = pieceLen * sizeof(float);
    for (int k = 1; k < kLocalRanks; k++) {
      int peer = (localRank + k) % kLocalRanks;
      chans.c[peer].putWithSignal(byteOff, byteOff, bytes);
    }
    for (int k = 1; k < kLocalRanks; k++) {
      int peer = (localRank - k + kLocalRanks) % kLocalRanks;
      chans.c[peer].wait();
    }
  }
  gMultiNodeSyncer.sync(gridDim.x);
}

__global__ void __launch_bounds__(1024)
    hierarchicalAllReduce(PortChan localSend, PortChan localRecv,
                          LocalAgChans localAg, PortChan ib, float* base,
                          size_t nelem, Layout layout, int localRank,
                          int ringPos, int pieces, int pipelineDistance,
                          bool flatAg) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* localScratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                                 layout.localScratch);
  float* ibSend =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  float* ibScratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                              layout.ibScratch);

  size_t chunkElems =
      ((nelem + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
  size_t ownChunk = (size_t)((ringPos + 1) % kLocalRanks);
  size_t ownBase = ownChunk * chunkElems;
  size_t ownLen = ownBase >= nelem ? 0 : min(chunkElems, nelem - ownBase);
  size_t pieceElems = ((ownLen + pieces - 1) / pieces + 31) & ~(size_t)31;

  // A bounded software pipeline.  The IB write for piece p remains in flight
  // while local RS works on later pieces.  Once the chosen distance is reached,
  // reduce and locally publish the oldest completed piece.
  for (int p = 0; p < pieces + pipelineDistance; p++) {
    if (p < pieces) {
      size_t off = (size_t)p * pieceElems;
      size_t len = off >= ownLen ? 0 : min(pieceElems, ownLen - off);
      localReduceScatterPiece(localSend, localRecv, data, localScratch, ibSend,
                              nelem, ringPos, off, len, chunkElems,
                              layout.localScratch, tid, nth);
      // The final local RS step writes directly into the immutable IB slot, so
      // no extra staging copy is needed and inbound reduction cannot race the
      // outbound RDMA read.
      publish();
      if (tid == 0)
        ib.putWithSignal(layout.ibScratch + off * sizeof(float),
                         layout.ibSend + off * sizeof(float),
                         len * sizeof(float));
    }

    int done = p - pipelineDistance;
    if (done >= 0 && done < pieces) {
      size_t off = (size_t)done * pieceElems;
      size_t len = off >= ownLen ? 0 : min(pieceElems, ownLen - off);
      if (tid == 0) ib.wait();
      gMultiNodeSyncer.sync(gridDim.x);
      if (len)
        addOut(data + ownBase + off, ibSend + off, ibScratch + off, len, tid,
               nth);
      gMultiNodeSyncer.sync(gridDim.x);
      if (flatAg)
        localFlatAllGatherPiece(localAg, data, ownBase, off, len, localRank,
                                tid);
      else
        localAllGatherPiece(localSend, localRecv, data, nelem, ringPos, off,
                            len, chunkElems, tid);
    }
  }
  if (tid == 0) {
    // CUDA-IPC copies do not consume an IB completion queue and need no flush.
    // Drain the IB QP once per collective to bound its CQ and protect reuse of
    // the source buffer by the next launch.
    ib.flush();
  }
}

__global__ void nullKernel() {}

struct Config {
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  int warmup = 10;
  int iters = 20;
  int blocks = 24;
  int threads = 512;
  size_t pipelineMinBytes = 4UL << 20;
  size_t pieceBytes = 2UL << 20;
  int pipelineDistance = 2;
  bool flatAg = false;
  bool check = true;
  std::array<int, kLocalRanks> railMap{0, 0, 1, 1};
};

static std::array<int, kLocalRanks> parseRailMap(std::string const& text) {
  std::array<int, kLocalRanks> out{};
  std::stringstream ss(text);
  std::string item;
  for (int i = 0; i < kLocalRanks; i++) {
    if (!std::getline(ss, item, ','))
      throw std::runtime_error("rail map needs four entries");
    out[i] = std::stoi(item);
  }
  if (std::getline(ss, item, ','))
    throw std::runtime_error("rail map needs four entries");
  return out;
}

static mscclpp::Transport ibTransport(int index) {
  if (index < 0 || index >= 8)
    throw std::runtime_error("IB rail index must be in [0,7]");
  return static_cast<mscclpp::Transport>(
      static_cast<int>(mscclpp::Transport::IB0) + index);
}

static int ringPosition(int localRank) {
  constexpr int pos[kLocalRanks] = {0, 1, 3, 2};
  return pos[localRank];
}

static int ringSuccessor(int localRank) {
  constexpr int succ[kLocalRanks] = {1, 3, 0, 2};
  return succ[localRank];
}

static int ringPredecessor(int localRank) {
  constexpr int pred[kLocalRanks] = {2, 0, 3, 1};
  return pred[localRank];
}

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (world != kWorld) {
    if (rank == 0)
      std::fprintf(stderr,
                   "requires exactly %d MPI ranks (two nodes x four GPUs)\n",
                   kWorld);
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
      else if (key == "-B")
        cfg.blocks = std::atoi(next());
      else if (key == "-T")
        cfg.threads = std::atoi(next());
      else if (key == "-p")
        cfg.pipelineMinBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-k")
        cfg.pieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-d")
        cfg.pipelineDistance = std::atoi(next());
      else if (key == "-a")
        cfg.flatAg = std::atoi(next()) != 0;
      else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else if (key == "--rail-map")
        cfg.railMap = parseRailMap(next());
      else
        throw std::runtime_error("unknown argument " + key);
    }
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
  if (localSize != kLocalRanks) {
    std::fprintf(stderr, "rank %d: expected four ranks per node, got %d\n",
                 rank, localSize);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  CUDACHECK(cudaSetDevice(localRank));

  int rail = cfg.railMap[localRank];
  int nIb = mscclpp::getIBDeviceCount();
  if (rail >= nIb) {
    std::fprintf(stderr,
                 "rank %d: rail %d requested but MSCCL++ sees %d IB devices; "
                 "build with MSCCLPP_USE_IB=ON\n",
                 rank, rail, nIb);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id{};
  if (rank == 0) id = mscclpp::TcpBootstrap::createUniqueId();
  MPICHECK(MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD));
  bootstrap->initialize(id, 120);
  mscclpp::Communicator comm(bootstrap);

  size_t dataBytes = cfg.maxBytes;
  size_t localScratchBytes = dataBytes;
  size_t ibSlotBytes = (dataBytes + kLocalRanks - 1) / kLocalRanks + 4096;
  Layout layout{0, dataBytes, dataBytes + localScratchBytes,
                dataBytes + localScratchBytes + ibSlotBytes};
  size_t bufferBytes = layout.ibScratch + ibSlotBytes;
  if (bufferBytes >= (1ULL << 32)) {
    if (rank == 0)
      std::fprintf(
          stderr,
          "registered region exceeds PortChannel's 32-bit offset limit\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  char* buffer = nullptr;
  CUDACHECK(cudaMalloc(&buffer, bufferBytes));
  CUDACHECK(cudaMemset(buffer, 0, bufferBytes));

  mscclpp::TransportFlags transports = mscclpp::Transport::CudaIpc;
  transports |= ibTransport(rail);
  auto localMemory = comm.registerMemory(buffer, bufferBytes, transports);

  int nodeBase = rank - localRank;
  int succ = nodeBase + ringSuccessor(localRank);
  int pred = nodeBase + ringPredecessor(localRank);
  int diag = nodeBase + (localRank ^ 3);
  int cross = rank ^ kLocalRanks;
  std::array<int, 4> peers{succ, pred, diag, cross};
  std::array<mscclpp::Transport, 4> peerTransports{
      mscclpp::Transport::CudaIpc, mscclpp::Transport::CudaIpc,
      mscclpp::Transport::CudaIpc, ibTransport(rail)};
  std::array<std::shared_future<mscclpp::Connection>, 4> connectionFutures;
  std::array<std::shared_future<mscclpp::RegisteredMemory>, 4> memoryFutures;
  for (int i = 0; i < 4; i++) {
    connectionFutures[i] =
        comm.connect(peerTransports[i], peers[i], i < 3 ? 100 : 102);
    comm.sendMemory(localMemory, peers[i], i < 3 ? 200 : 202);
    memoryFutures[i] = comm.recvMemory(peers[i], i < 3 ? 200 : 202);
  }

  std::array<mscclpp::Connection, 4> connections;
  std::array<mscclpp::RegisteredMemory, 4> remoteMemory;
  for (int i = 0; i < 4; i++) {
    connections[i] = connectionFutures[i].get();
    remoteMemory[i] = memoryFutures[i].get();
  }

  // Local DMA and IB must have independent proxy threads/FIFOs; sharing one
  // ProxyService serializes trigger submission and defeats the software
  // pipeline even though the PCIe copy engines and NIC can run concurrently.
  auto localProxy = std::make_shared<mscclpp::ProxyService>();
  auto ibProxy = std::make_shared<mscclpp::ProxyService>();
  std::array<std::shared_future<mscclpp::Semaphore>, 4> semaphoreFutures;
  for (int i = 0; i < 4; i++)
    semaphoreFutures[i] =
        comm.buildSemaphore(connections[i], peers[i], i < 3 ? 300 : 302);
  auto localSendSemaphore = localProxy->addSemaphore(semaphoreFutures[0].get());
  auto localRecvSemaphore = localProxy->addSemaphore(semaphoreFutures[1].get());
  auto localDiagSemaphore = localProxy->addSemaphore(semaphoreFutures[2].get());
  auto ibSemaphore = ibProxy->addSemaphore(semaphoreFutures[3].get());
  auto localLocalMemoryId = localProxy->addMemory(localMemory);
  auto localSendMemoryId = localProxy->addMemory(remoteMemory[0]);
  auto localRecvMemoryId = localProxy->addMemory(remoteMemory[1]);
  auto localDiagMemoryId = localProxy->addMemory(remoteMemory[2]);
  auto ibLocalMemoryId = ibProxy->addMemory(localMemory);
  auto ibRemoteMemoryId = ibProxy->addMemory(remoteMemory[3]);

  // Separate objects are retained because their device handles alias proxy
  // state.
  auto localSendObject = localProxy->portChannel(
      localSendSemaphore, localSendMemoryId, localLocalMemoryId);
  auto localRecvObject = localProxy->portChannel(
      localRecvSemaphore, localRecvMemoryId, localLocalMemoryId);
  auto localDiagObject = localProxy->portChannel(
      localDiagSemaphore, localDiagMemoryId, localLocalMemoryId);
  auto ibObject =
      ibProxy->portChannel(ibSemaphore, ibRemoteMemoryId, ibLocalMemoryId);
  PortChan localSend = localSendObject.deviceHandle();
  PortChan localRecv = localRecvObject.deviceHandle();
  LocalAgChans localAg{};
  localAg.c[ringSuccessor(localRank)] = localSendObject.deviceHandle();
  localAg.c[ringPredecessor(localRank)] = localRecvObject.deviceHandle();
  localAg.c[localRank ^ 3] = localDiagObject.deviceHandle();
  PortChan ib = ibObject.deviceHandle();
  localProxy->startProxy();
  ibProxy->startProxy();

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
  auto verify = [&](size_t count) {
    hostOut.resize(count);
    CUDACHECK(cudaMemcpy(hostOut.data(), buffer, count * sizeof(float),
                         cudaMemcpyDeviceToHost));
    double worst = 0.0;
    size_t firstBad = count;
    constexpr double rankSum = (double)kWorld * (kWorld + 1) / 2.0;
    for (size_t i = 0; i < count; i++) {
      double want = rankSum * (double)(1 + i % 97);
      double error = std::fabs((double)hostOut[i] - want) / want;
      if (error > 1e-5 && firstBad == count) firstBad = i;
      worst = std::max(worst, error);
    }
    if (firstBad != count) {
      double want = rankSum * (double)(1 + firstBad % 97);
      std::fprintf(
          stderr,
          "rank %d first bad index %zu got %.1f want %.1f relerr %.3e\n", rank,
          firstBad, (double)hostOut[firstBad], want,
          std::fabs((double)hostOut[firstBad] - want) / want);
    }
    return worst;
  };
  auto launch = [&](size_t count, size_t bytes) {
    size_t shardBytes = (bytes + kLocalRanks - 1) / kLocalRanks;
    int pieces =
        bytes >= cfg.pipelineMinBytes
            ? (int)std::max<size_t>(
                  2, std::min<size_t>(32, (shardBytes + cfg.pieceBytes - 1) /
                                              cfg.pieceBytes))
            : 1;
    int distance =
        pieces == 1 ? 1 : std::max(1, std::min(cfg.pipelineDistance, pieces));
    hierarchicalAllReduce<<<cfg.blocks, cfg.threads, 0, stream>>>(
        localSend, localRecv, localAg, ib, reinterpret_cast<float*>(buffer),
        count, layout, localRank, ringPosition(localRank), pieces, distance,
        cfg.flatAg);
  };

  char host[MPI_MAX_PROCESSOR_NAME];
  int hostLen = 0;
  MPI_Get_processor_name(host, &hostLen);
  cudaDeviceProp prop{};
  CUDACHECK(cudaGetDeviceProperties(&prop, localRank));
  char line[512]{};
  std::snprintf(line, sizeof(line),
                "# rank %d node %.*s gpu %d %s rail %d (%s)\n", rank, hostLen,
                host, localRank, prop.name, rail,
                mscclpp::getIBDeviceName(ibTransport(rail)).c_str());
  std::vector<char> lines(rank == 0 ? world * sizeof(line) : 0);
  MPICHECK(MPI_Gather(line, sizeof(line), MPI_BYTE, lines.data(), sizeof(line),
                      MPI_BYTE, 0, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("# UCCL Lite two-node hierarchical AllReduce, float32\n");
    for (int r = 0; r < world; r++)
      std::fputs(lines.data() + r * sizeof(line), stdout);
    std::printf(
        "# local ring 0->1->3->2->0; AG %s; rail map %d,%d,%d,%d; piece %zu B; "
        "pipeline >= %zu B\n",
        cfg.flatAg ? "flat" : "ring", cfg.railMap[0], cfg.railMap[1],
        cfg.railMap[2], cfg.railMap[3], cfg.pieceBytes, cfg.pipelineMinBytes);
    std::printf("# %12s %12s %8s %10s %10s %10s %10s\n", "size(B)", "count",
                "pieces", "time(us)", "algbw", "busbw", "maxrelerr");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes =
        (bytes + 511) & ~(size_t)511;  // divisible into aligned four-way chunks
    size_t count = bytes / sizeof(float);
    size_t shardBytes = bytes / kLocalRanks;
    int pieces =
        bytes >= cfg.pipelineMinBytes
            ? (int)std::max<size_t>(
                  2, std::min<size_t>(32, (shardBytes + cfg.pieceBytes - 1) /
                                              cfg.pieceBytes))
            : 1;
    double relerr = -1.0;
    if (cfg.check) {
      fill(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      launch(count, bytes);
      CUDACHECK(cudaStreamSynchronize(stream));
      relerr = verify(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    fill(count);
    for (int i = 0; i < cfg.warmup; i++) launch(count, bytes);
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; i++) launch(count, bytes);
    CUDACHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - begin).count() /
                cfg.iters;
    double worstUs = 0.0, worstErr = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&relerr, &worstErr, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    if (rank == 0) {
      double algbw = (double)bytes / (worstUs * 1e-6) / 1e9;
      double busbw = algbw * 2.0 * (kWorld - 1) / kWorld;
      std::printf("  %12zu %12zu %8d %10.2f %10.2f %10.2f %10.1e\n", bytes,
                  count, pieces, worstUs, algbw, busbw, worstErr);
      std::fflush(stdout);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  ibProxy->stopProxy();
  localProxy->stopProxy();
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(buffer));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
