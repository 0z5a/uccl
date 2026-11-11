// Experimental two-node host-RDMA AllGather.
// One MPI process owns the two GPUs in each NUMA group. Each source group
// uploads two shards into NUMA-local pinned host memory and sends the same
// packet to both remote NUMA groups. Each destination group also receives the
// other local group's packet, then its two GPUs read the shared host packets.
// Three slots overlap upload(j), network(j-1), and multicast(j-2).
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr int kNodes = 2;
constexpr int kGroups = 2;
constexpr int kGpusPerGroup = 2;
constexpr int kRanks = kNodes * kGroups;
constexpr int kGpus = kRanks * kGpusPerGroup;
constexpr int kSlots = 3;
constexpr int kTagRemote = 510;
constexpr int kTagLocal = 511;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void check(int result) {
  if (result != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(result, message, &length);
    throw std::runtime_error(std::string(message, length));
  }
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
struct Config {
  size_t begin = 1UL << 10;
  size_t end = 1UL << 30;
  size_t chunkBytes = 1UL << 20;
  int warmups = 20;
  int iterations = 10;
  int repeats = 3;
  bool dmaMulticast = false;
  bool compareNccl = false;
};
Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (option == "--help") {
      std::puts("Usage: uccl_lite_multinode_host_allgather "
                "[-b BYTES] [-e BYTES] [-k CHUNK_BYTES] "
                "[-w WARMUPS] [-n ITERATIONS] [-r REPEATS] "
                "[--download sm|dma] [--nccl 0|1]");
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    char const* value = argv[++i];
    if (option == "-b") cfg.begin = std::strtoull(value, nullptr, 0);
    else if (option == "-e") cfg.end = std::strtoull(value, nullptr, 0);
    else if (option == "-k") cfg.chunkBytes = std::strtoull(value, nullptr, 0);
    else if (option == "-w") cfg.warmups = std::atoi(value);
    else if (option == "-n") cfg.iterations = std::atoi(value);
    else if (option == "-r") cfg.repeats = std::atoi(value);
    else if (option == "--download") {
      if (std::strcmp(value, "sm") == 0) cfg.dmaMulticast = false;
      else if (std::strcmp(value, "dma") == 0) cfg.dmaMulticast = true;
      else throw std::invalid_argument("invalid download method");
    }
    else if (option == "--nccl") cfg.compareNccl = std::atoi(value) != 0;
    else throw std::invalid_argument("unknown option " + option);
  }
  if (cfg.begin < 1024 || cfg.begin > cfg.end || cfg.end > (1UL << 30) ||
      cfg.begin % 128 || cfg.end % 128 || cfg.chunkBytes < 128 ||
      cfg.chunkBytes % 128 || cfg.chunkBytes > (16UL << 20) ||
      cfg.warmups < 5 || cfg.iterations < 10 || cfg.repeats < 1)
    throw std::invalid_argument("invalid size or iteration settings");
  return cfg;
}
void bindGroupBeforeMpi() {
  char const* text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int group = text ? std::atoi(text) : 0;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  for (int cpu = group; cpu < 48; cpu += 2) CPU_SET(cpu, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    throw std::runtime_error("CPU affinity failed");
  if (!std::getenv("UCX_NET_DEVICES"))
    setenv("UCX_NET_DEVICES", group == 0 ? "mlx5_0:1" : "mlx5_1:1", 1);
  if (!std::getenv("UCX_TLS"))
    setenv("UCX_TLS", "rc_mlx5,ud_mlx5,sm,self", 1);
}
struct HostBlock {
  char* host = nullptr;
  size_t bytes = 0;
  std::array<char*, kGpusPerGroup> mapped{};
  void initialize(size_t capacity, int node,
                  std::array<int, kGpusPerGroup> const& devices) {
    bytes = capacity;
    void* address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) throw std::runtime_error("host mmap failed");
    unsigned long mask = 1UL << node;
    if (syscall(SYS_mbind, address, bytes, MPOL_BIND, &mask,
                8 * sizeof(mask), 0)) {
      munmap(address, bytes);
      throw std::runtime_error("host mbind failed");
    }
    std::memset(address, 0, bytes);
    check(cudaSetDevice(devices[0]));
    auto result = cudaHostRegister(address, bytes,
                                   cudaHostRegisterPortable |
                                       cudaHostRegisterMapped);
    if (result != cudaSuccess) {
      munmap(address, bytes);
      check(result);
    }
    host = static_cast<char*>(address);
    for (int i = 0; i < kGpusPerGroup; ++i) {
      check(cudaSetDevice(devices[i]));
      check(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&mapped[i]), host, 0));
    }
  }
  ~HostBlock() {
    if (host) {
      cudaHostUnregister(host);
      munmap(host, bytes);
    }
  }
  HostBlock() = default;
  HostBlock(HostBlock const&) = delete;
  HostBlock& operator=(HostBlock const&) = delete;
};
struct Slot {
  HostBlock send;
  HostBlock local;
  std::array<HostBlock, kGroups> remote;
  std::array<MPI_Request, 3> receive{MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                     MPI_REQUEST_NULL};
  std::array<MPI_Request, 3> outgoing{MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                      MPI_REQUEST_NULL};
  int piece = -1;
  size_t length = 0;
  bool active = false;
  bool sent = false;
  bool multicast = false;
  void initialize(size_t capacity, int node,
                  std::array<int, kGpusPerGroup> const& devices) {
    send.initialize(capacity, node, devices);
    local.initialize(capacity, node, devices);
    for (auto& block : remote) block.initialize(capacity, node, devices);
  }
};
struct Device {
  int physical;
  int globalRank;
  float *input = nullptr, *output = nullptr;
  unsigned int* wrong = nullptr;
  cudaStream_t upload = nullptr, consume = nullptr;
  std::array<cudaEvent_t, kSlots> uploadDone{};
  std::array<cudaEvent_t, kSlots> consumeDone{};
  Device(int device, int global, size_t maxShardBytes, size_t maxOutputBytes)
      : physical(device), globalRank(global) {
    check(cudaSetDevice(physical));
    check(cudaMalloc(&input, maxShardBytes));
    check(cudaMalloc(&output, maxOutputBytes));
    check(cudaMalloc(&wrong, sizeof(unsigned int)));
    check(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&consume, cudaStreamNonBlocking));
    for (int slot = 0; slot < kSlots; ++slot) {
      check(cudaEventCreateWithFlags(&uploadDone[slot], cudaEventDisableTiming));
      check(cudaEventCreateWithFlags(&consumeDone[slot], cudaEventDisableTiming));
    }
  }
  ~Device() {
    cudaSetDevice(physical);
    for (int slot = 0; slot < kSlots; ++slot) {
      if (uploadDone[slot]) cudaEventDestroy(uploadDone[slot]);
      if (consumeDone[slot]) cudaEventDestroy(consumeDone[slot]);
    }
    if (upload) cudaStreamDestroy(upload);
    if (consume) cudaStreamDestroy(consume);
    if (wrong) cudaFree(wrong);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
  }
};
__global__ void fillInput(float* input, size_t count, int rank, int generation) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  float factor = float(rank + 1 + generation * kGpus);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride)
    input[i] = factor * float(1 + i % 97);
}
struct Sources {
  uint4 const* pointer[kGpus];
};
__global__ void multicastPacket(Sources sources, uint4* output,
                                 size_t shardVectors, size_t pieceOffset,
                                 size_t pieceVectors) {
  int source = blockIdx.y;
  auto const* input = sources.pointer[source];
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pieceVectors; i += size_t(gridDim.x) * blockDim.x)
    output[size_t(source) * shardVectors + pieceOffset + i] = input[i];
}
__global__ void checkOutput(float const* output, unsigned int* wrong,
                            size_t count, size_t shard, int generation) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    int source = int(i / shard);
    size_t index = i % shard;
    float expected = float(source + 1 + generation * kGpus) *
                     float(1 + index % 97);
    if (!(fabsf(output[i] - expected) <= 1e-5f)) atomicAdd(wrong, 1);
  }
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  bool mpiReady = false;
  try {
    Config cfg = parse(argc, argv);
    bindGroupBeforeMpi();
    check(MPI_Init(&argc, &argv));
    mpiReady = true;
    int rank = -1, world = 0;
    check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    check(MPI_Comm_size(MPI_COMM_WORLD, &world));
    MPI_Comm localComm;
    check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                              MPI_INFO_NULL, &localComm));
    int localRank = -1, localSize = 0;
    check(MPI_Comm_rank(localComm, &localRank));
    check(MPI_Comm_size(localComm, &localSize));
    if (world != kRanks || localSize != kGroups ||
        localRank != rank % kGroups)
      throw std::runtime_error("requires two nodes, two NUMA groups per node");
    int node = rank / kGroups;
    int remoteBase = (1 - node) * kGroups;
    int localPeer = rank ^ 1;
    std::array<int, kGpusPerGroup> physical{localRank * 2, localRank * 2 + 1};
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < 4) throw std::runtime_error("requires four local GPUs");
    std::array<std::unique_ptr<Device>, kGpusPerGroup> devices;
    for (int gpu = 0; gpu < kGpusPerGroup; ++gpu)
      devices[gpu] = std::make_unique<Device>(
          physical[gpu], node * 4 + localRank * 2 + gpu,
          cfg.end / kGpus, cfg.end);
    size_t capacity = std::min(cfg.chunkBytes, cfg.end / kGpus);
    std::array<std::unique_ptr<Slot>, kSlots> slots;
    for (auto& slot : slots) {
      slot = std::make_unique<Slot>();
      slot->initialize(2 * capacity, localRank, physical);
    }
    std::array<ncclComm_t, kGpusPerGroup> ncclComms{};
    int ncclVersion = 0;
    if (cfg.compareNccl) {
      ncclUniqueId id{};
      if (rank == 0) check(ncclGetUniqueId(&id));
      check(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
      check(ncclGroupStart());
      for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
        check(cudaSetDevice(physical[gpu]));
        check(ncclCommInitRank(&ncclComms[gpu], kGpus, id,
                               devices[gpu]->globalRank));
      }
      check(ncclGroupEnd());
      check(ncclGetVersion(&ncclVersion));
    }
    // Keep MPI/NCCL progress workers on the NUMA node while the routing
    // thread stays on one CPU, avoiding scheduler migration in the hot loop.
    cpu_set_t mainAffinity;
    CPU_ZERO(&mainAffinity);
    CPU_SET(localRank == 0 ? 0 : 1, &mainAffinity);
    if (sched_setaffinity(0, sizeof(mainAffinity), &mainAffinity))
      throw std::runtime_error("routing CPU affinity failed");
    if (rank == 0) {
      std::printf("# two-node host-RDMA AllGather; FP32; 4 GPUs/node; "
                  "2 MPI groups/node; 2 IB rails; chunk=%zu B; slots=%d; "
                  "download=%s\n", cfg.chunkBytes, kSlots,
                  cfg.dmaMulticast ? "dma" : "sm");
      std::printf("# timing=host enqueue+completion, slowest of 4 groups/8 GPUs; "
                  "warmups=%d iterations=%d repeats=%d; every output checked\n",
                  cfg.warmups, cfg.iterations, cfg.repeats);
      if (cfg.compareNccl)
        std::printf("# paired NCCL version=%d; same two GPU streams and "
                    "complete output check\n", ncclVersion);
      std::puts("collective,bytes,backend,median_us,min_us,max_us");
    }
    auto fill = [&](size_t fullBytes, int generation) {
      size_t shardElems = fullBytes / sizeof(float) / kGpus;
      for (auto const& device : devices) {
        check(cudaSetDevice(device->physical));
        fillInput<<<256, 256, 0, device->upload>>>(
            device->input, shardElems, device->globalRank, generation);
        check(cudaGetLastError());
        check(cudaMemsetAsync(device->output, 0xff, fullBytes,
                              device->consume));
        check(cudaDeviceSynchronize());
      }
    };
    auto verify = [&](size_t fullBytes, int generation) {
      size_t outputElems = fullBytes / sizeof(float);
      size_t shardElems = outputElems / kGpus;
      unsigned int localWrong = 0;
      for (auto const& device : devices) {
        check(cudaSetDevice(device->physical));
        check(cudaMemsetAsync(device->wrong, 0, sizeof(unsigned int),
                              device->consume));
        checkOutput<<<256, 256, 0, device->consume>>>(
            device->output, device->wrong, outputElems, shardElems,
            generation);
        check(cudaGetLastError());
        unsigned int wrong = 0;
        check(cudaMemcpyAsync(&wrong, device->wrong, sizeof(wrong),
                              cudaMemcpyDeviceToHost, device->consume));
        check(cudaStreamSynchronize(device->consume));
        localWrong += wrong;
      }
      unsigned int totalWrong = 0;
      check(MPI_Allreduce(&localWrong, &totalWrong, 1, MPI_UNSIGNED,
                           MPI_SUM, MPI_COMM_WORLD));
      if (totalWrong)
        throw std::runtime_error("AllGather complete output mismatch: " +
                                 std::to_string(totalWrong));
    };
    auto launch = [&](size_t fullBytes) {
      size_t shardBytes = fullBytes / kGpus;
      size_t pieces = (shardBytes + cfg.chunkBytes - 1) / cfg.chunkBytes;
      auto retire = [&](int slotIndex) {
        Slot& slot = *slots[slotIndex];
        if (!slot.active) return;
        if (!slot.sent || !slot.multicast)
          throw std::runtime_error("incomplete host packet slot");
        for (auto const& device : devices) {
          check(cudaSetDevice(device->physical));
          check(cudaEventSynchronize(device->consumeDone[slotIndex]));
        }
        check(MPI_Waitall(3, slot.outgoing.data(), MPI_STATUSES_IGNORE));
        slot.active = false;
      };
      auto startPiece = [&](size_t piece) {
        int slotIndex = int(piece % kSlots);
        retire(slotIndex);
        Slot& slot = *slots[slotIndex];
        slot.active = true;
        slot.sent = slot.multicast = false;
        slot.piece = int(piece);
        size_t offset = piece * cfg.chunkBytes;
        slot.length = std::min(cfg.chunkBytes, shardBytes - offset);
        int count = int(2 * slot.length);
        check(MPI_Irecv(slot.remote[0].host, count, MPI_BYTE,
                         remoteBase, kTagRemote, MPI_COMM_WORLD,
                         &slot.receive[0]));
        check(MPI_Irecv(slot.remote[1].host, count, MPI_BYTE,
                         remoteBase + 1, kTagRemote, MPI_COMM_WORLD,
                         &slot.receive[1]));
        check(MPI_Irecv(slot.local.host, count, MPI_BYTE,
                         localPeer, kTagLocal, MPI_COMM_WORLD,
                         &slot.receive[2]));
        for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
          Device& device = *devices[gpu];
          check(cudaSetDevice(device.physical));
          check(cudaMemcpyAsync(slot.send.host + gpu * slot.length,
                                reinterpret_cast<char*>(device.input) + offset,
                                slot.length, cudaMemcpyDeviceToHost,
                                device.upload));
          check(cudaEventRecord(device.uploadDone[slotIndex], device.upload));
        }
      };
      auto sendPiece = [&](size_t piece) {
        int slotIndex = int(piece % kSlots);
        Slot& slot = *slots[slotIndex];
        for (auto const& device : devices) {
          check(cudaSetDevice(device->physical));
          check(cudaEventSynchronize(device->uploadDone[slotIndex]));
        }
        int count = int(2 * slot.length);
        check(MPI_Isend(slot.send.host, count, MPI_BYTE,
                         remoteBase, kTagRemote, MPI_COMM_WORLD,
                         &slot.outgoing[0]));
        check(MPI_Isend(slot.send.host, count, MPI_BYTE,
                         remoteBase + 1, kTagRemote, MPI_COMM_WORLD,
                         &slot.outgoing[1]));
        check(MPI_Isend(slot.send.host, count, MPI_BYTE,
                         localPeer, kTagLocal, MPI_COMM_WORLD,
                         &slot.outgoing[2]));
        slot.sent = true;
      };
      auto multicastPiece = [&](size_t piece) {
        int slotIndex = int(piece % kSlots);
        Slot& slot = *slots[slotIndex];
        check(MPI_Waitall(3, slot.receive.data(), MPI_STATUSES_IGNORE));
        for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
          Device& device = *devices[gpu];
          check(cudaSetDevice(device.physical));
          Sources source{};
          for (int rankGpu = 0; rankGpu < kGpus; ++rankGpu) {
            int process = rankGpu / kGpusPerGroup;
            HostBlock* block = process == rank
                                   ? &slot.send
                                   : process == localPeer
                                         ? &slot.local
                                         : &slot.remote[process % kGroups];
            char const* hostSource = block->host +
                size_t(rankGpu % kGpusPerGroup) * slot.length;
            if (cfg.dmaMulticast)
              check(cudaMemcpyAsync(
                  reinterpret_cast<char*>(device.output) +
                      size_t(rankGpu) * shardBytes +
                      piece * cfg.chunkBytes,
                  hostSource, slot.length, cudaMemcpyHostToDevice,
                  device.consume));
            else
              source.pointer[rankGpu] = reinterpret_cast<uint4 const*>(
                  block->mapped[gpu] +
                  size_t(rankGpu % kGpusPerGroup) * slot.length);
          }
          if (!cfg.dmaMulticast) {
            size_t vectors = slot.length / sizeof(uint4);
            int blocks = int(std::min<size_t>(
                128, (vectors + 255) / 256));
            multicastPacket<<<dim3(blocks, kGpus), 256, 0, device.consume>>>(
                source, reinterpret_cast<uint4*>(device.output),
                shardBytes / sizeof(uint4),
                piece * cfg.chunkBytes / sizeof(uint4), vectors);
            check(cudaGetLastError());
          }
          check(cudaEventRecord(device.consumeDone[slotIndex],
                                device.consume));
        }
        slot.multicast = true;
      };
      for (size_t step = 0; step < pieces + 2; ++step) {
        if (step < pieces) startPiece(step);
        if (step >= 1 && step - 1 < pieces) sendPiece(step - 1);
        if (step >= 2 && step - 2 < pieces) multicastPiece(step - 2);
      }
      for (int slot = 0; slot < kSlots; ++slot) retire(slot);
    };
    auto launchNccl = [&](size_t fullBytes) {
      size_t shardElements = fullBytes / sizeof(float) / kGpus;
      check(ncclGroupStart());
      for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
        Device& device = *devices[gpu];
        check(cudaSetDevice(device.physical));
        check(ncclAllGather(device.input, device.output, shardElements,
                            ncclFloat, ncclComms[gpu], device.consume));
      }
      check(ncclGroupEnd());
      for (auto const& device : devices) {
        check(cudaSetDevice(device->physical));
        check(cudaStreamSynchronize(device->consume));
      }
    };
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
      std::array<std::vector<double>, 2> runs;
      for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
        std::vector<int> order{0};
        if (cfg.compareNccl) {
          if (repeat % 2 == 0) order.push_back(1);
          else order = {1, 0};
        }
        for (int backend : order) {
          double sum = 0;
          for (int call = 0; call < cfg.warmups + cfg.iterations; ++call) {
            int generation = 10 + repeat * (cfg.warmups + cfg.iterations) + call;
            fill(bytes, generation);
            check(MPI_Barrier(MPI_COMM_WORLD));
            auto start = std::chrono::steady_clock::now();
            if (backend == 0) launch(bytes);
            else launchNccl(bytes);
            auto end = std::chrono::steady_clock::now();
            double localUs = std::chrono::duration<double, std::micro>(
                                 end - start).count();
            double maxUs = 0;
            check(MPI_Allreduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX,
                                 MPI_COMM_WORLD));
            verify(bytes, generation);
            if (call >= cfg.warmups) {
              sum += maxUs;
              if (rank == 0)
                std::printf("# sample,allgather,%zu,%s,%d,%d,%.3f,0\n",
                            bytes, backend ? "nccl" : "host-rdma-multicast",
                            repeat + 1, call - cfg.warmups + 1, maxUs);
            }
          }
          runs[backend].push_back(sum / cfg.iterations);
        }
      }
      if (rank == 0) {
        for (int backend = 0; backend < (cfg.compareNccl ? 2 : 1); ++backend) {
          auto bounds = std::minmax_element(runs[backend].begin(),
                                             runs[backend].end());
          std::printf("allgather,%zu,%s,%.3f,%.3f,%.3f\n", bytes,
                      backend ? "nccl" : "host-rdma-multicast",
                      median(runs[backend]), *bounds.first, *bounds.second);
        }
        std::fflush(stdout);
      }
      if (bytes > cfg.end / 2) break;
    }
    if (cfg.compareNccl)
      for (auto comm : ncclComms) check(ncclCommDestroy(comm));
    slots = {};
    devices = {};
    check(MPI_Comm_free(&localComm));
    check(MPI_Finalize());
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "host RDMA AllGather: %s\n", error.what());
    if (mpiReady) MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
