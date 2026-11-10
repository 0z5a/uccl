// Four-GPU direct peer AllGather and ReduceScatter benchmark.
// Every destination owns its output; source tensors stay immutable until all
// peer copies and the destination reduction have completed.
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kRanks = 4;
void cudaCheck(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void ncclCheck(ncclResult_t status) {
  if (status != ncclSuccess) throw std::runtime_error(ncclGetErrorString(status));
}
struct Config {
  size_t begin = 1UL << 10, end = 1UL << 26;
  int warmups = 5, iterations = 10, repeats = 3;
};
Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (option == "--help") {
      std::puts("Usage: uccl_lite_peer_ag_rs [-b BYTES] [-e BYTES] "
                "[-w WARMUPS] [-n ITERATIONS] [-r REPEATS]\n"
                "BYTES is full AllGather output / ReduceScatter input per "
                "rank; requires four peer-accessible GPUs.");
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    char* value = argv[++i];
    if (option == "-b") cfg.begin = std::strtoull(value, nullptr, 0);
    else if (option == "-e") cfg.end = std::strtoull(value, nullptr, 0);
    else if (option == "-w") cfg.warmups = std::atoi(value);
    else if (option == "-n") cfg.iterations = std::atoi(value);
    else if (option == "-r") cfg.repeats = std::atoi(value);
    else throw std::invalid_argument("unknown option: " + option);
  }
  if (cfg.begin < 16 || cfg.begin > cfg.end || cfg.begin % 16 ||
      cfg.end % 16 || cfg.warmups < 0 || cfg.iterations < 1 || cfg.repeats < 1)
    throw std::invalid_argument("invalid size or iteration settings");
  return cfg;
}
struct Device {
  int rank;
  float *input = nullptr, *output = nullptr, *scratch = nullptr,
        *reference = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, end = nullptr, transferDone = nullptr,
              reduceDone = nullptr;
  Device(int r, size_t bytes) : rank(r) {
    cudaCheck(cudaSetDevice(rank));
    cudaCheck(cudaMalloc(&input, bytes));
    cudaCheck(cudaMalloc(&output, bytes));
    cudaCheck(cudaMalloc(&scratch, bytes));
    cudaCheck(cudaMalloc(&reference, bytes));
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventCreateWithFlags(&transferDone, cudaEventDisableTiming));
    cudaCheck(cudaEventCreateWithFlags(&reduceDone, cudaEventDisableTiming));
  }
  ~Device() {
    cudaSetDevice(rank);
    if (reduceDone) cudaEventDestroy(reduceDone);
    if (transferDone) cudaEventDestroy(transferDone);
    if (end) cudaEventDestroy(end);
    if (start) cudaEventDestroy(start);
    if (stream) cudaStreamDestroy(stream);
    if (reference) cudaFree(reference);
    if (scratch) cudaFree(scratch);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
  }
};
__global__ void sumFour(float const* source, float* output, size_t shard) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= shard) return;
  float value = 0;
  for (int rank = 0; rank < kRanks; ++rank)
    value += source[size_t(rank) * shard + index];
  output[index] = value;
}
enum class Operation { Gather, Scatter };
enum class Backend { Peer, Nccl };
char const* opName(Operation op) {
  return op == Operation::Gather ? "all_gather" : "reduce_scatter";
}
char const* backendName(Backend backend) {
  return backend == Backend::Peer ? "peer" : "nccl";
}
void launchPeer(std::array<Device*, kRanks> const& devices, Operation op,
                size_t shard) {
  size_t bytes = shard * sizeof(float);
  for (int source = 0; source < kRanks; ++source) {
    auto& s = *devices[source];
    cudaCheck(cudaSetDevice(source));
    // The destination may still be reducing the previous generation. Retain
    // its scratch until that reader has completed.
    if (op == Operation::Scatter)
      for (int dest = 0; dest < kRanks; ++dest)
        cudaCheck(cudaStreamWaitEvent(s.stream, devices[dest]->reduceDone, 0));
    for (int phase = 0; phase < kRanks; ++phase) {
      int dest = (source + phase) % kRanks;
      float* target = op == Operation::Gather
                          ? devices[dest]->output + size_t(source) * shard
                          : devices[dest]->scratch + size_t(source) * shard;
      float const* input = op == Operation::Gather
                               ? s.input
                               : s.input + size_t(dest) * shard;
      if (source == dest)
        cudaCheck(cudaMemcpyAsync(target, input, bytes,
                                  cudaMemcpyDeviceToDevice, s.stream));
      else
        cudaCheck(cudaMemcpyPeerAsync(target, dest, input, source, bytes,
                                      s.stream));
    }
    cudaCheck(cudaEventRecord(s.transferDone, s.stream));
  }
  for (int dest = 0; dest < kRanks; ++dest) {
    auto& d = *devices[dest];
    cudaCheck(cudaSetDevice(dest));
    for (int source = 0; source < kRanks; ++source)
      cudaCheck(cudaStreamWaitEvent(d.stream, devices[source]->transferDone, 0));
    if (op == Operation::Scatter)
      sumFour<<<int((shard + 255) / 256), 256, 0, d.stream>>>(
          d.scratch, d.output, shard);
    if (op == Operation::Scatter)
      cudaCheck(cudaEventRecord(d.reduceDone, d.stream));
    cudaCheck(cudaGetLastError());
  }
}
void launchNccl(std::array<Device*, kRanks> const& devices,
                std::array<ncclComm_t, kRanks> const& comms, Operation op,
                size_t shard) {
  ncclCheck(ncclGroupStart());
  for (int rank = 0; rank < kRanks; ++rank) {
    auto& d = *devices[rank];
    cudaCheck(cudaSetDevice(rank));
    if (op == Operation::Gather)
      ncclCheck(ncclAllGather(d.input, d.reference, shard, ncclFloat,
                              comms[rank], d.stream));
    else
      ncclCheck(ncclReduceScatter(d.input, d.reference, shard, ncclFloat,
                                  ncclSum, comms[rank], d.stream));
  }
  ncclCheck(ncclGroupEnd());
}
void launch(std::array<Device*, kRanks> const& devices,
            std::array<ncclComm_t, kRanks> const& comms, Operation op,
            Backend backend, size_t shard) {
  if (backend == Backend::Peer) launchPeer(devices, op, shard);
  else launchNccl(devices, comms, op, shard);
}
void synchronize(std::array<Device*, kRanks> const& devices) {
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaStreamSynchronize(d->stream));
  }
}
void fill(std::array<Device*, kRanks> const& devices, Operation op,
          size_t shard, int generation) {
  size_t elements = shard * (op == Operation::Gather ? 1 : kRanks);
  std::vector<float> host(elements);
  for (auto* d : devices) {
    for (size_t i = 0; i < elements; ++i)
      host[i] = float(d->rank + 1 + generation * kRanks) *
                float(1 + i % 97);
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaMemcpy(d->input, host.data(), elements * sizeof(float),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d->output, 0xff, shard * kRanks * sizeof(float)));
    cudaCheck(cudaMemset(d->reference, 0xff,
                         shard * kRanks * sizeof(float)));
  }
}
void verify(std::array<Device*, kRanks> const& devices, Operation op,
            Backend backend, size_t shard, int generation) {
  size_t elements = shard * (op == Operation::Gather ? kRanks : 1);
  std::vector<float> host(elements);
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaMemcpy(host.data(), backend == Backend::Peer ? d->output
                                                              : d->reference,
                         elements * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elements; ++i) {
      float want = op == Operation::Gather
                       ? float(i / shard + 1 + generation * kRanks) *
                             float(1 + (i % shard) % 97)
                       : float(10 + generation * kRanks * kRanks) *
                             float(1 + (size_t(d->rank) * shard + i) % 97);
      if (std::fabs(host[i] - want) > 1e-5f) {
        std::fprintf(stderr,
                     "%s %s rank=%d index=%zu got=%.6f want=%.6f\n",
                     opName(op), backendName(backend), d->rank, i, host[i],
                     want);
        throw std::runtime_error("collective output mismatch");
      }
    }
  }
}
double measure(std::array<Device*, kRanks> const& devices,
               std::array<ncclComm_t, kRanks> const& comms, Operation op,
               Backend backend, size_t shard, Config const& cfg) {
  for (int n = 0; n < cfg.warmups; ++n) launch(devices, comms, op, backend, shard);
  synchronize(devices);
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaEventRecord(d->start, d->stream));
  }
  for (int n = 0; n < cfg.iterations; ++n)
    launch(devices, comms, op, backend, shard);
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaEventRecord(d->end, d->stream));
  }
  synchronize(devices);
  double slowest = 0;
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    float ms = 0;
    cudaCheck(cudaEventElapsedTime(&ms, d->start, d->end));
    slowest = std::max(slowest, double(ms) * 1000 / cfg.iterations);
  }
  return slowest;
}
double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = parse(argc, argv);
    int available = 0;
    cudaCheck(cudaGetDeviceCount(&available));
    if (available < kRanks)
      throw std::runtime_error("four GPUs are required");
    for (int rank = 0; rank < kRanks; ++rank) {
      for (int peer = 0; peer < kRanks; ++peer) {
        if (rank == peer) continue;
        int accessible = 0;
        cudaCheck(cudaDeviceCanAccessPeer(&accessible, rank, peer));
        if (!accessible) throw std::runtime_error("GPU peer access required");
        cudaCheck(cudaSetDevice(rank));
        auto result = cudaDeviceEnablePeerAccess(peer, 0);
        if (result == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else cudaCheck(result);
      }
    }
    std::array<std::unique_ptr<Device>, kRanks> owned;
    std::array<Device*, kRanks> devices{};
    for (int rank = 0; rank < kRanks; ++rank) {
      owned[rank] = std::make_unique<Device>(rank, cfg.end);
      devices[rank] = owned[rank].get();
    }
    std::array<ncclComm_t, kRanks> comms{};
    int ranks[kRanks] = {0, 1, 2, 3};
    ncclCheck(ncclCommInitAll(comms.data(), kRanks, ranks));
    int version = 0;
    ncclCheck(ncclGetVersion(&version));
    std::printf("# four L4 peer AG/RS, float32; NCCL version=%d; "
                "warmups=%d iterations=%d repeats=%d\n",
                version, cfg.warmups, cfg.iterations, cfg.repeats);
    std::puts("# full bytes per rank; slowest-rank CUDA-event microseconds; "
              "median of run-level means; every output checked");
    std::puts("collective,bytes,peer_median_us,peer_min_us,peer_max_us,"
              "nccl_median_us,nccl_min_us,nccl_max_us,speedup");
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
      size_t shard = bytes / sizeof(float) / kRanks;
      for (auto op : {Operation::Gather, Operation::Scatter}) {
        for (int generation = 0; generation < 2; ++generation) {
          fill(devices, op, shard, generation);
          for (auto backend : {Backend::Peer, Backend::Nccl}) {
            launch(devices, comms, op, backend, shard);
            synchronize(devices);
            verify(devices, op, backend, shard, generation);
          }
        }
        std::vector<double> peer, nccl;
        for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
          if (repeat % 2 == 0) {
            peer.push_back(measure(devices, comms, op, Backend::Peer, shard, cfg));
            nccl.push_back(measure(devices, comms, op, Backend::Nccl, shard, cfg));
          } else {
            nccl.push_back(measure(devices, comms, op, Backend::Nccl, shard, cfg));
            peer.push_back(measure(devices, comms, op, Backend::Peer, shard, cfg));
          }
        }
        verify(devices, op, Backend::Peer, shard, 1);
        verify(devices, op, Backend::Nccl, shard, 1);
        fill(devices, op, shard, 2);
        for (auto backend : {Backend::Peer, Backend::Nccl}) {
          launch(devices, comms, op, backend, shard);
          synchronize(devices);
          verify(devices, op, backend, shard, 2);
        }
        auto peerRange = std::minmax_element(peer.begin(), peer.end());
        auto ncclRange = std::minmax_element(nccl.begin(), nccl.end());
        std::printf("%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f\n",
                    opName(op), bytes, median(peer), *peerRange.first,
                    *peerRange.second, median(nccl), *ncclRange.first,
                    *ncclRange.second, median(nccl) / median(peer));
        std::fflush(stdout);
      }
      if (bytes > cfg.end / 2) break;
    }
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
  } catch (std::exception const& error) {
    std::fprintf(stderr, "peer AG/RS: %s\n", error.what());
    return 1;
  }
}
