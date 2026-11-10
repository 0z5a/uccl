#include "uccl_lite/local_collectives.hpp"
#include <linux/mempolicy.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void* pinnedNuma(size_t bytes, int node) {
  if (node < 0 || node >= 64) throw std::invalid_argument("invalid NUMA node");
  void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) throw std::runtime_error("host staging mmap failed");
  unsigned long mask = 1UL << node;
  if (syscall(SYS_mbind, ptr, bytes, MPOL_BIND, &mask, 64, 0)) {
    munmap(ptr, bytes);
    throw std::runtime_error("host staging mbind failed");
  }
  std::memset(ptr, 0, bytes);
  auto status = cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
  if (status != cudaSuccess) {
    munmap(ptr, bytes);
    check(status);
  }
  return ptr;
}
__global__ void sumFour(__nv_bfloat16 const* own, __nv_bfloat16 const* peers,
                        __nv_bfloat16* output, size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    float sum = __bfloat162float(own[i]);
    for (int p = 0; p < 3; ++p)
      sum += __bfloat162float(peers[size_t(p) * count + i]);
    output[i] = __float2bfloat16_rn(sum);
  }
}
__global__ void sumTwo(__nv_bfloat16* partial, __nv_bfloat16 const* remote,
                       size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    partial[i] = __float2bfloat16_rn(__bfloat162float(partial[i]) +
                                     __bfloat162float(remote[i]));
}
}  // namespace

LocalCollectives::LocalCollectives(Topology topology, size_t maxTensorBytes)
    : topology_(std::move(topology)), maxTensorBytes_(maxTensorBytes) {
  if (topology_.devices.size() != 8 || maxTensorBytes_ < 16 ||
      maxTensorBytes_ % 16 ||
      maxTensorBytes_ > std::numeric_limits<size_t>::max() / 4 ||
      topology_.numaNodes.size() != 8 || topology_.peerAccess.size() != 64)
    throw std::invalid_argument(
        "local collectives need eight GPUs and aligned capacity");
  std::map<int, int> nodes;
  for (int rank = 0; rank < 8; ++rank) {
    int node = topology_.numaNodes[rank];
    if (!nodes.count(node)) {
      if (nodes.size() == 2)
        throw std::invalid_argument("local collectives need two NUMA nodes");
      nodes[node] = int(nodes.size());
    }
    int group = nodes[node];
    groupOfRank_[rank] = group;
    localRank_[rank] = int(groups_[group].size());
    groups_[group].push_back(rank);
  }
  if (groups_[0].size() != 4 || groups_[1].size() != 4)
    throw std::invalid_argument(
        "local collectives need four GPUs per NUMA node");
  for (auto const& group : groups_)
    for (int source : group)
      for (int dest : group)
        if (source != dest && !topology_.peerAccess[source * 8 + dest])
          throw std::invalid_argument("local group lacks CUDA peer access");
}

LocalCollectives::~LocalCollectives() {
  for (int group = 0; group < 2; ++group) {
    if (scratch_[group] || remote_[group]) {
      cudaSetDevice(topology_.devices[groups_[group][0]]);
      if (scratch_[group]) cudaFree(scratch_[group]);
      if (remote_[group]) cudaFree(remote_[group]);
    }
    if (host_[group]) {
      cudaHostUnregister(host_[group]);
      munmap(host_[group], maxTensorBytes_);
    }
  }
}

void LocalCollectives::prepare() {
  if (initialized_) return;
  for (int group = 0; group < 2; ++group) {
    int device = topology_.devices[groups_[group][0]];
    check(cudaSetDevice(device));
    if (!host_[group])
      host_[group] =
          pinnedNuma(maxTensorBytes_, topology_.numaNodes[groups_[group][0]]);
    if (!scratch_[group])
      check(cudaMalloc(&scratch_[group], 4 * maxTensorBytes_));
    if (!remote_[group]) check(cudaMalloc(&remote_[group], maxTensorBytes_));
  }
  initialized_ = true;
}

void LocalCollectives::syncAll(CollectiveCall const& call) const {
  for (auto const& port : call.ports) {
    check(cudaSetDevice(port.device));
    check(cudaStreamSynchronize(port.stream));
  }
}
void LocalCollectives::syncLeaders(CollectiveCall const& call) const {
  for (auto const& group : groups_) {
    auto const& port = call.ports[group[0]];
    check(cudaSetDevice(port.device));
    check(cudaStreamSynchronize(port.stream));
  }
}

void LocalCollectives::enqueue(CollectiveCall const& call) {
  if (!initialized_)
    throw std::logic_error("prepare local collectives before enqueue");
  if (call.dtype != ScalarType::BFloat16 || call.ports.size() != 8)
    throw std::invalid_argument("local collectives support eight BF16 ranks");
  if (call.collective != Collective::AllReduce &&
      call.collective != Collective::AllGather &&
      call.collective != Collective::ReduceScatter)
    throw std::invalid_argument("unsupported local collective");
  size_t expansion = call.collective == Collective::AllReduce ? 1 : 8;
  if (!call.count || call.count > std::numeric_limits<size_t>::max() /
                                      (sizeof(__nv_bfloat16) * expansion))
    throw std::invalid_argument("invalid local collective extent");
  size_t bytes = call.count * sizeof(__nv_bfloat16) * expansion;
  if (bytes > maxTensorBytes_ || bytes % 16)
    throw std::invalid_argument(
        "local collective size exceeds capacity or alignment");
  for (int rank = 0; rank < 8; ++rank) {
    auto const& port = call.ports[rank];
    if (port.device != topology_.devices[rank] || !port.input || !port.output ||
        !port.stream)
      throw std::invalid_argument("local collective ports changed");
    check(cudaSetDevice(port.device));
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(port.stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("local collectives are eager-only");
  }
  if (call.collective == Collective::AllGather)
    gather(call, bytes);
  else
    reduce(call, bytes);
}

void LocalCollectives::gather(CollectiveCall const& call, size_t bytes) {
  size_t shard = bytes / 8;
  for (int source = 0; source < 8; ++source) {
    int group = groupOfRank_[source];
    auto const& port = call.ports[source];
    check(cudaSetDevice(port.device));
    auto* host = static_cast<char*>(host_[group]) + localRank_[source] * shard;
    check(cudaMemcpyAsync(host, port.input, shard, cudaMemcpyDeviceToHost,
                          port.stream));
  }
  syncAll(call);
  for (int dest = 0; dest < 8; ++dest) {
    auto const& port = call.ports[dest];
    check(cudaSetDevice(port.device));
    for (int source = 0; source < 8; ++source) {
      auto* output = static_cast<char*>(port.output) + source * shard;
      if (dest == source) {
        if (!call.inPlace)
          check(cudaMemcpyAsync(output, call.ports[source].input, shard,
                                cudaMemcpyDeviceToDevice, port.stream));
      } else {
        auto* host = static_cast<char*>(host_[groupOfRank_[source]]) +
                     localRank_[source] * shard;
        check(cudaMemcpyAsync(output, host, shard, cudaMemcpyHostToDevice,
                              port.stream));
      }
    }
  }
  syncAll(call);
}

void LocalCollectives::reduce(CollectiveCall const& call, size_t bytes) {
  size_t count = bytes / sizeof(__nv_bfloat16);
  for (int group = 0; group < 2; ++group) {
    auto const& members = groups_[group];
    auto const& leader = call.ports[members[0]];
    for (int index = 1; index < 4; ++index) {
      auto const& source = call.ports[members[index]];
      check(cudaSetDevice(source.device));
      check(cudaMemcpyPeerAsync(
          static_cast<char*>(scratch_[group]) + index * bytes, leader.device,
          source.input, source.device, bytes, source.stream));
    }
  }
  syncAll(call);
  for (int group = 0; group < 2; ++group) {
    auto const& leader = call.ports[groups_[group][0]];
    check(cudaSetDevice(leader.device));
    auto* partial = static_cast<__nv_bfloat16*>(scratch_[group]);
    auto* peers = reinterpret_cast<__nv_bfloat16*>(
        static_cast<char*>(scratch_[group]) + bytes);
    sumFour<<<64, 256, 0, leader.stream>>>(
        static_cast<__nv_bfloat16 const*>(leader.input), peers, partial, count);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(host_[group], partial, bytes, cudaMemcpyDeviceToHost,
                          leader.stream));
  }
  syncLeaders(call);
  for (int group = 0; group < 2; ++group) {
    auto const& leader = call.ports[groups_[group][0]];
    check(cudaSetDevice(leader.device));
    check(cudaMemcpyAsync(remote_[group], host_[group ^ 1], bytes,
                          cudaMemcpyHostToDevice, leader.stream));
    sumTwo<<<64, 256, 0, leader.stream>>>(
        static_cast<__nv_bfloat16*>(scratch_[group]),
        static_cast<__nv_bfloat16 const*>(remote_[group]), count);
    check(cudaGetLastError());
  }
  syncLeaders(call);
  for (int rank = 0; rank < 8; ++rank) {
    auto const& port = call.ports[rank];
    int group = groupOfRank_[rank];
    auto const& leader = call.ports[groups_[group][0]];
    size_t offset = call.collective == Collective::ReduceScatter
                        ? size_t(rank) * bytes / 8
                        : 0;
    size_t copyBytes =
        call.collective == Collective::ReduceScatter ? bytes / 8 : bytes;
    auto* source = static_cast<char*>(scratch_[group]) + offset;
    check(cudaSetDevice(port.device));
    if (port.device == leader.device)
      check(cudaMemcpyAsync(port.output, source, copyBytes,
                            cudaMemcpyDeviceToDevice, port.stream));
    else
      check(cudaMemcpyPeerAsync(port.output, port.device, source, leader.device,
                                copyBytes, port.stream));
  }
  syncAll(call);
}
}  // namespace uccl_lite
