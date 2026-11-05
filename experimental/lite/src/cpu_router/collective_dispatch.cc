#include "uccl_lite/collective_dispatch.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sched.h>

namespace uccl_lite {
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::string read(std::string const& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), {});
}
std::vector<int> cpuList(int node, cpu_set_t const& allowed) {
  std::istringstream input(read("/sys/devices/system/node/node" +
                                std::to_string(node) + "/cpulist"));
  std::string range;
  std::vector<int> result;
  while (std::getline(input, range, ',')) {
    auto dash = range.find('-');
    int begin = std::stoi(range), end = dash == std::string::npos
                                            ? begin
                                            : std::stoi(range.substr(dash + 1));
    if (begin < 0 || end >= CPU_SETSIZE || end < begin)
      throw std::runtime_error("invalid host CPU list");
    for (int cpu = begin; cpu <= end; ++cpu)
      if (CPU_ISSET(cpu, &allowed)) result.push_back(cpu);
  }
  return result;
}
cpu_router::PacketProgram program(CollectiveCall const& call) {
  int ranks = int(call.ports.size());
  switch (call.collective) {
    case Collective::AllReduce:
      return cpu_router::PacketProgram::allReduce(call.count, ranks);
    case Collective::AllGather:
      return cpu_router::PacketProgram::allGather(call.count, ranks);
    case Collective::ReduceScatter:
      return cpu_router::PacketProgram::reduceScatter(call.count, ranks);
  }
  throw std::invalid_argument("unsupported collective");
}
}  // namespace

Topology inspectTopology(std::vector<int> const& devices) {
  if (devices.empty() || devices.size() > 32 ||
      std::set<int>(devices.begin(), devices.end()).size() != devices.size())
    throw std::invalid_argument("expected 1..32 distinct local devices");
  Topology t;
  t.devices = devices;
  t.avx2 = __builtin_cpu_supports("avx2");
  check(cudaRuntimeGetVersion(&t.cudaRuntimeVersion));
  std::smatch match;
  std::string driver = read("/proc/driver/nvidia/version");
  if (std::regex_search(driver, match,
                        std::regex(R"(\b[0-9]{3}\.[0-9]+\.[0-9]+\b)")))
    t.driverVersion = match.str();
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed))
    throw std::runtime_error("cannot query CPU affinity");
  std::ostringstream identity;
  identity << "uccl-lite-router-v1;cuda=" << t.cudaRuntimeVersion
           << ";driver=" << t.driverVersion;
  identity << ";avx2=" << bool(__builtin_cpu_supports("avx2"))
           << ";avx512f=" << bool(__builtin_cpu_supports("avx512f"))
           << ";avx512bw=" << bool(__builtin_cpu_supports("avx512bw"));
  std::istringstream cpuInfo(read("/proc/cpuinfo"));
  std::string line;
  while (std::getline(cpuInfo, line))
    if (line.rfind("model name", 0) == 0) {
      identity << ";cpu=" << line;
      break;
    }
  for (int device : devices) {
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, device));
    int node = cpu_router::gpuNumaNode(device);
    t.numaNodes.push_back(node);
    t.gpuNames.emplace_back(property.name);
    t.mappedHost.push_back(property.canMapHostMemory);
    if (!t.permittedCpuIds.count(node))
      t.permittedCpuIds[node] = cpuList(node, allowed);
    t.permittedCpusPerNode[node] = int(t.permittedCpuIds[node].size());
    char pci[32];
    check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
    identity << ";gpu=" << device << ':' << pci << ':' << node << ':'
             << property.name << ':';
    for (unsigned char byte : property.uuid.bytes)
      identity << std::hex << std::setw(2) << std::setfill('0')
               << unsigned(byte);
    identity << std::dec;
    for (int peer : devices) {
      int accessible = 0;
      if (device != peer)
        check(cudaDeviceCanAccessPeer(&accessible, device, peer));
      t.peerAccess.push_back(accessible != 0);
      identity << accessible;
    }
  }
  for (auto const& entry : t.permittedCpuIds) {
    identity << ";node=" << entry.first << ":cpus=";
    for (int cpu : entry.second) identity << cpu << ',';
  }
  t.identity = identity.str();
  return t;
}

Workload CollectiveCall::workload() const {
  size_t scalar = dtype == ScalarType::Float32 ? 4 : 2;
  size_t expansion = collective == Collective::AllReduce ? 1 : ports.size();
  if (!count || !expansion ||
      count > std::numeric_limits<size_t>::max() / scalar / expansion)
    throw std::invalid_argument("empty or overflowing collective extent");
  return {collective, dtype, count * scalar * expansion, inPlace};
}

CollectiveDispatcher::CollectiveDispatcher(Topology topology,
                                           BackendPolicy policy,
                                           Fallback fallback,
                                           cpu_router::CpuRouterOptions options)
    : topology_(std::move(topology)),
      policy_(std::move(policy)),
      fallback_(std::move(fallback)),
      options_(std::move(options)) {
  if (!fallback_)
    throw std::invalid_argument("an existing-backend callback is required");
  if (options_.backend != cpu_router::TransportBackend::GpuKernel)
    throw std::invalid_argument(
        "adaptive CPU-router dispatch supports the SM transport only");
  if (options_.reduction != cpu_router::ReductionPlacement::Cpu &&
      options_.reduction != cpu_router::ReductionPlacement::Gpu)
    throw std::invalid_argument("invalid reduction placement");
  if (options_.interleaveScatter &&
      options_.reduction != cpu_router::ReductionPlacement::Gpu)
    throw std::invalid_argument("interleaved scatter requires GPU reduction");
  if (options_.gpuProgress &&
      options_.reduction != cpu_router::ReductionPlacement::Gpu)
    throw std::invalid_argument("GPU progress requires GPU reduction");
  topology_.referenceRouterOptions =
      options_.reduction == cpu_router::ReductionPlacement::Cpu &&
      options_.packetBytes == (512UL << 10) &&
      options_.minPacketBytes == (128UL << 10) &&
      options_.transportBlocks == 64 && options_.transportThreads == 256 &&
      options_.transportGrainBytes == (16UL << 10) &&
      options_.workersPerNode == 23 && options_.scatterWorkersPerNode == 16 &&
      options_.grainElements == 8192 && options_.workerCpus.empty() &&
      options_.numaNodes.empty() && options_.replayCpu == -1 &&
      !options_.inlineReplay && !options_.concurrentProducer;
  std::ostringstream key;
  // maxTensorBytes is a validation bound, not packet geometry or an allocation
  // size.
  key << ";options=" << options_.packetBytes << ',' << options_.minPacketBytes
      << ',' << options_.transportBlocks << ',' << options_.transportThreads
      << ',' << options_.transportGrainBytes << ',' << options_.workersPerNode
      << ',' << options_.scatterWorkersPerNode << ',' << options_.grainElements
      << ',' << options_.replayCpu << ',' << options_.inlineReplay << ','
      << options_.concurrentProducer;
  // Keep the legacy CPU identity stable, but never reuse its measurements for
  // a different placement. GPU mode needs its own exact local calibration.
  if (options_.reduction != cpu_router::ReductionPlacement::Cpu)
    key << ";reduction=gpu-v1";
  if (options_.interleaveScatter) key << ";scatter=interleaved-v2";
  if (options_.gpuProgress) key << ";progress=gpu-v1";
  for (int cpu : options_.workerCpus) key << ";worker=" << cpu;
  for (int node : options_.numaNodes) key << ";numa=" << node;
  topology_.identity += key.str();
  for (char const* knob : {"UCCL_LITE_ROUTER_PROFILE", "UCCL_LITE_TIMELINE",
                           "UCCL_LITE_GPU_TIMELINE_PATH"}) {
    bool enabled = std::getenv(knob) != nullptr;
    topology_.identity += std::string(";") + knob + "=" + (enabled ? "1" : "0");
    if (enabled) topology_.referenceRouterOptions = false;
  }
  // Use only the caller's currently permitted affinity, not the whole host
  // pool.
  if (options_.workerCpus.empty()) {
    std::vector<int> nodes;
    for (int node : topology_.numaNodes)
      if (std::find(nodes.begin(), nodes.end(), node) == nodes.end())
        nodes.push_back(node);
    for (int node : nodes)
      options_.workersPerNode = std::min(
          options_.workersPerNode,
          std::max(1, int(topology_.permittedCpuIds[node].size()) - 1));
    for (int node : nodes) {
      auto const& cpus = topology_.permittedCpuIds[node];
      if (cpus.size() < size_t(options_.workersPerNode))
        continue;  // Capability check will reject this topology.
      options_.workerCpus.insert(options_.workerCpus.end(), cpus.begin(),
                                 cpus.begin() + options_.workersPerNode);
    }
    if (options_.replayCpu < 0)
      for (int node : nodes)
        for (int cpu : topology_.permittedCpuIds[node])
          if (std::find(options_.workerCpus.begin(), options_.workerCpus.end(),
                        cpu) == options_.workerCpus.end())
            options_.replayCpu = cpu;
  }
}
CollectiveDispatcher::~CollectiveDispatcher() = default;

void CollectiveDispatcher::validatePorts(CollectiveCall const& call) const {
  if (call.ports.size() != topology_.devices.size())
    throw std::invalid_argument("port count changed");
  auto workload = call.workload();
  size_t scalar = call.dtype == ScalarType::Float32 ? 4 : 2;
  size_t contribution = call.count * scalar;
  size_t inputBytes = call.collective == Collective::ReduceScatter
                          ? workload.bytes
                          : contribution;
  size_t outputBytes =
      call.collective == Collective::AllGather ? workload.bytes : contribution;
  for (size_t rank = 0; rank < call.ports.size(); ++rank) {
    auto const& port = call.ports[rank];
    if (port.device != topology_.devices[rank] || !port.input || !port.output)
      throw std::invalid_argument(
          "invalid collective ports or changed device order");
    auto input = reinterpret_cast<uintptr_t>(port.input),
         output = reinterpret_cast<uintptr_t>(port.output);
    if (input > UINTPTR_MAX - inputBytes || output > UINTPTR_MAX - outputBytes)
      throw std::invalid_argument("collective pointer range overflow");
    if (call.inPlace) {
      bool aliases = call.collective == Collective::AllGather
                         ? input == output + rank * contribution
                     : call.collective == Collective::ReduceScatter
                         ? output == input + rank * contribution
                         : input == output;
      if (!aliases)
        throw std::invalid_argument(
            "in-place flag does not match buffer layout");
    } else if (input < output + outputBytes && output < input + inputBytes) {
      throw std::invalid_argument("out-of-place buffers overlap");
    }
  }
}
BackendDecision CollectiveDispatcher::decision(
    CollectiveCall const& call) const {
  validatePorts(call);
  Workload w = call.workload();
  if (w.bytes > options_.maxTensorBytes) {
    if (policy_.mode == RouterMode::Force)
      throw std::invalid_argument("CPU-router capacity exceeded");
    return {false, "CPU-router capacity exceeded; keep existing backend"};
  }
  return policy_.select(topology_, w);
}
BackendDecision CollectiveDispatcher::prepare(CollectiveCall const& call) {
  auto selected = decision(call);
  if (selected.cpuRouter && !router_) {
    for (auto const& port : call.ports) {
      check(cudaSetDevice(port.device));
      cudaStreamCaptureStatus status;
      check(cudaStreamIsCapturing(port.stream, &status));
      if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("prepare CPU-router before graph capture");
      check(cudaStreamSynchronize(port.stream));
    }
    auto router = std::make_unique<cpu_router::CpuRouter>(options_);
    router->prepare(call.ports);
    router_ = std::move(router);
  }
  return selected;
}
BackendDecision CollectiveDispatcher::enqueue(CollectiveCall const& call) {
  auto selected = decision(call);
  if (!selected.cpuRouter) {
    fallback_(call);
    return selected;
  }
  if (!router_)
    throw std::logic_error("call prepare before CPU-router enqueue");
  // Graph replay needs the core's explicit publication protocol. Do not
  // silently capture only GPU work.
  for (auto const& port : call.ports) {
    check(cudaSetDevice(port.device));
    cudaStreamCaptureStatus status;
    check(cudaStreamIsCapturing(port.stream, &status));
    if (status != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "adaptive router is eager-only; use PacketPipeline capture/replay "
          "API for graphs");
  }
  router_->enqueue(program(call), call.ports);
  return selected;
}
}  // namespace uccl_lite
