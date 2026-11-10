#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace uccl_lite {

enum class Collective { AllReduce, AllGather, ReduceScatter };
enum class ScalarType { BFloat16, Float16, Float32 };
enum class RouterMode { Auto, Off, Force };

struct Workload {
  Collective collective = Collective::AllReduce;
  ScalarType dtype = ScalarType::BFloat16;
  // Complete per-rank output for AG, complete input for RS, tensor size for AR.
  size_t bytes = 0;
  bool inPlace = false;
};

struct Topology {
  std::vector<int> devices, numaNodes;
  std::vector<std::string> gpuNames;
  std::vector<bool> mappedHost, peerAccess;  // Peer matrix is row-major.
  std::map<int, int> permittedCpusPerNode;
  std::map<int, std::vector<int>> permittedCpuIds;
  bool avx2 = false;
  int cudaRuntimeVersion = 0;
  std::string driverVersion;
  // Exact local identity: UUIDs, PCI IDs, CPU/affinity, driver/toolkit, router
  // knobs.
  std::string identity;
};

struct TimingRange {
  double minimum = 0, median = 0, maximum = 0;
};

struct Measurement {
  std::string topologyIdentity, fallbackIdentity;
  Workload workload;
  TimingRange router, fallback;
  int repetitions = 0;
  bool correct = false;
};

struct BackendDecision {
  bool cpuRouter = false;
  std::string reason;
  double measuredSpeedup = 0;
};

// CPU-only selection logic. Never benchmarks, allocates GPU memory, or starts
// workers.
class BackendPolicy {
 public:
  RouterMode mode = RouterMode::Auto;
  double minimumSpeedup = 1.03;
  std::string fallbackIdentity;
  std::vector<Measurement> measurements;

  BackendDecision select(Topology const& topology,
                         Workload const& workload) const;
};

char const* collectiveName(Collective collective);
Collective parseCollective(std::string const& name);
RouterMode parseRouterMode(std::string const& name);

// Archive reader for offline comparisons and integrity checks. Historical
// records never enable automatic routing on a live host.
std::vector<Measurement> loadL4ReferenceProfile(std::string const& path);
std::vector<Measurement> loadRouterProfile(std::string const& path);
void saveRouterProfile(std::string const& path,
                       std::vector<Measurement> const& measurements);

}  // namespace uccl_lite
