#include "uccl_lite/collective_dispatch.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace uccl_lite;
namespace {
int checks = 0;
void require(bool condition, char const* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template <class F>
void rejects(F run, char const* message) {
  bool rejected = false;
  try {
    run();
  } catch (std::logic_error const&) {
    rejected = true;
  }
  require(rejected, message);
}
}  // namespace

int main() {
  try {
    // Host-only test: Off and unmatched Auto must not call CUDA or create
    // workers.
    Topology t;
    t.devices = {0, 1};
    t.numaNodes = {0, 1};
    t.gpuNames = {"test", "test"};
    t.mappedHost = {true, true};
    t.peerAccess = {false, false, false, false};
    t.avx2 = true;
    t.identity = "host-only-dispatch-test";
    t.permittedCpusPerNode = {{0, 2}, {1, 2}};
    t.permittedCpuIds = {{0, {0, 1}}, {1, {2, 3}}};
    std::array<std::array<uint16_t, 16>, 2> inputs{}, outputs{};
    CollectiveCall call;
    call.count = 4;
    for (int r = 0; r < 2; ++r)
      call.ports.push_back({inputs[r].data(), outputs[r].data(), nullptr, r});
    int fallbackCalls = 0;
    BackendPolicy policy;
    auto callback = [&](CollectiveCall const& c) {
      require(c.count == 4, "fallback receives original call");
      ++fallbackCalls;
    };
    CollectiveDispatcher automatic(t, policy, callback);
    require(!automatic.prepare(call).cpuRouter,
            "unmatched Auto keeps fallback");
    require(!automatic.routerInitialized(), "no speculative router allocation");
    automatic.enqueue(call);
    require(fallbackCalls == 1 && !automatic.routerInitialized(),
            "fallback invoked without router");
    call.dtype = ScalarType::Float32;
    automatic.enqueue(call);
    require(fallbackCalls == 2,
            "unsupported router dtype reaches caller backend");
    policy.mode = RouterMode::Force;
    CollectiveDispatcher forced(t, policy, callback);
    rejects([&] { forced.prepare(call); },
            "force cannot reinterpret float32 as BF16");
    call.dtype = ScalarType::BFloat16;
    rejects([&] { forced.enqueue(call); },
            "force requires explicit preparation");
    require(!forced.routerInitialized(), "enqueue must not allocate router");
    call.ports[0].output = inputs[0].data();
    rejects([&] { automatic.enqueue(call); },
            "out-of-place overlap is rejected");
    call.ports[0].output = outputs[0].data();
    call.inPlace = true;
    rejects([&] { automatic.enqueue(call); },
            "incorrect in-place layout rejected");
    call.collective = Collective::AllGather;
    for (int r = 0; r < 2; ++r)
      call.ports[r].input = outputs[r].data() + r * call.count;
    automatic.enqueue(call);
    require(fallbackCalls == 3, "valid in-place AllGather fallback");
    call.collective = Collective::ReduceScatter;
    for (int r = 0; r < 2; ++r) {
      call.ports[r].input = inputs[r].data();
      call.ports[r].output = inputs[r].data() + r * call.count;
    }
    automatic.enqueue(call);
    require(fallbackCalls == 4, "valid in-place ReduceScatter fallback");
    call.ports[1].device = 0;
    rejects([&] { automatic.enqueue(call); }, "changed device order rejected");
    call.ports[1].device = 1;
    call.count = std::numeric_limits<size_t>::max();
    rejects([&] { automatic.enqueue(call); }, "extent overflow rejected");
    require(!automatic.routerInitialized(),
            "all fallback tests avoid router initialization");
    call.count = 4;
    cpu_router::CpuRouterOptions small;
    small.maxTensorBytes = 4;
    BackendPolicy autoPolicy;
    CollectiveDispatcher limited(t, autoPolicy, callback, small);
    require(!limited.prepare(call).cpuRouter && !limited.routerInitialized(),
            "capacity miss retains fallback without allocation");
    CollectiveDispatcher forcedLimited(t, policy, callback, small);
    rejects([&] { forcedLimited.prepare(call); },
            "force rejects capacity overflow");
    BackendPolicy off;
    off.mode = RouterMode::Off;
    CollectiveDispatcher disabled(t, off, callback);
    require(!disabled.prepare(call).cpuRouter && !disabled.routerInitialized(),
            "explicit off remains uninitialized");
    cpu_router::CpuRouterOptions gpu;
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher gpuAutomatic(t, autoPolicy, callback, gpu);
    require(!gpuAutomatic.prepare(call).cpuRouter &&
                !gpuAutomatic.routerInitialized(),
            "unmeasured GPU reduction retains fallback without allocation");
    require(gpuAutomatic.topology().identity != automatic.topology().identity,
            "GPU reduction requires its own local calibration");
    gpu.interleaveScatter = true;
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher striped(t, autoPolicy, callback, gpu);
    require(striped.topology().identity != gpuAutomatic.topology().identity &&
                !striped.prepare(call).cpuRouter &&
                !striped.routerInitialized(),
            "striped placement requires its own measurements and stays "
            "uninitialized");
    gpu.gpuProgress = true;
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher direct(t, autoPolicy, callback, gpu);
    require(direct.topology().identity != striped.topology().identity &&
                !direct.prepare(call).cpuRouter && !direct.routerInitialized(),
            "GPU progress needs separate evidence and remains uninitialized on "
            "fallback");
    gpu.interleaveScatter = false;
    gpu.reduction = cpu_router::ReductionPlacement::Cpu;
    rejects([&] { CollectiveDispatcher invalid(t, autoPolicy, callback, gpu); },
            "GPU progress cannot bypass CPU arithmetic");
    gpu.gpuProgress = false;
    gpu.interleaveScatter = true;
    rejects([&] { CollectiveDispatcher invalid(t, autoPolicy, callback, gpu); },
            "interleaved scatter cannot silently use CPU reduction");
    gpu.reduction = static_cast<cpu_router::ReductionPlacement>(99);
    rejects(
        [&] { CollectiveDispatcher invalid(t, autoPolicy, callback, gpu); },
        "invalid reduction placement is rejected before CUDA initialization");
    std::cout << "Dispatcher: " << checks
              << " checks passed; no GPU initialization needed\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
