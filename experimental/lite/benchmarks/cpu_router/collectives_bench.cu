// Standalone BF16 AR/AG/RS dispatcher example and paired local calibration.
// The optional module is independent of NCCL; only this fallback harness links
// it.
#include "nccl_identity.hpp"
#include "uccl_lite/collective_dispatch.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <nccl.h>

using namespace uccl_lite;
extern char** environ;
namespace {
void cudaCheck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void ncclCheck(ncclResult_t e) {
  if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e));
}
__global__ void fill(__nv_bfloat16* data, size_t count, int rank) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    data[i] = __float2bfloat16(float((rank + 1) * (1 + i % 7)) / 64.0f);
}
__global__ void verify(__nv_bfloat16 const* data, size_t count, int rank,
                       int ranks, int operation, size_t contribution,
                       int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    size_t index = operation == 2 ? rank * contribution + i : i;
    float expected =
        operation == 1
            ? float((i / contribution + 1) * (1 + (i % contribution) % 7)) /
                  64.0f
            : float(ranks * (ranks + 1) / 2 * (1 + index % 7)) / 64.0f;
    if (__bfloat162float(data[i]) != expected) atomicAdd(errors, 1);
  }
}
struct Config {
  std::vector<int> devices;
  int gpus = 4, warmup = 3, iterations = 10, repeats = 3;
  size_t begin = 1024, end = 1 << 20;
  size_t packetBytes = 512UL << 10;
  bool inPlace = false;
  bool gpuReduce = false;
  bool interleaveScatter = false;
  std::string operation = "all", profile, reference, calibrate, samples;
  RouterMode mode = RouterMode::Auto;
};
struct Buffers {
  std::vector<void*> inputs, outputs;
  std::vector<int*> errors;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> start, stop;
  std::vector<int> devices;
  Buffers(std::vector<int> const& ids, size_t bytes) : devices(ids) {
    for (int id : ids) {
      cudaCheck(cudaSetDevice(id));
      void *in, *out;
      int* err;
      cudaStream_t stream;
      cudaEvent_t a, b;
      cudaCheck(cudaMalloc(&in, bytes));
      cudaCheck(cudaMalloc(&out, bytes));
      cudaCheck(cudaMalloc(&err, sizeof(int)));
      cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cudaCheck(cudaEventCreate(&a));
      cudaCheck(cudaEventCreate(&b));
      inputs.push_back(in);
      outputs.push_back(out);
      errors.push_back(err);
      streams.push_back(stream);
      start.push_back(a);
      stop.push_back(b);
    }
  }
  ~Buffers() {
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaStreamSynchronize(streams[r]);
      cudaFree(inputs[r]);
      cudaFree(outputs[r]);
      cudaFree(errors[r]);
      cudaEventDestroy(start[r]);
      cudaEventDestroy(stop[r]);
      cudaStreamDestroy(streams[r]);
    }
  }
  CollectiveCall call(Collective op, size_t bytes, bool inPlace) {
    CollectiveCall result;
    result.collective = op;
    result.inPlace = inPlace;
    result.count =
        bytes / 2 / (op == Collective::AllReduce ? 1 : devices.size());
    for (size_t rank = 0; rank < devices.size(); ++rank) {
      auto* base =
          static_cast<__nv_bfloat16*>(inPlace ? outputs[rank] : inputs[rank]);
      auto* out = static_cast<__nv_bfloat16*>(outputs[rank]);
      auto* in = base;
      if (inPlace && op == Collective::AllGather) in += rank * result.count;
      if (inPlace && op == Collective::ReduceScatter)
        out += rank * result.count;
      result.ports.push_back({in, out, streams[rank], devices[rank]});
    }
    return result;
  }
};
TimingRange summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  return {samples.front(),
          n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2,
          samples.back()};
}
double run(CollectiveDispatcher& dispatch, CollectiveCall const& call,
           Buffers& buffers, int iterations, bool timed,
           std::ostream* samples = nullptr, char const* backend = "",
           int repeat = -1) {
  std::vector<double> latencies;
  int ranks = int(call.ports.size());
  size_t inputCount =
      call.count * (call.collective == Collective::ReduceScatter ? ranks : 1);
  size_t outputCount =
      call.count * (call.collective == Collective::AllGather ? ranks : 1);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (int rank = 0; rank < ranks; ++rank) {
      auto const& p = call.ports[rank];
      cudaCheck(cudaSetDevice(p.device));
      fill<<<32, 256, 0, p.stream>>>(
          static_cast<__nv_bfloat16*>(const_cast<void*>(p.input)), inputCount,
          rank);
      cudaCheck(
          cudaMemsetAsync(buffers.errors[rank], 0, sizeof(int), p.stream));
      cudaCheck(cudaEventRecord(buffers.start[rank], p.stream));
    }
    dispatch.enqueue(call);
    for (int rank = 0; rank < ranks; ++rank) {
      auto const& p = call.ports[rank];
      cudaCheck(cudaSetDevice(p.device));
      cudaCheck(cudaEventRecord(buffers.stop[rank], p.stream));
      verify<<<32, 256, 0, p.stream>>>(
          static_cast<__nv_bfloat16 const*>(p.output), outputCount, rank, ranks,
          int(call.collective), call.count, buffers.errors[rank]);
      cudaCheck(cudaGetLastError());
    }
    double worst = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      cudaCheck(cudaSetDevice(call.ports[rank].device));
      cudaCheck(cudaStreamSynchronize(call.ports[rank].stream));
      int errors = 0;
      cudaCheck(cudaMemcpy(&errors, buffers.errors[rank], sizeof(int),
                           cudaMemcpyDeviceToHost));
      if (errors)
        throw std::runtime_error("collective correctness failed on rank " +
                                 std::to_string(rank));
      float ms;
      cudaCheck(
          cudaEventElapsedTime(&ms, buffers.start[rank], buffers.stop[rank]));
      worst = std::max(worst, double(ms) * 1000);
    }
    if (timed) {
      latencies.push_back(worst);
      if (samples)
        *samples << collectiveName(call.collective) << ','
                 << call.workload().bytes << ',' << call.inPlace << ','
                 << backend << ',' << repeat << ',' << iteration << ',' << worst
                 << '\n';
    }
  }
  return timed ? summarize(latencies).median : 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto next = [&]() {
        if (++i >= argc) throw std::invalid_argument("missing option value");
        return std::string(argv[i]);
      };
      if (arg == "--gpus")
        cfg.gpus = std::stoi(next());
      else if (arg == "--devices") {
        std::istringstream list(next());
        std::string d;
        while (std::getline(list, d, ',')) cfg.devices.push_back(std::stoi(d));
      } else if (arg == "--begin")
        cfg.begin = std::stoull(next());
      else if (arg == "--end")
        cfg.end = std::stoull(next());
      else if (arg == "--packet-bytes")
        cfg.packetBytes = std::stoull(next());
      else if (arg == "--op")
        cfg.operation = next();
      else if (arg == "--backend")
        cfg.mode = parseRouterMode(next());
      else if (arg == "--profile")
        cfg.profile = next();
      else if (arg == "--reference")
        cfg.reference = next();
      else if (arg == "--calibrate")
        cfg.calibrate = next();
      else if (arg == "--samples")
        cfg.samples = next();
      else if (arg == "--in-place")
        cfg.inPlace = true;
      else if (arg == "--gpu-reduce")
        cfg.gpuReduce = true;
      else if (arg == "--interleave-scatter") {
        cfg.gpuReduce = true;
        cfg.interleaveScatter = true;
      } else if (arg == "--warmup")
        cfg.warmup = std::stoi(next());
      else if (arg == "--iterations")
        cfg.iterations = std::stoi(next());
      else if (arg == "--repeats")
        cfg.repeats = std::stoi(next());
      else if (arg == "--help") {
        std::puts(
            "UCCL Lite BF16 collectives (one process owns all selected GPUs)\n"
            "  --gpus N | --devices 0,1,...\n  --op "
            "all|allreduce|allgather|reducescatter\n"
            "  --backend auto|off|force  --begin BYTES --end BYTES --in-place\n"
            "  --reference CSV  --profile FILE  --calibrate NEW_FILE\n"
            "  --samples NEW_CSV (checked per-invocation timing records)\n"
            "  --gpu-reduce (experimental destination-GPU Sum arithmetic)\n"
            "  --interleave-scatter (GPU reduction with all-destination packet "
            "stripes)\n"
            "  --packet-bytes BYTES (upper bound; default 524288, minimum "
            "131072)\n"
            "  --warmup N --iterations N --repeats N (at least 3)\n"
            "Auto keeps NCCL unless matching validated evidence shows a stable "
            "gain.\n"
            "Calibration alternates forced CPU-router and NCCL; checks every "
            "invocation.");
        return 0;
      } else
        throw std::invalid_argument("unknown option: " + arg);
    }
    if (cfg.devices.empty())
      for (int r = 0; r < cfg.gpus; ++r) cfg.devices.push_back(r);
    if (cfg.devices.empty() || cfg.devices.size() > 32 || cfg.begin < 128 ||
        cfg.end < cfg.begin || cfg.end > (1UL << 30) ||
        cfg.begin % (cfg.devices.size() * 2) || cfg.warmup < 1 ||
        cfg.iterations < 1 || cfg.repeats < 3 ||
        cfg.packetBytes < (128UL << 10) || cfg.packetBytes > (32UL << 20) ||
        cfg.packetBytes % 16)
      throw std::invalid_argument(
          "invalid device count, size, or iteration settings");
    if (!cfg.calibrate.empty() && std::filesystem::exists(cfg.calibrate))
      throw std::invalid_argument(
          "calibration output exists; choose a new file");
    std::ofstream sampleOutput;
    if (!cfg.samples.empty()) {
      if (std::filesystem::exists(cfg.samples) ||
          (!cfg.calibrate.empty() &&
           std::filesystem::absolute(cfg.samples).lexically_normal() ==
               std::filesystem::absolute(cfg.calibrate).lexically_normal()))
        throw std::invalid_argument(
            "sample output must be a new, distinct file");
      sampleOutput.exceptions(std::ios::failbit | std::ios::badbit);
      sampleOutput.open(cfg.samples);
      sampleOutput.precision(10);
      sampleOutput << "collective,bytes,in_place,backend,repeat,iteration,"
                      "slowest_rank_us\n";
    }
    auto topology = inspectTopology(cfg.devices);
    // Invalidate pre-fingerprint local profiles even when this invocation has
    // no NCCL overrides. Historical reference matching is a separate opt-in.
    topology.identity += ";benchmark-nccl-config-v1";
    int ncclVersion;
    ncclCheck(ncclGetVersion(&ncclVersion));
    BackendPolicy policy;
    policy.mode = cfg.mode;
    policy.fallbackIdentity = benchmark::ncclFallbackIdentity(
        ncclVersion, benchmark::ncclConfiguration(environ));
    if (!cfg.reference.empty())
      policy.measurements = loadL4ReferenceProfile(cfg.reference);
    if (!cfg.profile.empty()) {
      auto local = loadRouterProfile(cfg.profile);
      policy.measurements.insert(policy.measurements.end(), local.begin(),
                                 local.end());
    }
    std::vector<ncclComm_t> comms(cfg.devices.size());
    ncclCheck(
        ncclCommInitAll(comms.data(), int(comms.size()), cfg.devices.data()));
    CollectiveDispatcher::Fallback fallback = [&](CollectiveCall const& call) {
      ncclCheck(ncclGroupStart());
      for (size_t r = 0; r < call.ports.size(); ++r) {
        const auto& p = call.ports[r];
        cudaCheck(cudaSetDevice(p.device));
        if (call.collective == Collective::AllReduce)
          ncclCheck(ncclAllReduce(p.input, p.output, call.count, ncclBfloat16,
                                  ncclSum, comms[r], p.stream));
        else if (call.collective == Collective::AllGather)
          ncclCheck(ncclAllGather(p.input, p.output, call.count, ncclBfloat16,
                                  comms[r], p.stream));
        else
          ncclCheck(ncclReduceScatter(p.input, p.output, call.count,
                                      ncclBfloat16, ncclSum, comms[r],
                                      p.stream));
      }
      ncclCheck(ncclGroupEnd());
    };
    std::vector<Measurement> calibrated;
    {
      Buffers buffers(cfg.devices, cfg.end);
      cpu_router::CpuRouterOptions options;
      options.maxTensorBytes = cfg.end;
      options.packetBytes = cfg.packetBytes;
      options.reduction = cfg.gpuReduce ? cpu_router::ReductionPlacement::Gpu
                                        : cpu_router::ReductionPlacement::Cpu;
      options.interleaveScatter = cfg.interleaveScatter;
      char const* routerName = cfg.interleaveScatter ? "cpu-router-gpu-striped"
                               : cfg.gpuReduce       ? "cpu-router-gpu-reduce"
                                                     : "cpu-router";
      CollectiveDispatcher selected(topology, policy, fallback, options);
      BackendPolicy forced = policy, off = policy;
      forced.mode = RouterMode::Force;
      off.mode = RouterMode::Off;
      std::unique_ptr<CollectiveDispatcher> router, baseline;
      if (!cfg.calibrate.empty()) {
        router = std::make_unique<CollectiveDispatcher>(topology, forced,
                                                        fallback, options);
        baseline = std::make_unique<CollectiveDispatcher>(topology, off,
                                                          fallback, options);
      }
      std::vector<Collective> operations =
          cfg.operation == "all"
              ? std::vector<Collective>{Collective::AllReduce,
                                        Collective::AllGather,
                                        Collective::ReduceScatter}
              : std::vector<Collective>{parseCollective(cfg.operation)};
      std::printf(
          "# GPUs=%zu dtype=BF16 fallback=%s; no hidden calibration; checks "
          "outside timed region\n",
          cfg.devices.size(), policy.fallbackIdentity.c_str());
      std::printf("# topology=%s\n", topology.identity.c_str());
      std::puts(
          "collective,bytes,in_place,backend,median_us,min_us,max_us,speedup,"
          "dispatcher_router_initialized");
      for (auto op : operations)
        for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
          auto call = buffers.call(op, bytes, cfg.inPlace);
          auto decision = selected.prepare(call);
          std::fprintf(stderr, "%s %zu: %s\n", collectiveName(op), bytes,
                       decision.reason.c_str());
          if (router) {
            router->prepare(call);
            baseline->prepare(call);
            run(*baseline, call, buffers, cfg.warmup, false);
            run(*router, call, buffers, cfg.warmup, false);
            std::vector<double> cpuSamples, baselineSamples;
            auto measure = [&](CollectiveDispatcher& dispatcher,
                               char const* backend, int repeat) {
              return run(dispatcher, call, buffers, cfg.iterations, true,
                         cfg.samples.empty() ? nullptr : &sampleOutput, backend,
                         repeat);
            };
            for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
              if (repeat % 2 == 0) {
                baselineSamples.push_back(measure(*baseline, "nccl", repeat));
                cpuSamples.push_back(measure(*router, routerName, repeat));
              } else {
                cpuSamples.push_back(measure(*router, routerName, repeat));
                baselineSamples.push_back(measure(*baseline, "nccl", repeat));
              }
            }
            Measurement m{router->topology().identity,
                          policy.fallbackIdentity,
                          call.workload(),
                          summarize(cpuSamples),
                          summarize(baselineSamples),
                          cfg.repeats,
                          true};
            calibrated.push_back(m);
            BackendPolicy measured = policy;
            measured.mode = RouterMode::Auto;
            measured.measurements = {m};
            auto verdict = measured.select(router->topology(), call.workload());
            for (int backend = 0; backend < 2; ++backend) {
              auto t = backend ? m.router : m.fallback;
              std::printf("%s,%zu,%d,%s,%.3f,%.3f,%.3f,%.4f,%d\n",
                          collectiveName(op), bytes, cfg.inPlace,
                          backend ? routerName : "fallback", t.median,
                          t.minimum, t.maximum, m.fallback.median / t.median,
                          backend ? router->routerInitialized()
                                  : baseline->routerInitialized());
            }
            std::fprintf(stderr, "calibrated auto=%s: %s\n",
                         verdict.cpuRouter ? "cpu-router" : "fallback",
                         verdict.reason.c_str());
          } else {
            run(selected, call, buffers, cfg.warmup, false);
            std::vector<double> samples;
            for (int repeat = 0; repeat < cfg.repeats; ++repeat)
              samples.push_back(
                  run(selected, call, buffers, cfg.iterations, true,
                      cfg.samples.empty() ? nullptr : &sampleOutput,
                      decision.cpuRouter ? routerName : "nccl", repeat));
            auto t = summarize(samples);
            std::printf("%s,%zu,%d,%s,%.3f,%.3f,%.3f,%.4f,%d\n",
                        collectiveName(op), bytes, cfg.inPlace,
                        decision.cpuRouter ? routerName : "fallback", t.median,
                        t.minimum, t.maximum, decision.measuredSpeedup,
                        selected.routerInitialized());
          }
        }
    }
    if (!calibrated.empty()) saveRouterProfile(cfg.calibrate, calibrated);
    if (sampleOutput.is_open()) sampleOutput.close();
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
    std::puts("# correctness: PASS (every invocation, all ranks)");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
