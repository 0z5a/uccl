#include "async_decode_gate.h"
#include "dietgpu_decompress.h"
#include "util/debug.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {
void require(bool ok, char const* message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

struct Case {
  size_t bytes;
  unsigned callers;
  unsigned queue;
};

struct CodecFixture {
  gpuStream_t stream = nullptr;
  std::mutex mutex;
  std::shared_ptr<DecompressPool> pool;
  std::unique_ptr<dietgpu::StackDeviceMemory> scratch;

  CodecFixture() {
    CodecContextScope context(0);
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking));
    pool = make_decompress_pool(0, 4096);
    scratch = std::make_unique<dietgpu::StackDeviceMemory>(0, 1ull << 30);
  }

  ~CodecFixture() {
    CodecContextScope context(0);
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
    scratch.reset();
    GPU_RT_CHECK(gpuStreamDestroy(stream));
  }

  std::unique_ptr<DecompressRequest> decode(void const* archive, void* output,
                                            uint32_t elements) {
    return make_decompress_request(0, *scratch, stream, mutex, pool, archive,
                                   output, dietgpu::FloatType::kBFloat16,
                                   elements);
  }
};

void run_gated_slot(CodecFixture& codec, bool early_ack) {
  constexpr uint32_t elements = 512 * 1024;
  constexpr size_t bytes = elements * sizeof(uint16_t);
  auto capacity = dietgpu::getMaxFloatCompressedSize(
      dietgpu::FloatType::kBFloat16, elements);
  std::vector<uint16_t> a(elements, 0x3f80), b(elements, 0x4000);
  void* source = nullptr;
  void* archive_a = nullptr;
  void* archive_b = nullptr;
  void* output = nullptr;
  uint32_t* compressed_size = nullptr;
  uint32_t archive_sizes[2] = {};
  GPU_RT_CHECK(gpuMalloc(&source, bytes));
  GPU_RT_CHECK(gpuMalloc(&archive_a, capacity));
  GPU_RT_CHECK(gpuMalloc(&archive_b, capacity));
  GPU_RT_CHECK(gpuMalloc(&output, bytes));
  GPU_RT_CHECK(gpuMalloc(&compressed_size, sizeof(uint32_t)));
  for (unsigned i = 0; i < 2; ++i) {
    GPU_RT_CHECK(gpuMemcpy(source, i == 0 ? a.data() : b.data(), bytes,
                           gpuMemcpyHostToDevice));
    CodecContextScope context(0);
    dietgpu::FloatCompressConfig config;
    config.floatType = dietgpu::FloatType::kBFloat16;
    config.is16ByteAligned = true;
    void const* inputs[1] = {source};
    void* archives[1] = {i == 0 ? archive_a : archive_b};
    uint32_t sizes[1] = {elements};
    dietgpu::floatCompress(*codec.scratch, config, 1, inputs, sizes, archives,
                           compressed_size, codec.stream);
    GPU_RT_CHECK(gpuStreamSynchronize(codec.stream));
    GPU_RT_CHECK(gpuMemcpy(&archive_sizes[i], compressed_size, sizeof(uint32_t),
                           gpuMemcpyDeviceToHost));
  }
  // Warm the decoder before holding its stream, avoiding lazy module load.
  auto warmup = codec.decode(archive_a, output, elements);
  require(warmup && warmup->wait() == DecompressStatus::kSuccess,
          "gated decoder warmup failed");
  warmup.reset();
  DecodeGate* host_gate = nullptr;
  DecodeGate* device_gate = nullptr;
  GPU_RT_CHECK(
      gpuHostAlloc(&host_gate, sizeof(DecodeGate), gpuHostAllocMapped));
  std::memset(host_gate, 0, sizeof(DecodeGate));
  GPU_RT_CHECK(cudaHostGetDevicePointer(&device_gate, host_gate, 0));
  gpuStream_t copy_stream = nullptr;
  GPU_RT_CHECK(gpuStreamCreateWithFlags(&copy_stream, gpuStreamNonBlocking));
  {
    CodecContextScope context(0);
    launch_decode_gate(codec.stream, device_gate, uint64_t{1} << 35);
    GPU_RT_CHECK(gpuGetLastError());
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!__atomic_load_n(&host_gate->entered, __ATOMIC_ACQUIRE)) {
    require(std::chrono::steady_clock::now() < deadline, "gate did not start");
    std::this_thread::yield();
  }
  auto request = codec.decode(archive_a, output, elements);
  require(request && request->poll() == DecompressStatus::kPending,
          "decoder completed while its input was gated");
  if (early_ack) {
    // Model the current receiver's immediate ACK: a sender may legally
    // overwrite the compressed slot before the gated decoder consumes it.
    GPU_RT_CHECK(gpuMemcpyAsync(archive_a, archive_b, archive_sizes[1],
                                gpuMemcpyDeviceToDevice, copy_stream));
    GPU_RT_CHECK(gpuStreamSynchronize(copy_stream));
  }
  __atomic_store_n(&host_gate->release, 1u, __ATOMIC_RELEASE);
  require(request->wait() == DecompressStatus::kSuccess,
          "gated valid archive failed to decode");
  require(!__atomic_load_n(&host_gate->expired, __ATOMIC_ACQUIRE),
          "gate expired before the host released it");
  std::vector<uint16_t> observed(elements);
  GPU_RT_CHECK(
      gpuMemcpy(observed.data(), output, bytes, gpuMemcpyDeviceToHost));
  require(observed == (early_ack ? b : a), "slot reuse outcome differed");
  request.reset();
  GPU_RT_CHECK(gpuStreamDestroy(copy_stream));
  GPU_RT_CHECK(gpuFreeHost(host_gate));
  GPU_RT_CHECK(gpuFree(compressed_size));
  GPU_RT_CHECK(gpuFree(source));
  GPU_RT_CHECK(gpuFree(archive_a));
  GPU_RT_CHECK(gpuFree(archive_b));
  GPU_RT_CHECK(gpuFree(output));
  std::cout << "{\"test\":\"gated_slot\",\"early_ack\":"
            << (early_ack ? "true" : "false")
            << ",\"bytes_match_original\":" << (early_ack ? "false" : "true")
            << "}\n";
}

void run_case(CodecFixture& codec, Case const& c) {
  std::vector<uint16_t> input(c.bytes / sizeof(uint16_t));
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint16_t>(0x3f00 | ((i * 13) & 0x7f));
  void* source = nullptr;
  void* archive = nullptr;
  uint32_t* compressed_size = nullptr;
  auto elements = static_cast<uint32_t>(input.size());
  size_t archive_capacity = dietgpu::getMaxFloatCompressedSize(
      dietgpu::FloatType::kBFloat16, elements);
  GPU_RT_CHECK(gpuMalloc(&source, c.bytes));
  GPU_RT_CHECK(gpuMalloc(&archive, archive_capacity));
  GPU_RT_CHECK(gpuMalloc(&compressed_size, sizeof(uint32_t)));
  GPU_RT_CHECK(gpuMemcpy(source, input.data(), c.bytes, gpuMemcpyHostToDevice));
  {
    CodecContextScope context(0);
    dietgpu::FloatCompressConfig config;
    config.floatType = dietgpu::FloatType::kBFloat16;
    config.is16ByteAligned = true;
    void const* inputs[1] = {source};
    void* archives[1] = {archive};
    uint32_t sizes[1] = {elements};
    dietgpu::floatCompress(*codec.scratch, config, 1, inputs, sizes, archives,
                           compressed_size, codec.stream);
    GPU_RT_CHECK(gpuStreamSynchronize(codec.stream));
  }
  GPU_RT_CHECK(gpuFree(source));
  GPU_RT_CHECK(gpuFree(compressed_size));

  size_t count = size_t{c.callers} * c.queue;
  std::vector<void*> outputs(count);
  std::vector<std::unique_ptr<DecompressRequest>> pending(count);
  for (auto& output : outputs) GPU_RT_CHECK(gpuMalloc(&output, c.bytes));
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> callers;
  auto begin = std::chrono::steady_clock::now();
  for (unsigned caller = 0; caller < c.callers; ++caller) {
    callers.emplace_back([&, caller] {
      GPU_RT_CHECK(gpuSetDevice(0));
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (unsigned q = 0; q < c.queue; ++q) {
        size_t index = size_t{caller} * c.queue + q;
        pending[index] = codec.decode(archive, outputs[index], elements);
        require(bool(pending[index]), "valid decode rejected");
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != c.callers)
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (auto& caller : callers) caller.join();
  for (auto& request : pending)
    require(request->wait() == DecompressStatus::kSuccess,
            "valid decode reported failure");
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();

  std::vector<uint16_t> observed(input.size());
  for (auto output : outputs) {
    GPU_RT_CHECK(
        gpuMemcpy(observed.data(), output, c.bytes, gpuMemcpyDeviceToHost));
    require(std::memcmp(input.data(), observed.data(), c.bytes) == 0,
            "decoded bytes differ");
  }
  // The immediate host status does not report device capacity failures.
  auto failed = codec.decode(archive, outputs[0], elements - 1);
  require(failed && failed->wait() == DecompressStatus::kFailed,
          "capacity error was reported as success");
  require(!codec.decode(archive, nullptr, elements),
          "invalid output was submitted");
  failed.reset();
  pending.clear();
  for (auto output : outputs) GPU_RT_CHECK(gpuFree(output));
  GPU_RT_CHECK(gpuFree(archive));
  std::cout << "{\"test\":\"async_decode\",\"bytes\":" << c.bytes
            << ",\"callers\":" << c.callers << ",\"queue\":" << c.queue
            << ",\"requests\":" << count << ",\"seconds\":" << seconds
            << ",\"exact_bytes\":true,\"capacity_error\":true}\n";
}
}  // namespace

int main(int argc, char** argv) {
  GPU_RT_CHECK(gpuSetDevice(0));
  GPU_RT_CHECK(gpuFree(nullptr));
  CUcontext caller_context = nullptr;
  UCCL_CHECK(cuCtxGetCurrent(&caller_context) == CUDA_SUCCESS);
  CodecFixture codec;
  CUcontext after_init = nullptr;
  UCCL_CHECK(cuCtxGetCurrent(&after_init) == CUDA_SUCCESS);
  require(after_init == caller_context, "constructor changed caller context");
  std::vector<Case> cases;
  if (argc > 1 && std::strcmp(argv[1], "--smoke") == 0) {
    cases = {{16ull << 20, 1, 2}, {16ull << 20, 8, 2}};
  } else {
    for (size_t bytes : {16ull << 20, 64ull << 20, 256ull << 20})
      for (unsigned queue : {1u, 2u, 4u, 8u, 16u})
        cases.push_back({bytes, 1, queue});
    cases.push_back({16ull << 20, 64, 2});
    cases.push_back({16ull << 20, 128, 4});
    cases.push_back({16ull << 20, 128, 16});
  }
  for (auto const& c : cases) run_case(codec, c);
  run_gated_slot(codec, true);
  run_gated_slot(codec, false);
  CUcontext after_tests = nullptr;
  UCCL_CHECK(cuCtxGetCurrent(&after_tests) == CUDA_SUCCESS);
  require(after_tests == caller_context, "decode changed caller context");
}
