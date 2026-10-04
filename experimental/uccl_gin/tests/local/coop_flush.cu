#include "queue_fixture.hpp"
#include "thirdparty/nccl-ep/adapter/uccl_gin_net.cuh"
#include <algorithm>

using namespace local_test;

// This file is also compiled against the original adapter, with only its
// signal-load compilation prerequisite applied identically to both builds.
#ifndef LOCAL_EXPECT_SCALAR_ADAPTER
#define LOCAL_EXPECT_SCALAR_ADAPTER 0
#endif

__host__ __device__ uint32_t pattern(int producer, int generation, int word) {
  return 0x12340000u + producer * 17u + generation * 131u + word;
}

template <bool Adapter, typename Coop>
__global__ void exercise(uccl_gin::UCCLGinResources resources, int queues,
                         bool shared, int iterations, int words,
                         bool invalid_order, uint32_t* returned) {
  int const group = blockIdx.x;
  int const producer = group * blockDim.x + threadIdx.x;
  int const lane = threadIdx.x;
  if (!shared) resources.d2h_queues += group * queues;
  resources.num_queues = queues;
  auto* source =
      reinterpret_cast<uint32_t*>(resources.window_base) + producer * words;
  size_t const source_offset = producer * words * sizeof(uint32_t);
  size_t const destination_offset = resources.window_bytes / 2 + source_offset;
  uccl_gin::UCCLGin gin(resources);
  nccl_ep_adapter::UcclGinNet adapter(resources, lane);
  for (int generation = 1; generation <= iterations; ++generation) {
    for (int i = 0; i < words; ++i)
      source[i] = pattern(producer, generation, i);
    // Model the caller's released source data; delay different lanes before
    // publishing their puts to stress the entry rendezvous.
    __threadfence_system();
    __nanosleep((lane % 7) * 64);
    if (words > 0) {
      if constexpr (Adapter) {
        adapter.put(ncclTeam{}, 1, ncclWindow_t{}, destination_offset,
                    ncclWindow_t{}, source_offset, words * sizeof(uint32_t));
      } else {
        gin.put<ncclTeamTagRail>(
            reinterpret_cast<void*>(resources.window_base + destination_offset),
            source, words * sizeof(uint32_t), 1, lane);
      }
    }
    if constexpr (Adapter) {
      adapter.flush(Coop{}, invalid_order ? cuda::memory_order_relaxed
                                          : cuda::memory_order_acquire);
    } else {
      gin.flush(Coop{});
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(
        returned + producer, static_cast<uint32_t>(generation),
        mscclpp::memoryOrderRelease);
    // The host oracle must capture the generation above, never this overwrite.
    for (int i = 0; i < words; ++i) source[i] = 0xdead0000u + producer;
  }
}

int main(int argc, char** argv) {
  int const device = integer(argc, argv, "--device", 0);
  int const groups = integer(argc, argv, "--warps", 1);
  int const queues = integer(argc, argv, "--queues", 4);
  int const capacity = integer(argc, argv, "--capacity", 512);
  int const iterations = integer(argc, argv, "--iterations", 100);
  int const batch = integer(argc, argv, "--batch-size", 0);
  int const hidden = integer(argc, argv, "--hidden-size", 2048);
  int const rounds = integer(argc, argv, "--rounds", 3);
  int const delay_us = integer(argc, argv, "--delay-us", 0);
  bool const shared = integer(argc, argv, "--shared", 1) != 0;
  bool const reuse = integer(argc, argv, "--check-source-reuse", 1) != 0;
  bool const invalid_order = integer(argc, argv, "--invalid-order", 0) != 0;
  char const* group = option(argc, argv, "--group", "warp");
  char const* via = option(argc, argv, "--via", "adapter");
  require(std::strcmp(group, "warp") == 0 || std::strcmp(group, "thread") == 0,
          "group must be thread or warp");
  require(
      std::strcmp(via, "adapter") == 0 || std::strcmp(via, "standalone") == 0,
      "via must be adapter or standalone");
  bool const warp = std::strcmp(group, "warp") == 0;
  bool const adapter = std::strcmp(via, "adapter") == 0;
  require(groups > 0 && groups <= 128 && queues > 0 && queues <= 64 &&
              iterations > 0 && rounds > 0,
          "invalid workload size");
  int const width = warp ? 32 : 1;
  int const producers = groups * width;
  size_t const group_bytes = static_cast<size_t>(batch) * hidden * 2;
  require(batch == 0 || (group_bytes % width == 0 &&
                         group_bytes / width <= kTransferCmdMaxBytes),
          "BF16 batch shape must fit the lane payloads");
  int const bytes =
      integer(argc, argv, "--bytes", batch > 0 ? group_bytes / width : 64);
  require(batch == 0 || static_cast<size_t>(bytes) == group_bytes / width,
          "payload bytes disagree with batch shape");
  require(bytes % 4 == 0 && bytes <= static_cast<int>(kTransferCmdMaxBytes),
          "payload must be word aligned and fit one WRITE");
  require((shared ? producers : width) <= capacity,
          "producer participation exceeds per-FIFO capacity");
  require(!reuse || bytes > 0, "source-reuse oracle requires payloads");
  require(!LOCAL_EXPECT_SCALAR_ADAPTER || adapter,
          "baseline covers the adapter only");
  require(!invalid_order || adapter, "order belongs to adapter API");
  size_t const source_bytes = static_cast<size_t>(producers) * bytes;
  require(source_bytes * 2 <= UINT32_MAX,
          "payload window exceeds command offsets");
  QueueFixture fixture(device, shared ? queues : groups * queues, capacity);
  auto source = mscclpp::detail::gpuCallocUnique<uint32_t>(
      std::max(size_t{1}, source_bytes / 2));
  auto capture =
      mscclpp::detail::gpuCallocHostUnique<uint32_t>(std::max(1, bytes / 4));
  auto returned = mscclpp::detail::gpuCallocHostUnique<uint32_t>(producers);
  fixture.resources.window_base = reinterpret_cast<uint64_t>(source.get());
  fixture.resources.window_bytes = source_bytes * 2;
  fixture.resources.num_scaleout_ranks = 2;
  cudaStream_t copy_stream;
  cudaEvent_t copied;
  CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
  // Initialize the independent stream/event before any kernel waits on a FIFO.
  CUDA_CHECK(cudaEventRecord(copied, copy_stream));
  CUDA_CHECK(cudaEventSynchronize(copied));
  for (int round = 0; round < rounds; ++round) {
    std::fill(returned.get(), returned.get() + producers, 0u);
    std::vector<int> generations(producers);
    std::vector<uint64_t> writes_per_queue(fixture.queues.size());
    std::vector<uint64_t> quiet_per_queue(fixture.queues.size());
    uint64_t writes = 0, quiet = 0;
    fixture.start([&](size_t queue, TransferCmd const& cmd) {
      auto const type = get_base_cmd(cmd.cmd_type);
      if (type == CmdType::WRITE) {
        size_t const offset = static_cast<size_t>(cmd.req_lptr)
                              << kWriteAddrShiftNormal;
        require(bytes > 0 && cmd.bytes == static_cast<uint32_t>(bytes) &&
                    offset % bytes == 0 && cmd.dst_rank == 1 &&
                    cmd.req_rptr == source_bytes / 4 + cmd.req_lptr,
                "incorrect WRITE encoding");
        int const producer = offset / bytes;
        require(producer < producers, "source offset exceeds fixture");
        int const generation = ++generations[producer];
        if (reuse) {
          CUDA_CHECK(cudaMemcpyAsync(capture.get(), source.get() + offset / 4,
                                     bytes, cudaMemcpyDeviceToHost,
                                     copy_stream));
          CUDA_CHECK(cudaEventRecord(copied, copy_stream));
          CUDA_CHECK(cudaEventSynchronize(copied));
          for (int word = 0; word < bytes / 4; ++word) {
            if (capture.get()[word] != pattern(producer, generation, word)) {
              std::fprintf(
                  stderr,
                  "FAIL: device=%d queue=%zu producer=%d generation=%d word=%d "
                  "source reused before completion\n",
                  device, queue, producer, generation, word);
              std::exit(1);
            }
          }
        }
        ++writes;
        ++writes_per_queue[queue];
      } else {
        require(type == CmdType::QUIET, "unexpected command type");
        uint64_t const generation = ++quiet_per_queue[queue];
        if (!shared && !LOCAL_EXPECT_SCALAR_ADAPTER && bytes > 0) {
          int const local_queue = queue % queues;
          int const writers =
              local_queue < width ? 1 + (width - 1 - local_queue) / queues : 0;
          require(writes_per_queue[queue] >= generation * writers,
                  "group QUIET preceded a participant's prior WRITE");
        }
        if (!shared && !LOCAL_EXPECT_SCALAR_ADAPTER &&
            queue % queues == static_cast<size_t>(queues - 1)) {
          int const first = (queue / queues) * width;
          for (int lane = 0; lane < width; ++lane) {
            require(mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(
                        returned.get() + first + lane,
                        mscclpp::memoryOrderAcquire) < generation,
                    "participant returned before final QUIET acknowledgement");
          }
        }
        if (delay_us > 0 && queue % queues == static_cast<size_t>(queues - 1)) {
          std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }
        ++quiet;
      }
    });
    auto const start = std::chrono::steady_clock::now();
    if (adapter && warp) {
      exercise<true, ncclCoopWarp><<<groups, width, 0, fixture.stream>>>(
          fixture.resources, queues, shared, iterations, bytes / 4,
          invalid_order, returned.get());
    } else if (adapter) {
      exercise<true, ncclCoopThread><<<groups, width, 0, fixture.stream>>>(
          fixture.resources, queues, shared, iterations, bytes / 4,
          invalid_order, returned.get());
    } else if (warp) {
#if !LOCAL_EXPECT_SCALAR_ADAPTER
      exercise<false, ncclCoopWarp><<<groups, width, 0, fixture.stream>>>(
          fixture.resources, queues, shared, iterations, bytes / 4, false,
          returned.get());
#endif
    } else {
#if !LOCAL_EXPECT_SCALAR_ADAPTER
      exercise<false, ncclCoopThread><<<groups, width, 0, fixture.stream>>>(
          fixture.resources, queues, shared, iterations, bytes / 4, false,
          returned.get());
#endif
    }
    if (invalid_order) {
      auto const error = cudaStreamSynchronize(fixture.stream);
      std::fprintf(stderr, "invalid-order completion: %s (%d)\n",
                   cudaGetErrorString(error), error);
      require(error == cudaErrorIllegalInstruction ||
                  error == cudaErrorLaunchFailure,
              "unsupported order did not trap");
      std::printf("{\"test\":\"unsupported_order\",\"pass\":true}\n");
      std::fflush(stdout);
      std::_Exit(
          0);  // CUDA context is poisoned; finish only this negative test.
    }
    fixture.finish();
    double const ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    uint64_t const expected_quiet = static_cast<uint64_t>(groups) * iterations *
                                    queues *
                                    (LOCAL_EXPECT_SCALAR_ADAPTER ? width : 1);
    require(quiet == expected_quiet, "incorrect cooperative QUIET count");
    require(
        writes == static_cast<uint64_t>(bytes > 0 ? producers : 0) * iterations,
        "missing WRITE commands");
    for (int generation : generations) {
      require(generation == (bytes > 0 ? iterations : 0),
              "missing producer generation");
    }
    std::printf(
        "{\"test\":\"coop_flush\",\"device\":%d,\"via\":\"%s\","
        "\"group\":\"%s\",\"groups\":%d,\"queues\":%d,\"shared\":%s,"
        "\"iterations\":%d,\"bytes\":%d,\"batch_size\":%d,\"hidden_size\":%d,"
        "\"round\":%d,\"writes\":%llu,"
        "\"quiet\":%llu,\"source_reuse_checked\":%s,\"elapsed_ms\":%.3f,"
        "\"pass\":true}\n",
        device, via, group, groups, queues, shared ? "true" : "false",
        iterations, bytes, batch, hidden, round,
        static_cast<unsigned long long>(writes),
        static_cast<unsigned long long>(quiet), reuse ? "true" : "false", ms);
  }
  CUDA_CHECK(cudaEventDestroy(copied));
  CUDA_CHECK(cudaStreamDestroy(copy_stream));
}
