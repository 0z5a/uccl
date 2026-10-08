#include "zip_ack_slots.h"
#include "zip_region_pool.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <unordered_set>

void test_fixed_budget() {
  constexpr size_t budget = 1024 * 1024 + 37;
  ZipRegionPool pool(budget, 4, 256);
  std::array<std::shared_ptr<ZipRegionPool::Lease>, 4> regions;
  for (size_t i = 0; i < regions.size(); ++i) {
    regions[i] = pool.acquire();
    assert(regions[i] && regions[i]->offset % 256 == 0);
    assert(regions[i]->offset + regions[i]->size <= budget);
    for (size_t j = 0; j < i; ++j)
      assert(regions[j]->offset + regions[j]->size <= regions[i]->offset);
  }
  assert(!pool.acquire());
  auto consumer = regions[1];
  size_t retired_offset = consumer->offset;
  regions[1].reset();
  assert(!pool.acquire());
  consumer.reset();
  auto replacement = pool.acquire();
  assert(replacement && replacement->offset == retired_offset);
  assert(!pool.acquire());

  std::shared_ptr<ZipRegionPool::Lease> survivor;
  {
    ZipRegionPool temporary(256, 1, 256);
    survivor = temporary.acquire();
  }
  assert(survivor->size == 256);
  survivor.reset();
}

void test_parallel_regions() {
  constexpr size_t bytes = 1024 * 1024;
  ZipRegionPool pool(bytes, 4, 256);
  std::vector<uint8_t> memory(bytes);
  std::array<std::atomic<unsigned>, 4> owners{};
  for (auto& owner : owners) owner.store(0);
  std::vector<std::thread> threads;
  for (unsigned worker = 1; worker <= 16; ++worker) {
    threads.emplace_back([&, worker] {
      for (size_t iteration = 0; iteration < 32; ++iteration) {
        auto lease = pool.acquire();
        while (!lease) {
          std::this_thread::yield();
          lease = pool.acquire();
        }
        assert(owners[lease->index].exchange(worker) == 0);
        auto first = memory.begin() + lease->offset;
        auto last = first + lease->size;
        auto value = static_cast<uint8_t>(worker + iteration * 16);
        std::fill(first, last, value);
        std::this_thread::yield();
        assert(std::all_of(first, last, [=](uint8_t x) { return x == value; }));
        assert(owners[lease->index].exchange(0) == worker);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (auto& owner : owners) assert(owner.load() == 0);
}

void test_ack_backpressure_and_generations() {
  ZipAckSlots slots(2);
  auto first = slots.acquire();
  auto second = slots.acquire();
  assert(first && second && first->slot != second->slot);
  assert(first->token != second->token && !slots.acquire());
  assert(compressed_token_status(0, first->token) ==
         CompressedWriteStatus::kPending);
  slots.release(first->slot);
  auto next = slots.acquire();
  assert(next && next->slot == first->slot && next->token != first->token);
  for (bool success : {false, true}) {
    uint64_t stale = compressed_token_value(first->token, success);
    assert(compressed_token_status(stale, next->token) ==
           CompressedWriteStatus::kPending);
  }
  assert(compressed_token_status(compressed_token_value(next->token, true),
                                 next->token) == CompressedWriteStatus::kReady);
  assert(compressed_token_status(compressed_token_value(next->token, false),
                                 next->token) ==
         CompressedWriteStatus::kFailed);
  assert(compressed_token_status(kAckErrBit, 0) ==
         CompressedWriteStatus::kPending);
  ZipAckSlots reconnected(1);
  auto fresh = reconnected.acquire();
  assert(fresh && fresh->token != first->token && fresh->token != next->token);
  assert(compressed_token_status(compressed_token_value(first->token, true),
                                 fresh->token) ==
         CompressedWriteStatus::kPending);
}

void test_parallel_tokens() {
  std::array<std::vector<uint64_t>, 16> tokens;
  std::vector<std::thread> threads;
  for (auto& batch : tokens) {
    threads.emplace_back([&batch] {
      for (size_t i = 0; i < 4096; ++i)
        batch.push_back(next_compressed_token());
    });
  }
  for (auto& thread : threads) thread.join();
  std::unordered_set<uint64_t> unique;
  for (auto const& batch : tokens) {
    for (auto token : batch) {
      assert(token > 0 && token < kAckErrBit);
      assert(unique.insert(token).second);
    }
  }
  assert(unique.size() == 16 * 4096);
}

void test_parallel_tickets() {
  ZipAckSlots slots(4);
  std::array<std::atomic<uint64_t>, 4> owners{};
  std::array<std::atomic<uint64_t>, 4> previous{};
  for (auto& owner : owners) owner.store(0);
  for (auto& token : previous) token.store(0);
  std::vector<std::thread> threads;
  for (size_t worker = 0; worker < 16; ++worker) {
    threads.emplace_back([&] {
      for (size_t round = 0; round < 128; ++round) {
        auto ticket = slots.acquire();
        while (!ticket) {
          std::this_thread::yield();
          ticket = slots.acquire();
        }
        assert(owners[ticket->slot].exchange(ticket->token) == 0);
        auto stale = previous[ticket->slot].load();
        assert(compressed_token_status(stale, ticket->token) ==
               CompressedWriteStatus::kPending);
        std::this_thread::yield();
        assert(owners[ticket->slot].exchange(0) == ticket->token);
        previous[ticket->slot].store(
            compressed_token_value(ticket->token, true));
        slots.release(ticket->slot);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (auto& owner : owners) assert(owner.load() == 0);
}

int main() {
  test_fixed_budget();
  test_parallel_regions();
  test_ack_backpressure_and_generations();
  test_parallel_tokens();
  test_parallel_tickets();
  std::cout
      << "{\"fixed_budget\":\"PASS\",\"parallel_regions\":\"PASS\","
         "\"ack_backpressure\":\"PASS\",\"stale_ack\":\"PASS\","
         "\"parallel_unique_tokens\":65536,\"parallel_ticket_reuse\":2048}\n";
}
