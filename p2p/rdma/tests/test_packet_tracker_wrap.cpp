#include "seq_num.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

void test_wrap() {
  AtomicBitmapPacketTrackerMultiAck tracker(UINT32_MAX - 2);
  std::vector<uint32_t> ids;
  for (size_t i = 0; i < 8; ++i) {
    auto id = tracker.send_packet(64, 2);
    ids.push_back(id);
    assert(!tracker.is_acknowledged(id));
    assert(tracker.get_packet_size(id) == 64);
  }
  assert(tracker.get_inflight_count() == 8);
  assert(tracker.get_total_inflight_bytes() == 8 * 64);
  assert(tracker.get_unacknowledged_packets() == ids);
  tracker.update_packet_size(ids[4], 128);
  assert(tracker.get_total_inflight_bytes() == 9 * 64);
  for (auto id : ids) {
    tracker.acknowledge(id);
    assert(!tracker.is_acknowledged(id));
    tracker.acknowledge(id);
    assert(tracker.is_acknowledged(id));
  }
  assert(tracker.get_inflight_count() == 0);
  assert(tracker.get_total_inflight_bytes() == 0);
  assert(tracker.get_unacknowledged_packets().empty());
}

void test_two_phase_expected_count() {
  AtomicBitmapPacketTrackerMultiAck tracker(UINT32_MAX);
  auto id = tracker.send_packet(64);
  assert(tracker.update_expected_ack_count(id, UINT32_MAX));
  tracker.acknowledge(id);
  tracker.acknowledge(id);
  assert(!tracker.is_acknowledged(id));
  assert(tracker.update_expected_ack_count(id, 3));
  assert(!tracker.is_acknowledged(id));
  tracker.acknowledge(id);
  assert(tracker.is_acknowledged(id));
  auto zero = tracker.send_packet(128);
  assert(zero == 0 && !tracker.is_acknowledged(zero));
  tracker.acknowledge(zero);
  assert(tracker.is_acknowledged(zero));
}

void test_concurrent_transitions() {
  AtomicBitmapPacketTrackerMultiAck tracker(UINT32_MAX - 127);
  for (size_t round = 0; round < 512; ++round) {
    std::vector<uint32_t> ids;
    for (size_t i = 0; i < 256; ++i) ids.push_back(tracker.send_packet(64, 2));
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < 16; ++worker) {
      threads.emplace_back([&, worker] {
        for (size_t i = worker % 8; i < ids.size(); i += 8) {
          if (worker < 8)
            tracker.acknowledge(ids[i]);
          else
            tracker.update_expected_ack_count(ids[i], 1);
          std::this_thread::yield();
        }
      });
    }
    for (auto& thread : threads) thread.join();
    assert(std::all_of(ids.begin(), ids.end(), [&](uint32_t id) {
      return tracker.is_acknowledged(id);
    }));
    assert(tracker.get_inflight_count() == 0);
    assert(tracker.get_unacknowledged_packets().empty());
  }
}

int main() {
  test_wrap();
  test_two_phase_expected_count();
  test_concurrent_transitions();
  std::cout << "{\"wrap\":\"PASS\",\"two_phase_counts\":\"PASS\","
               "\"concurrent_packet_transitions\":131072}\n";
}
