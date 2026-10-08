#pragma once
#include "compressed_ack.h"
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <vector>

inline uint64_t next_compressed_token() {
  static std::atomic<uint64_t> next{1};
  uint64_t token = next.fetch_add(1, std::memory_order_relaxed);
  if (!token || token >= kAckErrBit) std::abort();
  return token;
}

class ZipAckSlots {
 public:
  struct Ticket {
    uint32_t slot;
    uint64_t token;
  };

  explicit ZipAckSlots(size_t depth) : used_(depth, false) {}

  std::optional<Ticket> acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t n = 0; n < used_.size(); ++n) {
      size_t slot = (next_ + n) % used_.size();
      if (used_[slot]) continue;
      used_[slot] = true;
      next_ = (slot + 1) % used_.size();
      return Ticket{static_cast<uint32_t>(slot), next_compressed_token()};
    }
    return std::nullopt;
  }

  void release(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    used_[slot] = false;
  }

 private:
  std::mutex mutex_;
  std::vector<bool> used_;
  size_t next_ = 0;
};
