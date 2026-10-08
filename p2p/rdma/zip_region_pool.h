#pragma once
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

// One fixed budget shared by all peers; leases return only after their last
// owner (including QPs and pending decoders) has released them.
class ZipRegionPool {
  struct State {
    std::mutex mutex;
    std::vector<size_t> free;
  };

 public:
  struct Lease {
    size_t const index;
    size_t const offset;
    size_t const size;
    ~Lease() {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->free.push_back(index);
    }
    Lease(Lease const&) = delete;
    Lease& operator=(Lease const&) = delete;

   private:
    friend class ZipRegionPool;
    Lease(std::shared_ptr<State> owner, size_t id, size_t bytes)
        : index(id), offset(id * bytes), size(bytes), state(std::move(owner)) {}
    std::shared_ptr<State> state;
  };

  ZipRegionPool(size_t bytes, size_t peers, size_t alignment)
      : state_(std::make_shared<State>()),
        region_bytes_(peers && alignment ? bytes / peers / alignment * alignment
                                         : 0) {
    if (!region_bytes_)
      throw std::invalid_argument("Zip region budget too small");
    state_->free.reserve(peers);
    for (size_t i = peers; i > 0; --i) state_->free.push_back(i - 1);
  }

  std::shared_ptr<Lease> acquire() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->free.empty()) return nullptr;
    size_t id = state_->free.back();
    state_->free.pop_back();
    return std::shared_ptr<Lease>(new Lease(state_, id, region_bytes_));
  }

 private:
  std::shared_ptr<State> state_;
  size_t region_bytes_;
};
