#pragma once
#include "common.h"
#include "zip_region_pool.h"

struct ZipRegion {
  std::shared_ptr<RegMemBlock> backing;
  std::shared_ptr<ZipRegionPool::Lease> lease;
  std::shared_ptr<RegMemBlock> memory;

  ZipRegion(std::shared_ptr<RegMemBlock> buffer,
            std::shared_ptr<ZipRegionPool::Lease> region)
      : backing(std::move(buffer)),
        lease(std::move(region)),
        memory(std::make_shared<RegMemBlock>(
            static_cast<char*>(backing->addr) + lease->offset, lease->size,
            backing->mr_array, backing->type)) {}
};
