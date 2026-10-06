#include "staged_copy.h"
#include <type_traits>

using namespace uccl::local;
static_assert(std::is_standard_layout_v<Ticket> && sizeof(Ticket) == 16);
static_assert(std::is_standard_layout_v<CopyStats> && sizeof(CopyStats) == 128);

// C ABI for the qualification harness; production Endpoint integration is separate.
extern "C" {
StagedCopy* uccl_local_create(int source, int target, size_t maximum,
                              unsigned slots, size_t chunk, int zip) {
  return new StagedCopy(source, target, maximum, slots, chunk,
                        zip ? CopyMode::Zip : CopyMode::Raw);
}
void uccl_local_destroy(StagedCopy* copy) { delete copy; }
Ticket uccl_local_submit(StagedCopy* copy, uintptr_t source, uintptr_t target,
                         size_t bytes, size_t capacity, uint32_t dtype,
                         uintptr_t producer) {
  return copy->submit(reinterpret_cast<void*>(source), reinterpret_cast<void*>(target),
                      bytes, capacity, static_cast<DataType>(dtype),
                      reinterpret_cast<cudaStream_t>(producer));
}
int uccl_local_wait(StagedCopy* copy, Ticket ticket, uintptr_t consumer, CopyStats* stats) {
  return copy->wait(ticket, reinterpret_cast<cudaStream_t>(consumer), stats);
}
int uccl_local_release(StagedCopy* copy, Ticket ticket, uintptr_t consumer) {
  return copy->release(ticket, reinterpret_cast<cudaStream_t>(consumer));
}
}
