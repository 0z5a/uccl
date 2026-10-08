#include "compressed_ack.h"
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
  // A prior request's success or failure must not retire a reused ring slot.
  for (int64_t id : {int64_t{0}, int64_t{4096}, int64_t{0xffffffff}}) {
    assert(compressed_ack_status(0, id) == CompressedWriteStatus::kPending);
    assert(compressed_ack_status(compressed_ack_value(id, true), id) ==
           CompressedWriteStatus::kReady);
    assert(compressed_ack_status(compressed_ack_value(id, false), id) ==
           CompressedWriteStatus::kFailed);
    for (bool success : {false, true}) {
      auto stale = compressed_ack_value(id, success);
      assert(compressed_ack_status(stale, id + 4096) ==
             CompressedWriteStatus::kPending);
    }
  }
  assert(compressed_ack_value(0, true) != 0);
  std::cout << "PASS: zero sentinel, exact request ACK, decode errors, "
               "stale success/failure after slot reuse, 32-bit boundary\n";
}
