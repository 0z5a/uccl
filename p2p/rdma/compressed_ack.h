#pragma once
#include <cstdint>

inline constexpr uint64_t kAckErrBit = uint64_t{1} << 63;

enum class CompressedWriteStatus { kPending, kReady, kFailed };

inline uint64_t compressed_token_value(uint64_t token, bool success) {
  return token | (success ? 0 : kAckErrBit);
}

inline CompressedWriteStatus compressed_token_status(uint64_t value,
                                                     uint64_t token) {
  if (!token || (value & ~kAckErrBit) != token)
    return CompressedWriteStatus::kPending;
  return value & kAckErrBit ? CompressedWriteStatus::kFailed
                            : CompressedWriteStatus::kReady;
}

inline uint64_t compressed_ack_value(int64_t wr_id, bool success) {
  return (static_cast<uint64_t>(wr_id) + 1) | (success ? 0 : kAckErrBit);
}

inline CompressedWriteStatus compressed_ack_status(uint64_t value,
                                                   int64_t wr_id) {
  if ((value & ~kAckErrBit) != compressed_ack_value(wr_id, true))
    return CompressedWriteStatus::kPending;
  return value & kAckErrBit ? CompressedWriteStatus::kFailed
                            : CompressedWriteStatus::kReady;
}
