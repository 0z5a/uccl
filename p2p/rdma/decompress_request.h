#pragma once

enum class DecompressStatus { kPending, kSuccess, kFailed };

class DecompressRequest {
 public:
  virtual ~DecompressRequest() = default;
  virtual DecompressStatus poll() = 0;
  virtual DecompressStatus wait() = 0;
};
