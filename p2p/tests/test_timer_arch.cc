#include "util/timer.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

int main() {
  auto start = std::chrono::steady_clock::now();
  uint64_t ticks = uccl::rdtsc();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  double elapsed = uccl::to_nsec(uccl::rdtsc() - ticks, uccl::freq_ghz);
  double reference = std::chrono::duration<double, std::nano>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  assert(elapsed / reference > 0.9 && elapsed / reference < 1.1);
  std::cout << "PASS: counter conversion agrees with elapsed time\n";
}
