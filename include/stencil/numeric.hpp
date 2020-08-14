#pragma once

#include <cstdint>
#include <vector>

inline int64_t nextPowerOfTwo(int64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  return x;
}

/* return the prime factors of n, sorted largest to smallest
*/
std::vector<int64_t> prime_factors(int64_t n);

inline int64_t div_ceil(int64_t n, int64_t d) { return (n + d - 1) / d; }