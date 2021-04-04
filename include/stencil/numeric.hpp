#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <cstdlib>

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
template <typename T> std::vector<T> prime_factors(T n);

inline int64_t div_ceil(int64_t n, int64_t d) { return (n + d - 1) / d; }

template <typename T> T get_max_abs_error(const T *a, const T *b, const size_t n) {
  T maxAbsErr = std::numeric_limits<T>::lowest();
  for (size_t i = 0; i < n; ++i) {
    maxAbsErr = std::max(maxAbsErr, std::abs(a[i] - b[i]));
  }
  return maxAbsErr;
}
