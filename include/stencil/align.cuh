#pragma once

#include <cstdlib>

inline size_t __device__ __host__ next_align_of(size_t x, size_t a) {
  return (x + a - 1) & (-1 * int64_t(a));
}
