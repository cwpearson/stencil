#pragma once

#include <ostream>

#include "stencil/dim3.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Rect3 {
public:
  Rect3() {}
  Rect3(const Dim3 &_lo, const Dim3 &_hi) : lo(_lo), hi(_hi) {}

  Dim3 lo;
  Dim3 hi;

  Dim3 extent() const noexcept { return hi - lo; }
};

inline std::ostream &operator<<(std::ostream &os, const Rect3 &e) {
  os << e.lo << "..<" << e.hi;
  return os;
}
