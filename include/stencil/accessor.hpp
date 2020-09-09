#pragma once

#include <cassert>

#include "stencil/dim3.hpp"
#include "stencil/pitched_ptr.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <typename T> class Accessor {
private:
  PitchedPtr<T> ptr_;
  Dim3 origin_; // the 3D point in the space represented by offset 0

public:
  Accessor(const PitchedPtr<T> &ptr,
           const Dim3 &origin //<! [in] the 3D point that is offset 0
           )
      : ptr_(ptr), origin_(origin) {}

  // pre PitchedPtr constructor, for compatibility with tests. Do not use
  Accessor(T *raw, const Dim3 &origin, const Dim3 &elemPitch // pitch in elements, not bytes
           )
      : ptr_(elemPitch.x * sizeof(T), raw, elemPitch.x * sizeof(T), elemPitch.y), origin_(origin) {}

  //<! access point p
  CUDA_CALLABLE_MEMBER __forceinline__ T &operator[](const Dim3 &p) noexcept {
    const Dim3 off = p - origin_;
    assert(off.x >= 0);
    assert(off.y >= 0);
    assert(off.z >= 0);
    return ptr_.at(off.x, off.y, off.z);
  }

  CUDA_CALLABLE_MEMBER __forceinline__ const T &operator[](const Dim3 &p) const noexcept {
    const Dim3 off = p - origin_;
    assert(off.x >= 0);
    assert(off.y >= 0);
    assert(off.z >= 0);
    return ptr_.at(off.x, off.y, off.z);
  }

  const Dim3 &origin() const noexcept { return origin_; }
  const PitchedPtr<T> &ptr() const noexcept { return ptr_; }
};

#undef CUDA_CALLABLE_MEMBER