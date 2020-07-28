#pragma once

#include <cassert>

#include "stencil/dim3.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <typename T> class Accessor {
private:
  T *raw_;
  Dim3 origin_; // the 3D point in the space represented by offset 0
  Dim3 pitch_;  // pitch in elements, not bytes

public:
  Accessor(T *raw, 
  const Dim3 &origin, //<! [in] the 3D point that is offset 0
           const Dim3 &pitch //<! [in] pitch in elements of allocation
           )
      : raw_(raw), origin_(origin), pitch_(pitch) {}

  //<! access point p
  CUDA_CALLABLE_MEMBER __forceinline__ T &operator[](const Dim3 &p) noexcept {
    const Dim3 off = p - origin_;
#ifndef NDEBUG
    assert(off.x >= 0);
    assert(off.y >= 0);
    assert(off.z >= 0);
#endif
    return raw_[off.z * pitch_.y * pitch_.x + off.y * pitch_.x + off.x];
  }

  CUDA_CALLABLE_MEMBER __forceinline__ const T &operator[](const Dim3 &p) const noexcept {
    const Dim3 off = p - origin_;
    return raw_[off.z * pitch_.y * pitch_.x + off.y * pitch_.x + off.x];
  }

  const Dim3 &origin() const noexcept { return origin_; }
  const Dim3 &pitch() const noexcept { return pitch_; }
};

#undef CUDA_CALLABLE_MEMBER