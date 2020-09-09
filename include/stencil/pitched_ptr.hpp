#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define INLINE __forceinline__
#else
#define CUDA_CALLABLE_MEMBER
#define INLINE
#endif

/* a typed cudaPitchedPtr, convertible to/from a cudaPitchedPtr
 */
template <typename T> struct PitchedPtr {
  size_t pitch;
  T *ptr;
  size_t xsize;
  size_t ysize;

  // can only be called on PitchedPtr<void>
  //   template <class = typename std::enable_if<std::is_same<PitchedPtr<T>, PitchedPtr<void>>::value>::type>
  //   PitchedPtr(const cudaPitchedPtr &p) : ptr(p.ptr), xsize(p.xsize), ysize(p.ysize), pitch(p.pitch) {}

  PitchedPtr() : pitch(0), ptr(nullptr), xsize(0), ysize(0) {}
  PitchedPtr(size_t _pitch, T *_ptr, size_t _xsize, size_t _ysize)
      : pitch(_pitch), ptr(_ptr), xsize(_xsize), ysize(_ysize) {
    assert(xsize % sizeof(T) == 0);
  }

  // conversion to and from cudaPitchedPtr
  // make these explicit since it's basically a type-cast
  explicit PitchedPtr(const cudaPitchedPtr &p)
      : ptr(reinterpret_cast<T *>(p.ptr)), xsize(p.xsize), ysize(p.ysize), pitch(p.pitch) {}
  explicit operator cudaPitchedPtr() {
    cudaPitchedPtr p = {};
    p.ptr = reinterpret_cast<T *>(ptr);
    p.xsize = xsize;
    p.ysize = ysize;
    p.pitch = pitch;
    return p;
  }

  /* compare with another pointer
   */
  template <typename U> bool operator!=(const PitchedPtr<U> &rhs) const noexcept {
    return ((void *)ptr != (void *)rhs.ptr) || (xsize != rhs.xsize) || (ysize != rhs.ysize) || (pitch != rhs.pitch);
  }

  INLINE CUDA_CALLABLE_MEMBER T &at(size_t x, size_t y, size_t z) noexcept {
    // byte offset
    const size_t bo = z * ysize * pitch + y * pitch + x * sizeof(T);
    // printf("&[%lu,%lu,%lu]->%lu\n", x,y,z, bo);
    return *(T *)((char *)(ptr) + bo);
  }
  INLINE CUDA_CALLABLE_MEMBER const T &at(size_t x, size_t y, size_t z) const noexcept {
    // byte offset
    const size_t bo = z * ysize * pitch + y * pitch + x * sizeof(T);
    return *(T *)((char *)(ptr) + bo);
  }
};

#undef CUDA_CALLABLE_MEMBER
#undef INLINE