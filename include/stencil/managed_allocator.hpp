#pragma once

#include <limits>

#include "cuda_runtime.hpp"

template <class T> class ManagedAllocator {
private:
  int gpu_;

public:
  // type definitions
  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  // rebind allocator to type U
  template <class U> struct rebind { typedef ManagedAllocator<U> other; };

  // return address of values
  pointer address(reference value) const { return &value; }
  const_pointer address(const_reference value) const { return &value; }

  /* constructors and destructor
   * - nothing to do because the allocator has no state
   */
  ManagedAllocator(int gpu) noexcept : gpu_(gpu) {}
  ManagedAllocator() noexcept: ManagedAllocator(0) {}
  ManagedAllocator(const ManagedAllocator &other) noexcept : gpu_(other.gpu_) {}
  template <class U> ManagedAllocator(const ManagedAllocator<U> &other) noexcept : gpu_(other.gpu_) {}
  ~ManagedAllocator() noexcept {}

  // return maximum number of elements that can be allocated
  size_type max_size() const throw() {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

  // allocate but don't initialize num elements of type T
  pointer allocate(size_type num, const void * = 0) {
    pointer ret = nullptr;
    CUDA_RUNTIME(cudaSetDevice(gpu_));
    CUDA_RUNTIME(cudaMallocManaged(&ret, num * sizeof(T)));
    return ret;
  }

  // deallocate storage p of deleted elements
  void deallocate(pointer p, size_type num) {
    CUDA_RUNTIME(cudaFree(p));
  }
};

// return that all specializations of this allocator are interchangeable
template <class T1, class T2>
bool operator==(const ManagedAllocator<T1> &a, const ManagedAllocator<T2> &b) noexcept {
    return a.gpu_ == b.gpu_;
}
template <class T1, class T2>
bool operator!=(const ManagedAllocator<T1> &a, const ManagedAllocator<T2> &b) noexcept {
  return a.gpu_ != b.gpu_;
}
