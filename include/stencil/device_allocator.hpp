#pragma once

#include <limits>

#include "cuda_runtime.hpp"

template <class T> class DeviceAllocator {
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
  template <class U> struct rebind { typedef DeviceAllocator<U> other; };

  // return address of values
  pointer address(reference value) const { return &value; }
  const_pointer address(const_reference value) const { return &value; }

  /* constructors and destructor
   * - nothing to do because the allocator has no state
   */
  DeviceAllocator(int gpu) noexcept : gpu_(gpu) {}
  DeviceAllocator() noexcept: DeviceAllocator(0) {}
  DeviceAllocator(const DeviceAllocator &other) noexcept : gpu_(other.gpu_) {}
  template <class U> DeviceAllocator(const DeviceAllocator<U> &other) noexcept : gpu_(other.gpu_) {}
  ~DeviceAllocator() noexcept {}

  // return maximum number of elements that can be allocated
  size_type max_size() const throw() {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

  // allocate but don't initialize num elements of type T
  pointer allocate(size_type num, const void * = 0) {
    pointer ret = nullptr;
    CUDA_RUNTIME(cudaSetDevice(gpu_));
    CUDA_RUNTIME(cudaMalloc(&ret, num * sizeof(T)));
    return ret;
  }

  // deallocate storage p of deleted elements
  void deallocate(pointer p, size_type num) {
    CUDA_RUNTIME(cudaFree(p));
  }
};

// return that all specializations of this allocator are interchangeable
template <class T1, class T2>
bool operator==(const DeviceAllocator<T1> &a, const DeviceAllocator<T2> &b) noexcept {
    return a.gpu_ == b.gpu_;
}
template <class T1, class T2>
bool operator!=(const DeviceAllocator<T1> &a, const DeviceAllocator<T2> &b) noexcept {
  return a.gpu_ != b.gpu_;
}
