#pragma once

#include <iostream>

#include <cuda_runtime.h>

#include "stencil/cuda_runtime.hpp"

class RcStream {
private:
  size_t *count_;
  int dev_;
  cudaStream_t stream_;

  void try_release() {
    if (stream_ != 0) {
      if (0 == *count_) {
        CUDA_RUNTIME(cudaSetDevice(dev_));
        CUDA_RUNTIME(cudaStreamDestroy(stream_));
      }
    }
  }

  void decrement() {
    if (0 != stream_) {
      assert(*count_ > 0);
      --(*count_);
    }
  }

public:
  RcStream(int dev) : count_(new size_t), dev_(dev) {
    *count_ = 1;
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  RcStream() : RcStream(0) {}

  ~RcStream() {
    decrement();
    try_release();
  }

  // copy ctor
  RcStream(const RcStream &other)
      : count_(other.count_), dev_(other.dev_), stream_(other.stream_) {
    ++(*count_);
  }
  // move ctor
  RcStream(RcStream &&other)
      : count_(std::move(other.count_)), dev_(std::move(other.dev_)),
        stream_(std::move(other.stream_)) {
    other.stream_ = 0;
  }
  // copy assignment
  RcStream &operator=(const RcStream &rhs) {
    decrement();
    try_release();
    count_ = rhs.count_;
    ++(*count_);
    dev_ = rhs.dev_;
    stream_ = rhs.stream_;
    return *this;
  }
  // move assignment
  RcStream &operator=(RcStream &&rhs) {
    decrement();
    try_release();
    count_ = std::move(rhs.count_);
    dev_ = std::move(rhs.dev_);
    stream_ = std::move(rhs.stream_);
    rhs.stream_ = 0;
    return *this;
  }

  operator cudaStream_t() const noexcept { return stream_; }

  int device() const noexcept { return dev_; }

  bool operator==(const RcStream &rhs) const noexcept {
    return stream_ == rhs.stream_;
  }
};
