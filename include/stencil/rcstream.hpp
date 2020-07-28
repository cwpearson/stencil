#pragma once

#include <iostream>

#include <cuda_runtime.h>

#include "stencil/cuda_runtime.hpp"

class RcStream {
public:
  enum class Priority {
    DEFAULT, // priority value of 0
    HIGH,    // max priority allowable for this stream
  };

private:
  size_t *count_;
  int dev_;
  cudaStream_t stream_;

  /* release resources if count is zero
   */
  void maybe_release();

  /* decrement count
   */
  void decrement();

public:
  RcStream(int dev, Priority requestedPriority = Priority::DEFAULT);

  RcStream() : RcStream(0) {}

  ~RcStream() {
    decrement();
    maybe_release();
  }

  // copy ctor
  RcStream(const RcStream &other) : count_(other.count_), dev_(other.dev_), stream_(other.stream_) { ++(*count_); }
  // move ctor
  RcStream(RcStream &&other)
      : count_(std::move(other.count_)), dev_(std::move(other.dev_)), stream_(std::move(other.stream_)) {
    other.stream_ = 0;
  }
  // copy assignment
  RcStream &operator=(const RcStream &rhs) {
    decrement();
    maybe_release();
    count_ = rhs.count_;
    ++(*count_);
    dev_ = rhs.dev_;
    stream_ = rhs.stream_;
    return *this;
  }
  // move assignment
  RcStream &operator=(RcStream &&rhs) {
    decrement();
    maybe_release();
    count_ = std::move(rhs.count_);
    dev_ = std::move(rhs.dev_);
    stream_ = std::move(rhs.stream_);
    rhs.stream_ = 0;
    return *this;
  }

  operator cudaStream_t() const noexcept { return stream_; }

  int device() const noexcept { return dev_; }

  bool operator==(const RcStream &rhs) const noexcept { return stream_ == rhs.stream_; }
};
