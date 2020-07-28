#include "stencil/rcstream.hpp"

#include <cassert>

void RcStream::maybe_release() {
  if (stream_ != 0) {
    if (0 == *count_) {
      CUDA_RUNTIME(cudaSetDevice(dev_));
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
  }
}

void RcStream::decrement() {
  if (0 != stream_) {
    assert(*count_ > 0);
    --(*count_);
  }
}

RcStream::RcStream(int dev, Priority requestedPriority) : count_(new size_t), dev_(dev) {
  *count_ = 1;
  CUDA_RUNTIME(cudaSetDevice(dev_));

  int minPrio;
  int maxPrio;
  CUDA_RUNTIME(cudaDeviceGetStreamPriorityRange(&minPrio, &maxPrio));
  if (minPrio == maxPrio) {
    std::cerr << "WARN: stream priority not supported\n";
  }

  int priority;
  switch (requestedPriority) {
  case Priority::DEFAULT:
    priority = 0;
    break;
  case Priority::HIGH:
    priority = maxPrio;
    break;
  default:
    priority = 0;
    std::cerr << "unexpected priority\n";
    exit(EXIT_FAILURE);
  }
  CUDA_RUNTIME(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
}