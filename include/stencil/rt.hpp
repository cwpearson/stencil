#pragma once

#include <cuda_runtime.h>

#include "stencil/timer.hpp"

namespace rt {

template <typename Fn, typename... Args> cudaError_t time(Fn fn, Args... args) {
  cudaError_t err;
  CR_TIC();
  err = fn(args...);
  CR_TOC();
  return err;
}

template <typename Fn, typename... Args>
void launch(Fn fn, const dim3 &grid, const dim3 &block, const int shmem, cudaStream_t stream, Args... args) {
  CR_TIC();
  fn<<<grid, block, shmem, stream>>>(args...);
  CR_TOC();
}

}; // namespace rt