#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line, int(result), cudaGetErrorString(result));
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

enum class CudaErrorsFatal { NO, YES };