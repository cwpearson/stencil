#pragma once

#include <cstdio>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s@%d: CUDA Runtime Error(%d): %s\n",  file, line, int(result),
            cudaGetErrorString(result));
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);