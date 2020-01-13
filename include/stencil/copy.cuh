#pragma once

#include "dim3.hpp"

// pitch calculations
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c

static __global__ void pack(void *dst, const void *src, const Dim3 srcSize,
                            const size_t srcPitch, const Dim3 srcPos,
                            const Dim3 srcExtent, const size_t elemSize) {
  char *cDst = reinterpret_cast<char *>(dst);
  const char *cSrc = reinterpret_cast<const char *>(src);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t zo = tz; zo < srcExtent.z; zo += blockDim.z * gridDim.z) {
    for (size_t yo = ty; yo < srcExtent.y; yo += blockDim.y * gridDim.y) {
      for (size_t xo = tx; xo < srcExtent.x; xo += blockDim.x * gridDim.x) {
        size_t zi = zo + srcPos.z;
        size_t yi = yo + srcPos.y;
        size_t xi = xo + srcPos.x;
        size_t oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
        size_t ii = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //       yo, zo, oi);
        memcpy(&cDst[oi * elemSize], &cSrc[ii * elemSize], elemSize);
      }
    }
  }
}

static __global__ void unpack(void *dst, const Dim3 dstSize,
                              const size_t dstPitch, const Dim3 dstPos,
                              const Dim3 dstExtent, const void *src,
                              const size_t elemSize) {

  char *cDst = reinterpret_cast<char *>(dst);
  const char *cSrc = reinterpret_cast<const char *>(src);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t zi = tz; zi < dstExtent.z; zi += blockDim.z * gridDim.z) {
    for (size_t yi = ty; yi < dstExtent.y; yi += blockDim.y * gridDim.y) {
      for (size_t xi = tx; xi < dstExtent.x; xi += blockDim.x * gridDim.x) {
        size_t zo = zi + dstPos.z;
        size_t yo = yi + dstPos.y;
        size_t xo = xi + dstPos.x;
        size_t oi = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        size_t ii = zi * dstExtent.y * dstExtent.x + yi * dstExtent.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //        yo, zo, oi);
        memcpy(&cDst[oi * elemSize], &cSrc[ii * elemSize], elemSize);
      }
    }
  }
}
