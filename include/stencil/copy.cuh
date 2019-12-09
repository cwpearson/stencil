#pragma once

#include "dim3.cuh"

// pitch calculations
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c

__global__ void pack(void *dst, const void *src, const Dim3 srcSize,
                     const size_t srcPitch, const Dim3 srcPos,
                     const Dim3 srcExtent, const size_t elemSize) {
  char *cDst = reinterpret_cast<char *>(dst);
  const char *cSrc = reinterpret_cast<const char *>(src);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = tz; z < srcExtent.z; z += blockDim.z * gridDim.z) {
    for (size_t y = ty; y < srcExtent.y; y += blockDim.y * gridDim.y) {
      for (size_t x = tx; x < srcExtent.x; x += blockDim.x * gridDim.x) {
        size_t zo = z;
        size_t yo = y;
        size_t xo = x;
        size_t zi = z + srcPos.z;
        size_t yi = y + srcPos.y;
        size_t xi = x + srcPos.x;
        size_t oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
        size_t ii = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo, yo, zo, oi);
        memcpy(&cDst[oi * elemSize], &cSrc[ii * elemSize], elemSize);
      }
    }
  }
}

__global__ void unpack(void *dst, const Dim3 dstSize, const size_t dstPitch,
                       const Dim3 dstPos, const Dim3 dstExtent, const void *src,
                       const size_t elemSize) {

  char *cDst = reinterpret_cast<char *>(dst);
  const char *cSrc = reinterpret_cast<const char *>(src);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = tz; z < dstExtent.z; z += blockDim.z * gridDim.z) {
    for (size_t y = ty; y < dstExtent.y; y += blockDim.y * gridDim.y) {
      for (size_t x = tx; x < dstExtent.x; x += blockDim.x * gridDim.x) {
        size_t zo = z + dstPos.z;
        size_t yo = y + dstPos.y;
        size_t xo = x + dstPos.x;
        size_t zi = z;
        size_t yi = y;
        size_t xi = x;
        size_t oi = zo * dstExtent.y * dstExtent.x + yo * dstExtent.x + xo;
        size_t ii = zi * dstSize.y * dstSize.x + yi * dstSize.x + xi;
        memcpy(&cDst[oi * elemSize], &cSrc[ii * elemSize], elemSize);
      }
    }
  }
}
