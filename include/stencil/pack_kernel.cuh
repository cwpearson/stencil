#pragma once

#include "stencil/dim3.hpp"

inline int64_t nextPowerOfTwo(int64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  return x;
}

inline Dim3 make_block_dim(const Dim3 extent, int64_t threads) {
  Dim3 ret;
  ret.x = std::min(threads, nextPowerOfTwo(extent.x));
  threads /= ret.x;
  ret.y = std::min(threads, nextPowerOfTwo(extent.y));
  threads /= ret.y;
  ret.z = std::min(threads, nextPowerOfTwo(extent.z));
  assert(ret.x <= 1024);
  assert(ret.y <= 1024);
  assert(ret.z <= 1024);
  assert(ret.x * ret.y * ret.z <= 1024);
  return ret;
}

inline __device__ void grid_pack(void *__restrict__ dst,
                                 const void *__restrict__ src,
                                 const Dim3 srcSize, const Dim3 srcPos,
                                 const Dim3 srcExtent, const size_t elemSize) {

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t zo = tz; zo < srcExtent.z; zo += blockDim.z * gridDim.z) {
    size_t zi = zo + srcPos.z;
    for (size_t yo = ty; yo < srcExtent.y; yo += blockDim.y * gridDim.y) {
      size_t yi = yo + srcPos.y;
      for (size_t xo = tx; xo < srcExtent.x; xo += blockDim.x * gridDim.x) {

        size_t xi = xo + srcPos.x;
        size_t oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
        size_t ii = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;

        if (4 == elemSize) {
          uint32_t *pDst = reinterpret_cast<uint32_t *>(dst);
          const uint32_t *pSrc = reinterpret_cast<const uint32_t *>(src);
          pDst[oi] = pSrc[ii];
        } else if (8 == elemSize) {
          uint64_t *pDst = reinterpret_cast<uint64_t *>(dst);
          const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
          pDst[oi] = pSrc[ii];
        } else {
          char *pDst = reinterpret_cast<char *>(dst);
          const char *pSrc = reinterpret_cast<const char *>(src);
          memcpy(&pDst[oi * elemSize], &pSrc[ii * elemSize], elemSize);
        }
      }
    }
  }
}

static __global__ void pack_kernel(void *__restrict__ dst,
                                   const void *__restrict__ src,
                                   const Dim3 srcSize, const Dim3 srcPos,
                                   const Dim3 srcExtent,
                                   const size_t elemSize) {
  grid_pack(dst, src, srcSize, srcPos, srcExtent, elemSize);
}
