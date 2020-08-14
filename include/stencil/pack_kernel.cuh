#pragma once

#include "stencil/dim3.hpp"

/* used in a variety of places, so we'll leave it as inlineable for now
 */
inline __device__ void grid_pack(void *__restrict__ dst, const void *__restrict__ src, const Dim3 srcSize,
                                 const Dim3 srcPos, const Dim3 srcExtent, const size_t elemSize) {

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  assert(srcExtent.x >= 0);
  assert(srcExtent.y >= 0);
  assert(srcExtent.z >= 0);

  for (unsigned int zo = tz; zo < srcExtent.z; zo += blockDim.z * gridDim.z) {
    unsigned int zi = zo + srcPos.z;
    for (unsigned int yo = ty; yo < srcExtent.y; yo += blockDim.y * gridDim.y) {
      unsigned int yi = yo + srcPos.y;
      for (unsigned int xo = tx; xo < srcExtent.x; xo += blockDim.x * gridDim.x) {
        unsigned int xi = xo + srcPos.x;
        unsigned int oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
        unsigned int ii = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
        if (4 == elemSize) {
          float *pDst = reinterpret_cast<float *>(dst);
          const float *pSrc = reinterpret_cast<const float *>(src);
          float v = pSrc[ii];
          pDst[oi] = v;
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

/* TODO:
   for some reason, moving this into its own object files causes weird runtime errors, either invalid PTX or invalid
   configuration arguments
*/
static __global__ void pack_kernel(void *__restrict__ dst, const void *__restrict__ src, const Dim3 srcSize,
                                   const Dim3 srcPos, const Dim3 srcExtent, const size_t elemSize) {
  grid_pack(dst, src, srcSize, srcPos, srcExtent, elemSize);
}
