#include "stencil/copy.cuh"
#include "stencil/pack_kernel.cuh"
#include "stencil/unpack_kernel.cuh"

__global__ void multi_pack(void *__restrict__ dst,                      // dst buffer
                           const size_t *__restrict__ offsets,          // offsets into dst
                           void *__restrict__ *__restrict__ const srcs, // n src pointers
                           const Dim3 srcSize, const Dim3 srcPos, const Dim3 srcExtent,
                           const size_t *__restrict__ elemSizes, // n elem sizes
                           const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    void *dstp = &(static_cast<char *>(dst)[offsets[i]]);
    grid_pack(dstp, srcs[i], srcSize, srcPos, srcExtent, elemSizes[i]);
  }
}

__global__ void multi_unpack(void **__restrict__ dsts, const Dim3 dstSize, const Dim3 dstPos, const Dim3 dstExtent,
                             const void *__restrict__ const src, const size_t *__restrict__ offsets,
                             const size_t *__restrict__ elemSizes, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const void *srcp = &(static_cast<const char *>(src)[offsets[i]]);
    grid_unpack(dsts[i], dstSize, dstPos, dstExtent, srcp, elemSizes[i]);
  }
}

inline __device__ void translate_grid(void *__restrict__ dst, const Dim3 dstPos, const Dim3 dstSize,
                                      const void *__restrict__ src, const Dim3 srcPos, const Dim3 srcSize,
                                      const Dim3 extent, // the extent of the region to be copied
                                      const size_t elemSize) {

  char *cDst = reinterpret_cast<char *>(dst);
  const char *cSrc = reinterpret_cast<const char *>(src);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  const Dim3 dstStop = dstPos + extent;

  for (size_t z = tz; z < extent.z; z += blockDim.z * gridDim.z) {
    for (size_t y = ty; y < extent.y; y += blockDim.y * gridDim.y) {
      for (size_t x = tx; x < extent.x; x += blockDim.x * gridDim.x) {
        // input coorindates
        size_t zi = z + srcPos.z;
        size_t yi = y + srcPos.y;
        size_t xi = x + srcPos.x;
        // output coordinates
        size_t zo = z + dstPos.z;
        size_t yo = y + dstPos.y;
        size_t xo = x + dstPos.x;
        // linearized
        size_t lo = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        size_t li = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //        yo, zo, oi);
        memcpy(&cDst[lo * elemSize], &cSrc[li * elemSize], elemSize);
      }
    }
  }
}

__global__ void translate(void *__restrict__ dst, const Dim3 dstPos, const Dim3 dstSize, const void *__restrict__ src,
                          const Dim3 srcPos, const Dim3 srcSize,
                          const Dim3 extent, // the extent of the region to be copied
                          const size_t elemSize) {

  translate_grid(dst, dstPos, dstSize, src, srcPos, srcSize, extent, elemSize);
}

__global__ void multi_translate(void *__restrict__ *__restrict__ dsts, const Dim3 dstPos, const Dim3 dstSize,
                                void *__restrict__ *__restrict__ const srcs, const Dim3 srcPos, const Dim3 srcSize,
                                const Dim3 extent, // the extent of the region to be copied
                                size_t *const __restrict__ elemSizes, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    translate_grid(dsts[i], dstPos, dstSize, srcs[i], srcPos, srcSize, extent, elemSizes[i]);
  }
}
