#include "stencil/copy.cuh"
#include "stencil/pack_kernel.cuh"

#if 0
/* replaced by dev_packer_pack_domain and dev_packer_unpack_domain, which compute the offsets into dst on the fly
*/

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
#endif

/*! \brief grid-collaborative 3d translation

   non-overlapping src and dst
 */
inline __device__ void translate_grid(cudaPitchedPtr dst, const Dim3 dstPos, const cudaPitchedPtr src,
                                      const Dim3 srcPos,
                                      const Dim3 extent, // the extent of the region to be copied
                                      const size_t elemSize) {

  char *__restrict__ cDst = reinterpret_cast<char *>(dst.ptr);
  const char *__restrict__ cSrc = reinterpret_cast<const char *>(src.ptr);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  const Dim3 dstStop = dstPos + extent;

  for (size_t z = tz; z < extent.z; z += blockDim.z * gridDim.z) {
    for (size_t y = ty; y < extent.y; y += blockDim.y * gridDim.y) {
      for (size_t x = tx; x < extent.x; x += blockDim.x * gridDim.x) {
        // input coordinates
        unsigned int zi = z + srcPos.z;
        unsigned int yi = y + srcPos.y;
        unsigned int xi = x + srcPos.x;
        // output coordinates
        unsigned int zo = z + dstPos.z;
        unsigned int yo = y + dstPos.y;
        unsigned int xo = x + dstPos.x;
        // linearized byte offset
        size_t lo = zo * dst.ysize * dst.pitch + yo * dst.pitch + xo * elemSize;
        size_t li = zi * src.ysize * src.pitch + yi * src.pitch + xi * elemSize;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //        yo, zo, oi);
        // TODO: specialize to elemSize?
        memcpy(cDst + lo, cSrc + li, elemSize);
      }
    }
  }
}

__global__ void translate(cudaPitchedPtr dst, const Dim3 dstPos, cudaPitchedPtr src, const Dim3 srcPos,
                          const Dim3 extent, // the extent of the region to be copied
                          const size_t elemSize) {

  translate_grid(dst, dstPos, src, srcPos, extent, elemSize);
}

__global__ void multi_translate(cudaPitchedPtr *dsts, const Dim3 dstPos, const cudaPitchedPtr *srcs, const Dim3 srcPos,
                                const Dim3 extent, // the extent of the region to be copied
                                const size_t *__restrict__ elemSizes, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    translate_grid(dsts[i], dstPos, srcs[i], srcPos, extent, elemSizes[i]);
  }
}
