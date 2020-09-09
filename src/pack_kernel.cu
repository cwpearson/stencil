#include "stencil/pack_kernel.cuh"

__device__ void grid_pack(void *__restrict__ dst, const cudaPitchedPtr src,
                          const Dim3 srcPos,    // logical offset into the 3D region, in elements
                          const Dim3 srcExtent, // logical extent of the 3D region to pack, in elements
                          const size_t elemSize // size of the element in bytes
) {

  // dst.pitch: width of allocation in bytes
  // dst.ptr: raw data
  // dst.xsize: logical width of allocation in bytes
  // dst.ysize: logical height of allocation in bytes

  const char *__restrict__ sp = static_cast<char *>(src.ptr);

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  assert(srcExtent.x >= 0);
  assert(srcExtent.y >= 0);
  assert(srcExtent.z >= 0);
  assert(src.pitch > 0);

  for (unsigned int zo = tz; zo < srcExtent.z; zo += blockDim.z * gridDim.z) {
    unsigned int zi = zo + srcPos.z;
    for (unsigned int yo = ty; yo < srcExtent.y; yo += blockDim.y * gridDim.y) {
      unsigned int yi = yo + srcPos.y;
      for (unsigned int xo = tx; xo < srcExtent.x; xo += blockDim.x * gridDim.x) {
        unsigned int xi = xo + srcPos.x;

        // logical offset of packed output
        const size_t oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
        // printf("[xo, yo, zo]->oi = [%u, %u, %u]->%lu\n", xo, yo, zo, oi);
        // byte offset of input
        const size_t bi = zi * src.ysize * src.pitch + yi * src.pitch + xi * elemSize;
        // printf("[xi, yi, zi]->bi = [%u, %u, %u]->%lu\n", xi, yi, zi, bi);
        if (4 == elemSize) {
          uint32_t v = *reinterpret_cast<const uint32_t *>(sp + bi);
          reinterpret_cast<uint32_t *>(dst)[oi] = v;
        } else if (8 == elemSize) {
          uint64_t v = *reinterpret_cast<const uint64_t *>(sp + bi);
          reinterpret_cast<uint64_t *>(dst)[oi] = v;
        } else {
          char *pDst = reinterpret_cast<char *>(dst);
          memcpy(&pDst[oi * elemSize], sp + bi, elemSize);
        }
      }
    }
  }
}

__global__ void pack_kernel(void *dst, const cudaPitchedPtr src, const Dim3 srcPos, const Dim3 srcExtent,
                            const size_t elemSize) {
  grid_pack(dst, src, srcPos, srcExtent, elemSize);
}

__device__ void grid_unpack(cudaPitchedPtr dst, const void *__restrict__ src, const Dim3 dstPos, const Dim3 dstExtent,
                            const size_t elemSize) {

  // dst.pitch: width of allocation in bytes
  // dst.ptr: raw data
  // dst.xsize: logical width of allocation in bytes
  // dst.ysize: logical height of allocation in bytes

  char *__restrict__ dp = static_cast<char *>(dst.ptr);

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  assert(dstExtent.z >= 0);
  assert(dstExtent.y >= 0);
  assert(dstExtent.x >= 0);
  assert(dst.pitch > 0);

  for (unsigned int zi = tz; zi < dstExtent.z; zi += blockDim.z * gridDim.z) {
    unsigned int zo = zi + dstPos.z;
    for (unsigned int yi = ty; yi < dstExtent.y; yi += blockDim.y * gridDim.y) {
      unsigned int yo = yi + dstPos.y;
      for (unsigned int xi = tx; xi < dstExtent.x; xi += blockDim.x * gridDim.x) {
        unsigned int xo = xi + dstPos.x;
        // logical offset of packed input
        unsigned int ii = zi * dstExtent.y * dstExtent.x + yi * dstExtent.x + xi;
        // byte offset of output
        unsigned int bo = zo * dst.ysize * dst.pitch + yo * dst.pitch + xo * elemSize;
        if (4 == elemSize) {
          const uint32_t *pSrc = reinterpret_cast<const uint32_t *>(src);
          *reinterpret_cast<uint32_t *>(dp + bo) = pSrc[ii];
        } else if (8 == elemSize) {
          const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
          *reinterpret_cast<uint64_t *>(dp + bo) = pSrc[ii];
        } else {
          const char *pSrc = reinterpret_cast<const char *>(src);
          memcpy(dp + bo, &pSrc[ii * elemSize], elemSize);
        }
      }
    }
  }
}

__global__ void unpack_kernel(cudaPitchedPtr dst, const void *src, const Dim3 dstPos, const Dim3 dstExtent,
                              const size_t elemSize) {
  grid_unpack(dst, src, dstPos, dstExtent, elemSize);
}