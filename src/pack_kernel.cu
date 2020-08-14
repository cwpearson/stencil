#include "stencil/pack_kernel.cuh"

__global__ void pack_kernel(void *__restrict__ dst, const void *__restrict__ src, const Dim3 srcSize, const Dim3 srcPos,
                            const Dim3 srcExtent, const size_t elemSize) {
  grid_pack(dst, src, srcSize, srcPos, srcExtent, elemSize);
}
