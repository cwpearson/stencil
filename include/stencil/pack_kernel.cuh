#pragma once

#include "stencil/dim3.hpp"

/* used in a variety of places, so we'll leave it as inlineable for now
 */
__device__ void grid_pack(void *__restrict__ dst, const cudaPitchedPtr src, const Dim3 srcPos, const Dim3 srcExtent,
                          const size_t elemSize);

__global__ void pack_kernel(void *__restrict__ dst, const cudaPitchedPtr src, const Dim3 srcPos, const Dim3 srcExtent,
                            const size_t elemSize);

__device__ void grid_unpack(cudaPitchedPtr dst, const void *__restrict__ src, const Dim3 dstPos, const Dim3 dstExtent,
                            const size_t elemSize);

__global__ void unpack_kernel(cudaPitchedPtr dst, const void *src, const Dim3 dstPos, const Dim3 dstExtent,
                              const size_t elemSize);