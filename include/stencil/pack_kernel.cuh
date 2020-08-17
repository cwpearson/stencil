#pragma once

#include "stencil/dim3.hpp"

/* used in a variety of places, so we'll leave it as inlineable for now
 */
__device__ void grid_pack(void *__restrict__ dst, const void *__restrict__ src, const Dim3 srcSize, const Dim3 srcPos,
                          const Dim3 srcExtent, const size_t elemSize);

/* TODO:
   for some reason, moving this into its own object files causes weird runtime errors, either invalid PTX or invalid
   configuration arguments
*/
__global__ void pack_kernel(void *__restrict__ dst, const void *__restrict__ src, const Dim3 srcSize, const Dim3 srcPos,
                            const Dim3 srcExtent, const size_t elemSize);
