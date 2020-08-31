#pragma once

#include "dim3.hpp"

// pitch calculations
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c

/*! same as calling pack(&dst[offsets[i]]...srcs[i]...elemSizes[i])
 */
__global__ void multi_pack(void *__restrict__ dst,                      // dst buffer
                           const size_t *__restrict__ offsets,          // offsets into dst
                           void *__restrict__ *__restrict__ const srcs, // n src pointers
                           const Dim3 srcSize, const Dim3 srcPos, const Dim3 srcExtent,
                           const size_t *__restrict__ elemSizes, // n elem sizes
                           const size_t n);

/*! same as calling unpack(dsts[i]...srcs[offsets[i]]...elemSizes[i])
 */
__global__ void multi_unpack(void **__restrict__ dsts, const Dim3 dstSize, const Dim3 dstPos, const Dim3 dstExtent,
                             const void *__restrict__ const src, const size_t *__restrict__ offsets,
                             const size_t *__restrict__ elemSizes, const size_t n);

// take the 3D region src[srcPos...srcPos+extent] and translate it to the 3D
// region dst[dstPos...dstPos+extent]
// srcSize and dstSize may not be the same
__global__ void translate(void *__restrict__ dst, const Dim3 dstPos, const Dim3 dstSize, const void *__restrict__ src,
                          const Dim3 srcPos, const Dim3 srcSize,
                          const Dim3 extent, // the extent of the region to be copied
                          const size_t elemSize);

__global__ void multi_translate(void *__restrict__ *__restrict__ dsts, const Dim3 dstPos, const Dim3 dstSize,
                                void *__restrict__ *__restrict__ const srcs, const Dim3 srcPos, const Dim3 srcSize,
                                const Dim3 extent, // the extent of the region to be copied
                                size_t *const __restrict__ elemSizes, const size_t n);
