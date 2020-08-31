#pragma once

#include "stencil/dim3.hpp"

__device__ void grid_unpack(void *__restrict__ dst, const Dim3 dstSize, const Dim3 dstPos, const Dim3 dstExtent,
                            const void *__restrict__ src, const size_t elemSize);

__global__ void unpack(void *__restrict__ dst, const Dim3 dstSize, const size_t dstPitch, const Dim3 dstPos,
                       const Dim3 dstExtent, const void *__restrict__ src, const size_t elemSize);