#pragma once

#include "dim3.cuh"

// pitch calculations https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c

template<typename T>
__global__ void pack(T *dst, const T* src, const Dim3 srcSize, const size_t srcPitch, const Dim3 srcPos, const Dim3 srcExtent) {

    const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

    // if (tx == 0 && ty == 0&& tz == 0) {
    //     printf("size: %lu %lu %lu\n", srcSize.x, srcSize.y, srcSize.z);
    //     printf("pos:  %lu %lu %lu\n", srcPos.x, srcPos.y, srcPos.z);
    //     printf("ext:  %lu %lu %lu\n", srcExtent.x, srcExtent.y, srcExtent.z);
    // }

    for (size_t z = tz; z < srcExtent.z; z += blockDim.z * gridDim.z) {
        for (size_t y = ty; y < srcExtent.y; y += blockDim.y * gridDim.y) {
            for (size_t x = tx; x < srcExtent.x; x += blockDim.x * gridDim.x) {
                size_t zo = z;
                size_t yo = y;
                size_t xo = x;
                size_t zi = z + srcPos.z;
                size_t yi = y + srcPos.y;
                size_t xi = x + srcPos.x;
                size_t oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
                size_t ii = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
                // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii, xo, yo, zo, oi);
                dst[oi] = src[ii];
            }
        }
    }

}

