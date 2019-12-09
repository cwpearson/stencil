#include "catch2/catch.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.cuh"
#include "stencil/copy.cuh"

TEMPLATE_TEST_CASE( "pack", "[factorial][template]", int ) {
    REQUIRE( 1 == 1 );

    Dim3 arrSz(3,4,5);
    size_t pitch = sizeof(TestType) * (arrSz.x * arrSz.y);

    TestType *src = nullptr;
    TestType *dst = nullptr;

    // 3*4*5 array
    INFO("alloc dst");
    CUDA_RUNTIME(cudaMallocManaged(&src, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));

/*
  z faces
  (z = 0)   (z = 1)   (z = 4)
  x ->      x ->      x ->
y 0  1  2   12 13 14  48 49 50
| 3  4  5   15 16 17  51 52 53
v 6  7  8   18 19 20  54 55 56
  9 10 11   21 22 23  57 58 59

  x faces
  (x = 0)   
  y ->      
z  0  3  6  9  
| 12 15 18 21   
v 24 27 30 33
  36 39 42 45
  48 51 54 57
    

*/

    INFO("set dst");
    for (size_t zi = 0; zi < arrSz.z; ++zi) {
        for (size_t yi = 0; yi < arrSz.y; ++yi) {
            for (size_t xi = 0; xi < arrSz.x; ++xi) {
                src[zi * arrSz.y *arrSz.x + yi * arrSz.x + xi] = zi * arrSz.y *arrSz.x + yi * arrSz.x + xi;
            }
        }   
    }
    INFO("dev sync");
    CUDA_RUNTIME(cudaDeviceSynchronize());

    #if 1
    SECTION("copy z = 4") {
        CUDA_RUNTIME(cudaFree(dst));
        CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.x * arrSz.y));
        dim3 dimGrid(2,2,2);
        dim3 dimBlock(2,2,2);
        pack<<<dimGrid, dimBlock>>>(dst, src, arrSz, pitch, Dim3(0,0,arrSz.z-1), Dim3(arrSz.x, arrSz.y, 1));
        CUDA_RUNTIME(cudaDeviceSynchronize());

        REQUIRE(dst[0] == 48);
        REQUIRE(dst[1] == 49);
        REQUIRE(dst[11] == 59);
    }
    #endif

    #if 1
    SECTION("copy x = 0") {
        CUDA_RUNTIME(cudaFree(dst));
        CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.y * arrSz.z));
        dim3 dimGrid(2,2,2);
        dim3 dimBlock(2,2,2);
        pack<<<dimGrid, dimBlock>>>(dst, src, arrSz, pitch, Dim3(0,0,0), Dim3(1, arrSz.y, arrSz.z));
        CUDA_RUNTIME(cudaDeviceSynchronize());

        REQUIRE(dst[0] == 0);
        REQUIRE(dst[1] == 3);
        REQUIRE(dst[11] == 33);
        REQUIRE(dst[19] == 57);
    }
    #endif

    
}