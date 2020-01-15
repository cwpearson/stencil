#include "catch2/catch.hpp"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"

TEMPLATE_TEST_CASE("translate", "[cuda]", int) {
  Dim3 arrSz(3, 4, 5);

  TestType *src = nullptr;
  TestType *dst = nullptr;

  // 3*4*5 array
  INFO("alloc src");
  CUDA_RUNTIME(
      cudaMallocManaged(&src, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  INFO("alloc dst");
  CUDA_RUNTIME(
      cudaMallocManaged(&dst, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));

  INFO("set src");
  for (size_t zi = 0; zi < arrSz.z; ++zi) {
    for (size_t yi = 0; yi < arrSz.y; ++yi) {
      for (size_t xi = 0; xi < arrSz.x; ++xi) {
        src[zi * arrSz.y * arrSz.x + yi * arrSz.x + xi] =
            zi * arrSz.y * arrSz.x + yi * arrSz.x + xi;
      }
    }
  }
  INFO("dev sync");
  CUDA_RUNTIME(cudaDeviceSynchronize());

  dim3 dimGrid(2, 2, 2);
  dim3 dimBlock(2, 2, 2);

  SECTION("0,0,0 -> 2,3,4") {
    translate<<<dimGrid, dimBlock>>>(dst, Dim3(0, 0, 0), src, Dim3(0, 0, 0),
                                     arrSz, Dim3(1, 1, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst[0] == 0);
  }

  SECTION("0,0,0 -> 2,3,4") {
    translate<<<dimGrid, dimBlock>>>(dst, Dim3(2, 3, 4), src, Dim3(0, 0, 0),
                                     arrSz, Dim3(1, 1, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst[59] == 0);
  }

  SECTION("2,3,4 -> 1,1,1") {
    translate<<<dimGrid, dimBlock>>>(dst, Dim3(1, 1, 1), src, Dim3(2, 3, 4),
                                     arrSz, Dim3(1, 1, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst[1 * (4 * 3) + 1 * (3) + 1] == 59);
  }

  SECTION("1,2,3 [2x2x2] -> 1,1,1") {
    translate<<<dimGrid, dimBlock>>>(dst, Dim3(1, 1, 1), src, Dim3(1, 2, 3),
                                     arrSz, Dim3(2, 2, 2), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
#define _at(x, y, z) dst[z * (4 * 3) + y * (3) + x]
#define _val(x, y, z) (z * (4 * 3) + y * (3) + x)
    REQUIRE(_at(1, 1, 1) == _val(1, 2, 3));
    REQUIRE(_at(1, 1, 2) == _val(1, 2, 4));
    REQUIRE(_at(1, 2, 1) == _val(1, 3, 3));
    REQUIRE(_at(1, 2, 2) == _val(1, 3, 4));
    REQUIRE(_at(2, 1, 1) == _val(2, 2, 3));
    REQUIRE(_at(2, 1, 2) == _val(2, 2, 4));
    REQUIRE(_at(2, 2, 1) == _val(2, 3, 3));
    REQUIRE(_at(2, 2, 2) == _val(2, 3, 4));
#undef _at
#undef _val
  }

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(dst));
}