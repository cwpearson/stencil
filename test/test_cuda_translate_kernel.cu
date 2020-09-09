#include "catch2/catch.hpp"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/pitched_ptr.hpp"

TEMPLATE_TEST_CASE("translate kernel", "[cuda]", int) {
  Dim3 arrSz(3, 4, 5);

  PitchedPtr<TestType> src;
  PitchedPtr<TestType> dst;

  // 3*4*5 array
  INFO("alloc src");
  CUDA_RUNTIME(cudaMallocManaged(&src.ptr, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  INFO("alloc dst");
  CUDA_RUNTIME(cudaMallocManaged(&dst.ptr, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));

  // add pitch info
  src.xsize = arrSz.x * sizeof(TestType);
  src.pitch = arrSz.x * sizeof(TestType);
  src.ysize = arrSz.y;
  dst.xsize = arrSz.x * sizeof(TestType);
  dst.pitch = arrSz.x * sizeof(TestType);
  dst.ysize = arrSz.y;

  INFO("set src");
  for (size_t zi = 0; zi < arrSz.z; ++zi) {
    for (size_t yi = 0; yi < arrSz.y; ++yi) {
      for (size_t xi = 0; xi < arrSz.x; ++xi) {
        src.ptr[zi * arrSz.y * arrSz.x + yi * arrSz.x + xi] = zi * arrSz.y * arrSz.x + yi * arrSz.x + xi;
      }
    }
  }
  INFO("dev sync");
  CUDA_RUNTIME(cudaDeviceSynchronize());

  dim3 dimGrid(2, 2, 2);
  dim3 dimBlock(2, 2, 2);

  SECTION("0,0,0 -> 2,3,4") {
    translate<<<dimGrid, dimBlock>>>(cudaPitchedPtr(dst), Dim3(0, 0, 0), cudaPitchedPtr(src), Dim3(0, 0, 0),
                                     Dim3(1, 1, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[0] == 0);
  }

  SECTION("0,0,0 -> 2,3,4") {
    translate<<<dimGrid, dimBlock>>>(cudaPitchedPtr(dst), Dim3(2, 3, 4), cudaPitchedPtr(src), Dim3(0, 0, 0),
                                     Dim3(1, 1, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[59] == 0);
  }

  SECTION("2,3,4 -> 1,1,1") {
    translate<<<dimGrid, dimBlock>>>(cudaPitchedPtr(dst), Dim3(1, 1, 1), cudaPitchedPtr(src), Dim3(2, 3, 4),
                                     Dim3(1, 1, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[1 * (4 * 3) + 1 * (3) + 1] == 59);
  }

  SECTION("1,2,3 [2x2x2] -> 1,1,1") {
    translate<<<dimGrid, dimBlock>>>(cudaPitchedPtr(dst), Dim3(1, 1, 1), cudaPitchedPtr(src), Dim3(1, 2, 3),
                                     Dim3(2, 2, 2), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());
#define _at(x, y, z) dst.ptr[z * (4 * 3) + y * (3) + x]
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

  CUDA_RUNTIME(cudaFree(src.ptr));
  CUDA_RUNTIME(cudaFree(dst.ptr));
}