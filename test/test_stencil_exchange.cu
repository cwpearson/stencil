#include "catch2/catch.hpp"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.cuh"
#include "stencil/stencil.cuh"

TEST_CASE("exchange") {

//Should Inialize MPI

//Let us try to do multi GPU stencil
DistributedDomain dd(10,10,10);

dd.set_radius(1);

auto a = dd.add_data<float>();
dd.realize();

}










/*


  Dim3 arrSz(3, 4, 5);
  size_t pitch = sizeof(TestType) * (arrSz.x * arrSz.y);

  // src->dst and src -> dst -> dst2
  TestType *src = nullptr;
  TestType *dst = nullptr;
  TestType *dst2 = nullptr;

  // 3*4*5 array
  INFO("alloc src");
  CUDA_RUNTIME(
      cudaMallocManaged(&src, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  INFO("alloc dst2");
  CUDA_RUNTIME(
      cudaMallocManaged(&dst2, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));

  
  //   z faces
  //   (z = 0)   (z = 1)   (z = 4)
  //   x ->      x ->      x ->
  // y 0  1  2   12 13 14  48 49 50
  // | 3  4  5   15 16 17  51 52 53
  // v 6  7  8   18 19 20  54 55 56
  //   9 10 11   21 22 23  57 58 59

  //   x faces
  //   (x = 0)
  //   y ->
  // z  0  3  6  9
  // | 12 15 18 21
  // v 24 27 30 33
  //   36 39 42 45
  //   48 51 54 57

  //   y faces
  //   (y = 0)       (y = 1)
  //   x ->          x ->
  // z  0  1  2    z  3  4  5
  // | 12 13 14    | 15 16 17
  // v 24 25 26    v 27 28 29
  //   36 37 38      39 40 41
  //   48 49 50      51 52 53
  

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

  SECTION("pack z = 4") {
    CUDA_RUNTIME(cudaFree(dst));
    CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.x * arrSz.y));
    dim3 dimGrid(2, 2, 2);
    dim3 dimBlock(2, 2, 2);
    pack<<<dimGrid, dimBlock>>>(dst, src, arrSz, pitch, Dim3(0, 0, arrSz.z - 1),
                                Dim3(arrSz.x, arrSz.y, 1), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(dst[0] == 48);
    REQUIRE(dst[1] == 49);
    REQUIRE(dst[11] == 59);

    SECTION("unpack") {
      CUDA_RUNTIME(
          cudaMemset(dst2, 0, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
      unpack<<<dimGrid, dimBlock>>>(dst2, arrSz, pitch, Dim3(0, 0, arrSz.z - 1),
                                    Dim3(arrSz.x, arrSz.y, 1), dst,
                                    sizeof(TestType));
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(dst2[48] == 48);
      REQUIRE(dst2[59] == 59);
    }
  }

  SECTION("pack x = 0") {
    CUDA_RUNTIME(cudaFree(dst));
    CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.y * arrSz.z));
    dim3 dimGrid(2, 2, 2);
    dim3 dimBlock(2, 2, 2);
    pack<<<dimGrid, dimBlock>>>(dst, src, arrSz, pitch, Dim3(0, 0, 0),
                                Dim3(1, arrSz.y, arrSz.z), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(dst[0] == 0);
    REQUIRE(dst[1] == 3);
    REQUIRE(dst[11] == 33);
    REQUIRE(dst[19] == 57);
  }

  SECTION("pack y = 1") {
    CUDA_RUNTIME(cudaFree(dst));
    CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.y * arrSz.z));
    dim3 dimGrid(2, 2, 2);
    dim3 dimBlock(2, 2, 2);
    pack<<<dimGrid, dimBlock>>>(dst, src, arrSz, pitch, Dim3(0, 1, 0),
                                Dim3(arrSz.x, 1, arrSz.z), sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(dst[0] == 3);
    REQUIRE(dst[1] == 4);
    REQUIRE(dst[11] == 41);
    REQUIRE(dst[14] == 53);
  }

  */




}