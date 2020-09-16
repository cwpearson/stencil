#include "catch2/catch.hpp"

// #include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/pack_kernel.cuh"
#include "stencil/pitched_ptr.hpp"

TEMPLATE_TEST_CASE("pack", "[pack][template]", int, double) {
  std::cerr << "TEST: \"pack -*\"\n";

  Dim3 arrSz(3, 4, 5);

  // src->dst and src -> dst -> dst2
  PitchedPtr<TestType> src;
  PitchedPtr<TestType> dst2;

  // 3*4*5 array
  CUDA_RUNTIME(cudaSetDevice(0));
  INFO("alloc src");
  CUDA_RUNTIME(cudaMallocManaged(&src.ptr, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  INFO("alloc dst2");
  CUDA_RUNTIME(cudaMallocManaged(&dst2.ptr, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  src.xsize = arrSz.x * sizeof(TestType);
  src.pitch = arrSz.x * sizeof(TestType);
  src.ysize = arrSz.y;
  dst2.xsize = arrSz.x * sizeof(TestType);
  dst2.pitch = arrSz.x * sizeof(TestType);
  dst2.ysize = arrSz.y;

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

    y faces
    (y = 0)       (y = 1)
    x ->          x ->
  z  0  1  2    z  3  4  5
  | 12 13 14    | 15 16 17
  v 24 25 26    v 27 28 29
    36 37 38      39 40 41
    48 49 50      51 52 53


  */

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

  SECTION("pack z = 4") {
    std::cerr << "pack z = 4\n";
    TestType *dst = nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.x * arrSz.y));
    dim3 dimGrid(2,2,2);
    dim3 dimBlock(2,2,2);
    std::cerr << dimGrid << " " << dimBlock << "\n";
    pack_kernel<<<dimGrid, dimBlock>>>(dst, cudaPitchedPtr(src), Dim3(0, 0, arrSz.z - 1), Dim3(arrSz.x, arrSz.y, 1),
                                       sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(dst[0] == 48);
    REQUIRE(dst[1] == 49);
    REQUIRE(dst[11] == 59);

    SECTION("unpack") {
      CUDA_RUNTIME(cudaMemset(dst2.ptr, 0, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
      unpack_kernel<<<dimGrid, dimBlock>>>(cudaPitchedPtr(dst2), dst, Dim3(0, 0, arrSz.z - 1),
                                           Dim3(arrSz.x, arrSz.y, 1), sizeof(TestType));
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(dst2.ptr[48] == 48);
      REQUIRE(dst2.ptr[59] == 59);
    }

    CUDA_RUNTIME(cudaFree(dst));
  }

  SECTION("pack x = 0") {
    TestType *dst = nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.y * arrSz.z));
    dim3 dimGrid(2, 2, 2);
    dim3 dimBlock(2, 2, 2);
    pack_kernel<<<dimGrid, dimBlock>>>(dst, cudaPitchedPtr(src), Dim3(0, 0, 0), Dim3(1, arrSz.y, arrSz.z),
                                       sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(dst[0] == 0);
    REQUIRE(dst[1] == 3);
    REQUIRE(dst[11] == 33);
    REQUIRE(dst[19] == 57);
    CUDA_RUNTIME(cudaFree(dst));
  }

  SECTION("pack y = 1") {
    TestType *dst = nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.y * arrSz.z));
    dim3 dimGrid(2, 2, 2);
    dim3 dimBlock(2, 2, 2);
    pack_kernel<<<dimGrid, dimBlock>>>(dst, cudaPitchedPtr(src), Dim3(0, 1, 0), Dim3(arrSz.x, 1, arrSz.z),
                                       sizeof(TestType));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(dst[0] == 3);
    REQUIRE(dst[1] == 4);
    REQUIRE(dst[11] == 41);
    REQUIRE(dst[14] == 53);
    CUDA_RUNTIME(cudaFree(dst));
  }

  CUDA_RUNTIME(cudaFree(src.ptr));
  CUDA_RUNTIME(cudaFree(dst2.ptr));
  CUDA_RUNTIME(cudaDeviceSynchronize());
}

TEST_CASE("real cases", "[cuda]") {
  SECTION("30x40x50, radius 4, +x face") {
    size_t radius = 4;
    Dim3 arrSz(30, 40, 50);
    Dim3 rawSz(38, 48, 58);
    size_t elemSize = 4;

    char *buf;
    cudaPitchedPtr dst;

    CUDA_RUNTIME(cudaSetDevice(0));
    CUDA_RUNTIME(cudaMallocManaged(&dst.ptr, elemSize * rawSz.x * rawSz.y * rawSz.z));
    dst.pitch = elemSize * rawSz.x;
    dst.xsize = elemSize * rawSz.x;
    dst.ysize = rawSz.y;
    CUDA_RUNTIME(cudaMallocManaged(&buf, elemSize * radius * arrSz.y * arrSz.z));

    dim3 dimGrid(20, 20, 20);
    dim3 dimBlock(32, 4, 4);

    Dim3 haloPos(34, 4, 4);
    Dim3 haloExtent(4, 40, 50);

    unpack_kernel<<<dimGrid, dimBlock>>>(dst, buf, haloPos, haloExtent, elemSize);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    CUDA_RUNTIME(cudaFree(buf));
    CUDA_RUNTIME(cudaFree(dst.ptr));
  }
}
