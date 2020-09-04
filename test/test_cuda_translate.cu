#include "catch2/catch.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/translate.cuh"

TEMPLATE_TEST_CASE("translate", "[cuda]", int) {
  Dim3 arrSz(3, 4, 5);

  TestType *src = nullptr;
  TestType *dst = nullptr;

  // 3*4*5 array
  INFO("alloc src");
  CUDA_RUNTIME(cudaMallocManaged(&src, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  INFO("alloc dst");
  CUDA_RUNTIME(cudaMallocManaged(&dst, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));

  INFO("set src");
  for (size_t zi = 0; zi < arrSz.z; ++zi) {
    for (size_t yi = 0; yi < arrSz.y; ++yi) {
      for (size_t xi = 0; xi < arrSz.x; ++xi) {
        src[zi * arrSz.y * arrSz.x + yi * arrSz.x + xi] = zi * arrSz.y * arrSz.x + yi * arrSz.x + xi;
      }
    }
  }
  INFO("blank dst");
  for (size_t zi = 0; zi < arrSz.z; ++zi) {
    for (size_t yi = 0; yi < arrSz.y; ++yi) {
      for (size_t xi = 0; xi < arrSz.x; ++xi) {
        dst[zi * arrSz.y * arrSz.x + yi * arrSz.x + xi] = -1;
      }
    }
  }
  INFO("dev sync");
  CUDA_RUNTIME(cudaDeviceSynchronize());

  Translate translate;

  SECTION("0,0,0 -> 0,0,0") {
    size_t elemSizes = sizeof(TestType);
    void **dsts = (void**) &dst;
    void **srcs = (void**) &src;

    Translate::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(0, 0, 0),
        .dstSize = arrSz,
        .srcs = srcs,
        .srcPos = Dim3(0, 0, 0),
        .srcSize = arrSz,
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translate.prepare(std::vector<Translate::Params>(1, ps));
    translate.async(0);

    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst[0] == 0);
  }

  SECTION("0,0,0 -> 2,3,4") {
    size_t elemSizes = sizeof(TestType);
    void **dsts = (void**) &dst;
    void **srcs = (void**) &src;

    Translate::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(2, 3, 4),
        .dstSize = arrSz,
        .srcs = srcs,
        .srcPos = Dim3(0, 0, 0),
        .srcSize = arrSz,
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translate.prepare(std::vector<Translate::Params>(1, ps));
    translate.async(0);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst[59] == 0);
  }

  SECTION("2,3,4 -> 1,1,1") {
    size_t elemSizes = sizeof(TestType);
    void **dsts = (void**) &dst;
    void **srcs = (void**) &src;

    Translate::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(1, 1, 1),
        .dstSize = arrSz,
        .srcs = srcs,
        .srcPos = Dim3(2, 3, 4),
        .srcSize = arrSz,
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translate.prepare(std::vector<Translate::Params>(1, ps));
    translate.async(0);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst[1 * (4 * 3) + 1 * (3) + 1] == 59);
  }
  SECTION("1,2,3 [2x2x2] -> 1,1,1") {
    size_t elemSizes = sizeof(TestType);
    void **dsts = (void**) &dst;
    void **srcs = (void**) &src;

    Translate::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(1, 1, 1),
        .dstSize = arrSz,
        .srcs = srcs,
        .srcPos = Dim3(1, 2, 3),
        .srcSize = arrSz,
        .extent = Dim3(2, 2, 2),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translate.prepare(std::vector<Translate::Params>(1, ps));
    translate.async(0);
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