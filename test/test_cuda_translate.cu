#include "catch2/catch.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/pitched_ptr.hpp"
#include "stencil/translator.cuh"

template <typename TestType> void all_sections(Translator *translator) {
  Dim3 arrSz(3, 4, 5);

  PitchedPtr<TestType> src;
  PitchedPtr<TestType> dst;

  // 3*4*5 array
  INFO("alloc src");
  CUDA_RUNTIME(cudaMallocManaged(&src.ptr, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
  INFO("alloc dst");
  CUDA_RUNTIME(cudaMallocManaged(&dst.ptr, sizeof(TestType) * arrSz.x * arrSz.y * arrSz.z));
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
  INFO("blank dst");
  for (size_t zi = 0; zi < arrSz.z; ++zi) {
    for (size_t yi = 0; yi < arrSz.y; ++yi) {
      for (size_t xi = 0; xi < arrSz.x; ++xi) {
        dst.ptr[zi * arrSz.y * arrSz.x + yi * arrSz.x + xi] = -1;
      }
    }
  }
  INFO("dev sync");
  CUDA_RUNTIME(cudaDeviceSynchronize());

  SECTION("0,0,0 -> 0,0,0") {
    size_t elemSizes = sizeof(TestType);
    cudaPitchedPtr cudaSrc = cudaPitchedPtr(src);
    cudaPitchedPtr *srcs = &cudaSrc;
    cudaPitchedPtr cudaDst = cudaPitchedPtr(dst);
    cudaPitchedPtr *dsts = &cudaDst;

    Translator::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(0, 0, 0),
        .srcs = srcs,
        .srcPos = Dim3(0, 0, 0),
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translator->prepare(std::vector<Translator::Params>(1, ps));
    translator->async(0);

    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[0] == 0);
  }

  SECTION("0,0,0 -> 0,0,0 multi") {
    size_t elemSizes = sizeof(TestType);
    cudaPitchedPtr cudaSrc = cudaPitchedPtr(src);
    cudaPitchedPtr *srcs = &cudaSrc;
    cudaPitchedPtr cudaDst = cudaPitchedPtr(dst);
    cudaPitchedPtr *dsts = &cudaDst;

    Translator::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(0, 0, 0),
        .srcs = srcs,
        .srcPos = Dim3(0, 0, 0),
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translator->prepare(std::vector<Translator::Params>(5, ps));
    translator->async(0);

    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[0] == 0);
  }

  SECTION("0,0,0 -> 2,3,4") {
    size_t elemSizes = sizeof(TestType);
    cudaPitchedPtr cudaSrc = cudaPitchedPtr(src);
    cudaPitchedPtr *srcs = &cudaSrc;
    cudaPitchedPtr cudaDst = cudaPitchedPtr(dst);
    cudaPitchedPtr *dsts = &cudaDst;

    Translator::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(2, 3, 4),
        .srcs = srcs,
        .srcPos = Dim3(0, 0, 0),
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translator->prepare(std::vector<Translator::Params>(1, ps));
    translator->async(0);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[59] == 0);
  }

  SECTION("2,3,4 -> 1,1,1") {
    size_t elemSizes = sizeof(TestType);
    cudaPitchedPtr cudaSrc = cudaPitchedPtr(src);
    cudaPitchedPtr *srcs = &cudaSrc;
    cudaPitchedPtr cudaDst = cudaPitchedPtr(dst);
    cudaPitchedPtr *dsts = &cudaDst;

    Translator::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(1, 1, 1),
        .srcs = srcs,
        .srcPos = Dim3(2, 3, 4),
        .extent = Dim3(1, 1, 1),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translator->prepare(std::vector<Translator::Params>(1, ps));
    translator->async(0);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    REQUIRE(dst.ptr[1 * (4 * 3) + 1 * (3) + 1] == 59);
  }
  SECTION("1,2,3 [2x2x2] -> 1,1,1") {
    size_t elemSizes = sizeof(TestType);
    cudaPitchedPtr cudaSrc = cudaPitchedPtr(src);
    cudaPitchedPtr *srcs = &cudaSrc;
    cudaPitchedPtr cudaDst = cudaPitchedPtr(dst);
    cudaPitchedPtr *dsts = &cudaDst;

    Translator::Params ps{
        .dsts = dsts,
        .dstPos = Dim3(1, 1, 1),
        .srcs = srcs,
        .srcPos = Dim3(1, 2, 3),
        .extent = Dim3(2, 2, 2),
        .elemSizes = &elemSizes,
        .n = 1,
    };

    translator->prepare(std::vector<Translator::Params>(1, ps));
    translator->async(0);
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

TEMPLATE_TEST_CASE("translate", "[cuda]", int) {
  INFO("run TranslatorDirectAccess");
  Translator *t = new TranslatorDirectAccess(0);
  all_sections<TestType>(t);
  delete t;
  INFO("run TranslatorMemcpy3D");
  t = new TranslatorMemcpy3D();
  all_sections<TestType>(t);
  delete t;
}