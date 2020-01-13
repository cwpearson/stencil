#include "catch2/catch.hpp"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/stencil.hpp"


__device__ int pack_dim3(int x, int y, int z) {
  int ret = 0;
  ret |= x & 0x3FF;
  ret |= (y & 0x3FF) << 10;
  ret |= (z & 0x3FF) << 20;
  return ret;
}

int unpack_x(int a) {
  return a & 0x3FF;
}

int unpack_y(int a) {
  return (a >> 10) & 0x3FF;
}

int unpack_z(int a) {
  return (a >> 20) & 0x3FF;
}


template <typename T>
__global__ void init_kernel(
    T *dst, //<! [out] pointer to beginning of dst allocation
    const Dim3 rawSz     //<! [in] 3D size of the dst and src allocations
) {

  constexpr size_t radius = 1;
  const Dim3 domSz = rawSz - Dim3(2 * radius, 2 * radius, 2 * radius);

  const size_t gdz = gridDim.z;
  const size_t biz = blockIdx.z;
  const size_t bdz = blockDim.z;
  const size_t tiz = threadIdx.z;

  const size_t gdy = gridDim.y;
  const size_t biy = blockIdx.y;
  const size_t bdy = blockDim.y;
  const size_t tiy = threadIdx.y;

  const size_t gdx = gridDim.x;
  const size_t bix = blockIdx.x;
  const size_t bdx = blockDim.x;
  const size_t tix = threadIdx.x;

#define _at(arr, _x, _y, _z) arr[_z * rawSz.y * rawSz.x + _y * rawSz.x + _x]

  // initialize the compute domain and set halos to zero
  for (size_t z = biz * bdz + tiz; z < rawSz.z; z += gdz * bdz) {
    for (size_t y = biy * bdy + tiy; y < rawSz.y; y += gdy * bdy) {
      for (size_t x = bix * bdx + tix; x < rawSz.x; x += gdx * bdx) {

        if (z >= radius && x >= radius && y >= radius && z < rawSz.z - radius &&
            y < rawSz.y - radius && x < rawSz.x - radius) {
          _at(dst, x, y, z) = 1.0;
        } else {
          _at(dst, x, y, z) = 0.0;
        }
      }
    }
  }

#undef _at
}

TEST_CASE("exchange") {

  size_t radius = 1;
  typedef float TestType1;

  INFO("create");
  DistributedDomain dd(10,10,10);
  dd.set_radius(radius);
  auto dh1 = dd.add_data<TestType1>();
  dd.realize();

  INFO("init");
  dim3 dimGrid(2,2,2);
  dim3 dimBlock(8,8,8);
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    init_kernel<<<dimGrid, dimBlock>>>(d.get_curr(dh1), d.raw_size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  INFO("exchange");
  dd.exchange();

  // test exchange


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




