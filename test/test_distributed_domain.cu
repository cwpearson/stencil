#include "catch2/catch.hpp"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/stencil.hpp"

__device__ int pack_xyz(int x, int y, int z) {
  int ret = 0;
  ret |= x & 0x3FF;
  ret |= (y & 0x3FF) << 10;
  ret |= (z & 0x3FF) << 20;
  return ret;
}

int unpack_x(int a) { return a & 0x3FF; }

int unpack_y(int a) { return (a >> 10) & 0x3FF; }

int unpack_z(int a) { return (a >> 20) & 0x3FF; }

template <typename T>
__global__ void
init_kernel(T *dst,          //<! [out] pointer to beginning of dst allocation
            const Dim3 rawSz //<! [in] 3D size of the dst and src allocations
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
          _at(dst, x, y, z) = pack_xyz(x, y, z);
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
  DistributedDomain dd(10, 10, 10);
  dd.set_radius(radius);
  auto dh1 = dd.add_data<TestType1>();
  dd.realize();

  MPI_Barrier(MPI_COMM_WORLD);

  INFO("init");
  dim3 dimGrid(2, 2, 2);
  dim3 dimBlock(8, 8, 8);
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    init_kernel<<<dimGrid, dimBlock>>>(d.get_curr(dh1), d.raw_size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  MPI_Barrier(MPI_COMM_WORLD);

  INFO("exchange");
  dd.exchange();

  // test exchange
}