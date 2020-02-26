#include "catch2/catch.hpp"

#include "stencil/local_domain.cuh"

TEMPLATE_TEST_CASE("local domain", "[cuda][template]", int, double) {

  const Dim3 sz(30, 40, 50);
  const int gpu = 0;
  const size_t radius = 4;
  LocalDomain d0(sz, gpu);

  d0.set_radius(radius);
  auto handle = d0.add_data<TestType>();

  d0.realize();
  TestType *p = d0.get_curr(handle);
  REQUIRE(p != nullptr);

  SECTION("face position in halo") {
    bool isHalo = true;

    REQUIRE(Dim3(0, 4, 4) == d0.halo_pos(Dim3(-1, 0, 0), isHalo)); // -x
    REQUIRE(Dim3(34, 4, 4) == d0.halo_pos(Dim3(1, 0, 0), isHalo)); // +x
    REQUIRE(Dim3(4, 0, 4) == d0.halo_pos(Dim3(0, -1, 0), isHalo)); // -y
    REQUIRE(Dim3(4, 44, 4) == d0.halo_pos(Dim3(0, 1, 0), isHalo)); // +y
    REQUIRE(Dim3(4, 4, 0) == d0.halo_pos(Dim3(0, 0, -1), isHalo)); // -z
    REQUIRE(Dim3(4, 4, 54) == d0.halo_pos(Dim3(0, 0, 1), isHalo)); // +z
  }

  SECTION("face position in compute") {
    bool isHalo = false;

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, 0, 0), isHalo)); // -x
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, 0, 0), isHalo)); // +x
    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(0, -1, 0), isHalo)); // -y
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(0, 1, 0), isHalo)); // +y
    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(0, 0, -1), isHalo)); // -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(0, 0, 1), isHalo)); // +z
  }

  SECTION("face extent") {
    REQUIRE(Dim3(4, 40, 50) == d0.halo_extent(Dim3(-1, 0, 0))); // x face
    REQUIRE(Dim3(30, 4, 50) == d0.halo_extent(Dim3(0, -1, 0))); // y face
    REQUIRE(Dim3(30, 40, 4) == d0.halo_extent(Dim3(0, 0, -1))); // z face
  }

  SECTION("edge position in halo") {
    bool isHalo = true;
    REQUIRE(Dim3(0, 0, 4) == d0.halo_pos(Dim3(-1, -1, 0), isHalo)); // -x -y
    REQUIRE(Dim3(34, 0, 4) == d0.halo_pos(Dim3(1, -1, 0), isHalo)); // +x -y
    REQUIRE(Dim3(0, 44, 4) == d0.halo_pos(Dim3(-1, 1, 0), isHalo)); // -x +y
    REQUIRE(Dim3(34, 44, 4) == d0.halo_pos(Dim3(1, 1, 0), isHalo)); // +x +y

    REQUIRE(Dim3(0, 4, 0) == d0.halo_pos(Dim3(-1, 0, -1), isHalo)); // -x -z
    REQUIRE(Dim3(34, 4, 0) == d0.halo_pos(Dim3(1, 0, -1), isHalo)); // +x -z
    REQUIRE(Dim3(0, 4, 54) == d0.halo_pos(Dim3(-1, 0, 1), isHalo)); // -x +z
    REQUIRE(Dim3(34, 4, 54) == d0.halo_pos(Dim3(1, 0, 1), isHalo)); // +x +z

    REQUIRE(Dim3(4, 0, 0) == d0.halo_pos(Dim3(0, -1, -1), isHalo)); // -y -z
    REQUIRE(Dim3(4, 44, 0) == d0.halo_pos(Dim3(0, 1, -1), isHalo)); // +y -z
    REQUIRE(Dim3(4, 0, 54) == d0.halo_pos(Dim3(0, -1, 1), isHalo)); // -y +z
    REQUIRE(Dim3(4, 44, 54) == d0.halo_pos(Dim3(0, 1, 1), isHalo)); // +y +z

    REQUIRE(Dim3(0, 0, 4) == d0.halo_pos(Dim3(-1, -1, 0), isHalo)); // -x -y
    REQUIRE(Dim3(34, 0, 4) == d0.halo_pos(Dim3(1, -1, 0), isHalo)); // +x -y
    REQUIRE(Dim3(0, 44, 4) == d0.halo_pos(Dim3(-1, 1, 0), isHalo)); // -x +y
    REQUIRE(Dim3(34, 44, 4) == d0.halo_pos(Dim3(1, 1, 0), isHalo)); // +x +y

    REQUIRE(Dim3(0, 4, 0) == d0.halo_pos(Dim3(-1, 0, -1), isHalo)); // -x -z
    REQUIRE(Dim3(34, 4, 0) == d0.halo_pos(Dim3(1, 0, -1), isHalo)); // +x -z
    REQUIRE(Dim3(0, 4, 54) == d0.halo_pos(Dim3(-1, 0, 1), isHalo)); // -x +z
    REQUIRE(Dim3(34, 4, 54) == d0.halo_pos(Dim3(1, 0, 1), isHalo)); // +x +z

    REQUIRE(Dim3(4, 0, 0) == d0.halo_pos(Dim3(0, -1, -1), isHalo)); // -y -z
    REQUIRE(Dim3(4, 44, 0) == d0.halo_pos(Dim3(0, 1, -1), isHalo)); // +y -z
    REQUIRE(Dim3(4, 0, 54) == d0.halo_pos(Dim3(0, -1, 1), isHalo)); // -y +z
    REQUIRE(Dim3(4, 44, 54) == d0.halo_pos(Dim3(0, 1, 1), isHalo)); // +y +z
  }

  SECTION("edge position in compute") {
    bool isHalo = false;
    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, -1, 0), isHalo)); // -x -y
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, -1, 0), isHalo)); // +x -y
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(-1, 1, 0), isHalo)); // -x +y
    REQUIRE(Dim3(30, 40, 4) == d0.halo_pos(Dim3(1, 1, 0), isHalo)); // +x +y

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, 0, -1), isHalo)); // -x -z
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, 0, -1), isHalo)); // +x -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(-1, 0, 1), isHalo)); // -x +z
    REQUIRE(Dim3(30, 4, 50) == d0.halo_pos(Dim3(1, 0, 1), isHalo)); // +x +z

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(0, -1, -1), isHalo)); // -y -z
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(0, 1, -1), isHalo)); // +y -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(0, -1, 1), isHalo)); // -y +z
    REQUIRE(Dim3(4, 40, 50) == d0.halo_pos(Dim3(0, 1, 1), isHalo)); // +y +z

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, -1, 0), isHalo)); // -x -y
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, -1, 0), isHalo)); // +x -y
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(-1, 1, 0), isHalo)); // -x +y
    REQUIRE(Dim3(30, 40, 4) == d0.halo_pos(Dim3(1, 1, 0), isHalo)); // +x +y

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, 0, -1), isHalo)); // -x -z
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, 0, -1), isHalo)); // +x -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(-1, 0, 1), isHalo)); // -x +z
    REQUIRE(Dim3(30, 4, 50) == d0.halo_pos(Dim3(1, 0, 1), isHalo)); // +x +z

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(0, -1, -1), isHalo)); // -y -z
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(0, 1, -1), isHalo)); // +y -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(0, -1, 1), isHalo)); // -y +z
    REQUIRE(Dim3(4, 40, 50) == d0.halo_pos(Dim3(0, 1, 1), isHalo)); // +y +z
  }

  SECTION("edge extent") {
    REQUIRE(Dim3(4, 4, 50) == d0.halo_extent(Dim3(1, 1, 0))); // x y edge
    REQUIRE(Dim3(4, 40, 4) == d0.halo_extent(Dim3(1, 0, 1))); // x z edge
    REQUIRE(Dim3(30, 4, 4) == d0.halo_extent(Dim3(0, 1, 1))); // y z edge
    REQUIRE(Dim3(4, 4, 50) == d0.halo_extent(Dim3(1, 1, 0))); // x y edge
    REQUIRE(Dim3(4, 40, 4) == d0.halo_extent(Dim3(1, 0, 1))); // x z edge
    REQUIRE(Dim3(30, 4, 4) == d0.halo_extent(Dim3(0, 1, 1))); // y z edge
  }

  SECTION("corner position in halo") {
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(-1, -1, -1), true)); // -x -y -z
    REQUIRE(Dim3(34, 0, 0) == d0.halo_pos(Dim3(1, -1, -1), true)); // +x -y -z
    REQUIRE(Dim3(0, 44, 0) == d0.halo_pos(Dim3(-1, 1, -1), true)); // -x +y -z
    REQUIRE(Dim3(34, 44, 0) == d0.halo_pos(Dim3(1, 1, -1), true)); // +x +y -z
    REQUIRE(Dim3(0, 0, 54) == d0.halo_pos(Dim3(-1, -1, 1), true)); // -x -y +z
    REQUIRE(Dim3(34, 0, 54) == d0.halo_pos(Dim3(1, -1, 1), true)); // +x -y +z
    REQUIRE(Dim3(0, 44, 54) == d0.halo_pos(Dim3(-1, 1, 1), true)); // -x +y +z
    REQUIRE(Dim3(34, 44, 54) == d0.halo_pos(Dim3(1, 1, 1), true)); // +x +y +z

    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(-1, -1, -1), true)); // -x -y -z
    REQUIRE(Dim3(34, 0, 0) == d0.halo_pos(Dim3(1, -1, -1), true)); // +x -y -z
    REQUIRE(Dim3(0, 44, 0) == d0.halo_pos(Dim3(-1, 1, -1), true)); // -x +y -z
    REQUIRE(Dim3(34, 44, 0) == d0.halo_pos(Dim3(1, 1, -1), true)); // +x +y -z
    REQUIRE(Dim3(0, 0, 54) == d0.halo_pos(Dim3(-1, -1, 1), true)); // -x -y +z
    REQUIRE(Dim3(34, 0, 54) == d0.halo_pos(Dim3(1, -1, 1), true)); // +x -y +z
    REQUIRE(Dim3(0, 44, 54) == d0.halo_pos(Dim3(-1, 1, 1), true)); // -x +y +z
    REQUIRE(Dim3(34, 44, 54) == d0.halo_pos(Dim3(1, 1, 1), true)); // +x +y +z
  }

  SECTION("corner position in compute") {
    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, -1, -1), false)); // -x -y -z
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, -1, -1), false)); // +x -y -z
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(-1, 1, -1), false)); // -x +y -z
    REQUIRE(Dim3(30, 40, 4) == d0.halo_pos(Dim3(1, 1, -1), false)); // +x +y -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(-1, -1, 1), false)); // -x -y +z
    REQUIRE(Dim3(30, 4, 50) == d0.halo_pos(Dim3(1, -1, 1), false)); // +x -y +z
    REQUIRE(Dim3(4, 40, 50) == d0.halo_pos(Dim3(-1, 1, 1), false)); // -x +y +z
    REQUIRE(Dim3(30, 40, 50) == d0.halo_pos(Dim3(1, 1, 1), false)); // +x +y +z

    REQUIRE(Dim3(4, 4, 4) == d0.halo_pos(Dim3(-1, -1, -1), false)); // -x -y -z
    REQUIRE(Dim3(30, 4, 4) == d0.halo_pos(Dim3(1, -1, -1), false)); // +x -y -z
    REQUIRE(Dim3(4, 40, 4) == d0.halo_pos(Dim3(-1, 1, -1), false)); // -x +y -z
    REQUIRE(Dim3(30, 40, 4) == d0.halo_pos(Dim3(1, 1, -1), false)); // +x +y -z
    REQUIRE(Dim3(4, 4, 50) == d0.halo_pos(Dim3(-1, -1, 1), false)); // -x -y +z
    REQUIRE(Dim3(30, 4, 50) == d0.halo_pos(Dim3(1, -1, 1), false)); // +x -y +z
    REQUIRE(Dim3(4, 40, 50) == d0.halo_pos(Dim3(-1, 1, 1), false)); // -x +y +z
    REQUIRE(Dim3(30, 40, 50) == d0.halo_pos(Dim3(1, 1, 1), false)); // +x +y +z
  }

  SECTION("corner extent") {
    REQUIRE(Dim3(4, 4, 4) == d0.halo_extent(Dim3(1, 1, 1)));
  }
}

template <typename T>
__global__ void init_kernel(
    T *__restrict__ dst, //<! [out] pointer to beginning of dst allocation
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

  // initialize the compute domain
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

template <typename T>
__global__ void
stencil_kernel(T *__restrict__ dst,       //<! [out] beginning of dst allocation
               const T *__restrict__ src, //<! [in] beginning of src allooation
               const Dim3 rawSz //<! [in] 3D size of the dst allocations
) {

  constexpr size_t radius = 1;
  const Dim3 domSz = rawSz - Dim3(2 * radius, 2 * radius, 2 * radius);

// assume arr is the beginning of the allocation, not the beginning of the
// compute domain
#define _at(arr, _x, _y, _z)                                                   \
  arr[(_z + radius) * rawSz.y * rawSz.x + (_y + radius) * rawSz.x + _x + radius]

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

  for (size_t z = biz * bdz + tiz; z < domSz.z; z += gdz * bdz) {
    for (size_t y = biy * bdy + tiy; y < domSz.y; y += gdy * bdy) {
      for (size_t x = bix * bdx + tix; x < domSz.x; x += gdx * bdx) {

        T acc = 0;
        for (int dz = -1; dz <= 1; dz += 1) {
          for (int dy = -1; dy <= 1; dy += 1) {
            for (int dx = -1; dx <= 1; dx += 1) {
              size_t srcX = x + dx;
              size_t srcY = y + dy;
              size_t srcZ = z + dz;

              T inc = _at(src, srcX, srcY, srcZ);
              acc += inc;
            }
          }
        }
        _at(dst, x, y, z) = acc;
      }
    }
  }

#undef _at
}

TEMPLATE_TEST_CASE("local domain stencil", "[cuda][template]", int, double) {

  // create a domain
  INFO("ctor");
  LocalDomain ld(Dim3(10, 10, 10), /*gpu*/ 0);
  ld.set_radius(1);
  auto h = ld.add_data<TestType>();
  INFO("realize");
  ld.realize();

  // initialize the domain
  INFO("init");
  dim3 dimGrid(2, 16, 32);
  dim3 dimBlock(1, 1, 1);
  init_kernel<<<dimGrid, dimBlock>>>(ld.get_curr(h), ld.raw_size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // check the initialization
  INFO("d2h");
  auto vec = ld.quantity_to_host(0);
  REQUIRE(vec.size() == 12 * 12 * 12 * sizeof(TestType));
  TestType *host = reinterpret_cast<TestType *>(vec.data());

  INFO("check initialization");
#define at_host(_x, _y, _z) host[(_z + 1) * 12 * 12 + (_y + 1) * 12 + (_x + 1)]
  REQUIRE(at_host(-1, -1, -1) == 0);
  REQUIRE(at_host(0, 0, 0) == 1);
  REQUIRE(at_host(0, 0, 9) == 1);
  REQUIRE(at_host(0, 9, 0) == 1);
  REQUIRE(at_host(0, 9, 9) == 1);
  REQUIRE(at_host(9, 0, 0) == 1);
  REQUIRE(at_host(9, 0, 9) == 1);
  REQUIRE(at_host(9, 9, 0) == 1);
  REQUIRE(at_host(9, 9, 9) == 1);
#undef at_host

  // apply the stencil
  stencil_kernel<<<dimGrid, dimBlock>>>(ld.get_next(h), ld.get_curr(h),
                                        ld.raw_size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // check the results
  CUDA_RUNTIME(cudaMemcpy(host, ld.get_next(h),
                          ld.raw_size().flatten() * sizeof(TestType),
                          cudaMemcpyDefault));

#define at_host(_x, _y, _z) host[(_z + 1) * 12 * 12 + (_y + 1) * 12 + (_x + 1)]
  // halo should be untouched
  REQUIRE(at_host(-1, -1, -1) == 0);
  REQUIRE(at_host(10, 10, 10) == 0);
  // corners have 8 nbrs
  REQUIRE(at_host(0, 0, 0) == 8);
  REQUIRE(at_host(0, 0, 9) == 8);
  REQUIRE(at_host(0, 9, 0) == 8);
  REQUIRE(at_host(0, 9, 9) == 8);
  REQUIRE(at_host(9, 0, 0) == 8);
  REQUIRE(at_host(9, 0, 9) == 8);
  REQUIRE(at_host(9, 9, 0) == 8);
  REQUIRE(at_host(9, 9, 9) == 8);
  // edges have 12 nbrs
  REQUIRE(at_host(0, 0, 4) == 12);
  // faces have 18 ones
  REQUIRE(at_host(0, 4, 4) == 18);
  // center has 27 ones
  REQUIRE(at_host(1, 1, 1) == 27);
#undef at_host
}