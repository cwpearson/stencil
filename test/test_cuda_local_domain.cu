#include "catch2/catch.hpp"

#include "stencil/local_domain.cuh"

TEST_CASE("case1", "[cuda]") {
  Dim3 arrSz(3, 4, 5);
  Dim3 origin(0, 0, 0);
  LocalDomain ld(arrSz, origin, 0);
  Radius radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 2;
  radius.dir(-1, 0, 0) = 1;
  ld.set_radius(radius);
  ld.realize();

  // +x send is size of -x side halo
  REQUIRE(ld.halo_extent(Dim3(1, 0, 0) * -1) == Dim3(1, 4, 5));
}

TEST_CASE("curr!=next", "[cuda]") {
  Dim3 arrSz(3, 4, 5);
  Dim3 origin(0, 0, 0);
  LocalDomain ld(arrSz, origin, 0);
  auto h = ld.add_data<float>();
  Radius radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 2;
  radius.dir(-1, 0, 0) = 1;
  ld.set_radius(radius);
  ld.realize();

  // cur and next pointers should be different
  REQUIRE(ld.get_curr(h).ptr != ld.get_next(h).ptr);
}

TEMPLATE_TEST_CASE("symmetric radius", "[cuda][template]", int, double) {

  const Dim3 sz(30, 40, 50);
  const Dim3 origin(0, 0, 0);
  const int gpu = 0;
  const size_t radius = 4;
  LocalDomain d0(sz, origin, gpu);

  d0.set_radius(radius);
  auto handle = d0.add_data<TestType>();

  d0.realize();
  PitchedPtr<TestType> p = d0.get_curr(handle);
  REQUIRE(p != PitchedPtr<TestType>());

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
    const bool isHalo = true;
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(-1, -1, -1), isHalo)); // -x -y -z
    REQUIRE(Dim3(34, 0, 0) == d0.halo_pos(Dim3(1, -1, -1), isHalo)); // +x -y -z
    REQUIRE(Dim3(0, 44, 0) == d0.halo_pos(Dim3(-1, 1, -1), isHalo)); // -x +y -z
    REQUIRE(Dim3(34, 44, 0) == d0.halo_pos(Dim3(1, 1, -1), isHalo)); // +x +y -z
    REQUIRE(Dim3(0, 0, 54) == d0.halo_pos(Dim3(-1, -1, 1), isHalo)); // -x -y +z
    REQUIRE(Dim3(34, 0, 54) == d0.halo_pos(Dim3(1, -1, 1), isHalo)); // +x -y +z
    REQUIRE(Dim3(0, 44, 54) == d0.halo_pos(Dim3(-1, 1, 1), isHalo)); // -x +y +z
    REQUIRE(Dim3(34, 44, 54) == d0.halo_pos(Dim3(1, 1, 1), isHalo)); // +x +y +z

    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(-1, -1, -1), isHalo)); // -x -y -z
    REQUIRE(Dim3(34, 0, 0) == d0.halo_pos(Dim3(1, -1, -1), isHalo)); // +x -y -z
    REQUIRE(Dim3(0, 44, 0) == d0.halo_pos(Dim3(-1, 1, -1), isHalo)); // -x +y -z
    REQUIRE(Dim3(34, 44, 0) == d0.halo_pos(Dim3(1, 1, -1), isHalo)); // +x +y -z
    REQUIRE(Dim3(0, 0, 54) == d0.halo_pos(Dim3(-1, -1, 1), isHalo)); // -x -y +z
    REQUIRE(Dim3(34, 0, 54) == d0.halo_pos(Dim3(1, -1, 1), isHalo)); // +x -y +z
    REQUIRE(Dim3(0, 44, 54) == d0.halo_pos(Dim3(-1, 1, 1), isHalo)); // -x +y +z
    REQUIRE(Dim3(34, 44, 54) == d0.halo_pos(Dim3(1, 1, 1), isHalo)); // +x +y +z
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

  SECTION("corner extent") { REQUIRE(Dim3(4, 4, 4) == d0.halo_extent(Dim3(1, 1, 1))); }
}

TEMPLATE_TEST_CASE("x-leaning radius", "[cuda][template]", int, double) {

  const Dim3 sz(30, 40, 50);
  const Dim3 origin(0, 0, 0);
  const int gpu = 0;
  Radius radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 3; // +x

  LocalDomain d0(sz, origin, gpu);

  d0.set_radius(radius);
  auto handle = d0.add_data<TestType>();

  d0.realize();
  PitchedPtr<TestType> p = d0.get_curr(handle);
  REQUIRE(p != PitchedPtr<TestType>());

  SECTION("face position in halo") {
    bool isHalo = true;
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(-1, 0, 0), isHalo)); // -x
    REQUIRE(Dim3(30, 0, 0) == d0.halo_pos(Dim3(1, 0, 0), isHalo)); // +x
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(0, -1, 0), isHalo)); // -y
    REQUIRE(Dim3(0, 40, 0) == d0.halo_pos(Dim3(0, 1, 0), isHalo)); // +y
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(0, 0, -1), isHalo)); // -z
    REQUIRE(Dim3(0, 0, 50) == d0.halo_pos(Dim3(0, 0, 1), isHalo)); // +z
  }

  SECTION("face position in compute") {
    bool isHalo = true;
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(-1, 0, 0), isHalo)); // -x
    // no interior halo region on this side since it would send to -x
    REQUIRE(Dim3(30, 0, 0) == d0.halo_pos(Dim3(1, 0, 0), isHalo)); // +x
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(0, -1, 0), isHalo)); // -y
    REQUIRE(Dim3(0, 40, 0) == d0.halo_pos(Dim3(0, 1, 0), isHalo)); // +y
    REQUIRE(Dim3(0, 0, 0) == d0.halo_pos(Dim3(0, 0, -1), isHalo)); // -z
    REQUIRE(Dim3(0, 0, 50) == d0.halo_pos(Dim3(0, 0, 1), isHalo)); // +z
  }

  SECTION("face extent") {
    REQUIRE(Dim3(3, 40, 50) == d0.halo_extent(Dim3(1, 0, 0)));  // +x face
    REQUIRE(Dim3(0, 40, 50) == d0.halo_extent(Dim3(-1, 0, 0))); // -x face
    REQUIRE(Dim3(30, 0, 50) == d0.halo_extent(Dim3(0, 1, 0)));  // +y face
    REQUIRE(Dim3(30, 0, 50) == d0.halo_extent(Dim3(0, -1, 0))); // -y face
    REQUIRE(Dim3(30, 40, 0) == d0.halo_extent(Dim3(0, 0, 1)));  // +z face
    REQUIRE(Dim3(30, 40, 0) == d0.halo_extent(Dim3(0, 0, -1))); // -z face
  }
}

template <typename T>
__global__ void init_kernel(PitchedPtr<T> dst, //<! [out] pointer to beginning of dst allocation
                            const Dim3 rawSz   //<! [in] logical extent of the dst allocation
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

  // initialize the compute domain
  for (size_t z = biz * bdz + tiz; z < rawSz.z; z += gdz * bdz) {
    for (size_t y = biy * bdy + tiy; y < rawSz.y; y += gdy * bdy) {
      for (size_t x = bix * bdx + tix; x < rawSz.x; x += gdx * bdx) {

        if (z >= radius && x >= radius && y >= radius && z < rawSz.z - radius && y < rawSz.y - radius &&
            x < rawSz.x - radius) {
          dst.at(x, y, z) = 1.0;
        } else {
          dst.at(x, y, z) = 0.0;
        }
      }
    }
  }
}

template <typename T>
__global__ void stencil_kernel(PitchedPtr<T> dst,       //<! [out] beginning of dst allocation
                               const PitchedPtr<T> src, //<! [in] beginning of src allooation
                               const Dim3 rawSz         //<! [in] 3D size of the dst allocations
) {

  constexpr size_t radius = 1;
  const Dim3 domSz = rawSz - Dim3(2 * radius, 2 * radius, 2 * radius);

  // assume arr is the beginning of the allocation, not the beginning of the compute domain

  for (int64_t z = blockIdx.z * blockDim.z + threadIdx.z; z < domSz.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = blockIdx.y * blockDim.y + threadIdx.y; y < domSz.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < domSz.x; x += gridDim.x * blockDim.x) {

        T acc = 0;
        for (int dz = -1; dz <= 1; dz += 1) {
          for (int dy = -1; dy <= 1; dy += 1) {
            for (int dx = -1; dx <= 1; dx += 1) {
              int64_t srcX = x + dx;
              int64_t srcY = y + dy;
              int64_t srcZ = z + dz;

              T inc = src.at(srcX + radius, srcY + radius, srcZ + radius);
              acc += inc;
            }
          }
        }
        dst.at(x + radius, y + radius, z + radius) = acc;
      }
    }
  }
}

TEMPLATE_TEST_CASE("local domain stencil", "[cuda][template]", int) {

  // TODO: why does this test fail without this
  // test passes if run alone
  CUDA_RUNTIME(cudaDeviceReset());

  // create a domain
  INFO("ctor");
  const Dim3 origin(0, 0, 0);
  LocalDomain ld(Dim3(10, 10, 10), origin, /*gpu*/ 0);
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
  REQUIRE(at_host(10, 10, 10) == 0);
#undef at_host

  INFO("apply stencil");
  stencil_kernel<<<dimGrid, dimBlock>>>(ld.get_next(h), ld.get_curr(h), ld.raw_size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  /* swap so we can copy the stencil results to the host
   */
  INFO("swap");
  ld.swap();

  INFO("d2h");
  vec.clear();
  vec = ld.quantity_to_host(0);
  REQUIRE(vec.size() == 12 * 12 * 12 * sizeof(TestType));
  host = reinterpret_cast<TestType *>(vec.data());

  // // check the results
  // CUDA_RUNTIME(
  //     cudaMemcpy(host, ld.get_next(h).ptr, ld.raw_size().flatten() * sizeof(TestType), cudaMemcpyDeviceToHost));

  INFO("check results");
#define at_host(_x, _y, _z) host[(_z + 1) * 12 * 12 + (_y + 1) * 12 + (_x + 1)]
  INFO("halo unchanged");
  REQUIRE(0 == at_host(-1, -1, -1));
  REQUIRE(0 == at_host(-1, -1, 0));
  REQUIRE(0 == at_host(-1, 0, 0));
  REQUIRE(0 == at_host(-1, 6, 3));
  REQUIRE(0 == at_host(10, 10, 10));

  INFO("corners have 8 nbrs");

#if 0
  for (int y = -1; y < 11; ++y) {
    for (int x = -1; x < 11; ++x) {
      std::cerr << at_host(x, y, 1) << " ";
    }
    std::cerr << "\n";
  }
#endif

  REQUIRE(at_host(0, 0, 0) == 8);
  REQUIRE(at_host(0, 0, 9) == 8);
  REQUIRE(at_host(0, 9, 0) == 8);
  REQUIRE(at_host(0, 9, 9) == 8);
  REQUIRE(at_host(9, 0, 0) == 8);
  REQUIRE(at_host(9, 0, 9) == 8);
  REQUIRE(at_host(9, 9, 0) == 8);
  REQUIRE(at_host(9, 9, 9) == 8);

  INFO("edges have 12 nbrs");
  REQUIRE(at_host(0, 0, 4) == 12);

  INFO("faces have 18 nbrs");
  REQUIRE(at_host(0, 4, 4) == 18);

  INFO("center has 27 nbrs");
  REQUIRE(at_host(1, 1, 1) == 27);
#undef at_host
}