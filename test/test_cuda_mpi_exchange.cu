#include "catch2/catch.hpp"

#include <cstring> // std::memcpy

#include "stencil/accessor.hpp"
#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/rect3.hpp"
#include "stencil/stencil.hpp"

template <typename T>
static __global__ void init_kernel(Accessor<T> dst, //<! [out] region to fill
                                   Rect3 dstExt     //<! [in] the extent of the region to initialize
) {
  const T ripple[4] = {0, 0.25, 0, -0.25};
  const size_t period = sizeof(ripple) / sizeof(ripple[0]);

  const size_t tiz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tix = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = dstExt.lo.z + tiz; z < dstExt.hi.z; z += gridDim.z * blockDim.z) {
    for (size_t y = dstExt.lo.y + tiy; y < dstExt.hi.y; y += gridDim.y * blockDim.y) {
      for (size_t x = dstExt.lo.x + tix; x < dstExt.hi.x; x += gridDim.x * blockDim.x) {

        Dim3 p(x, y, z);
        T val = p.x + ripple[p.x % period] + p.y + ripple[p.y % period] + p.z + ripple[p.z % period];
        dst[p] = val;
      }
    }
  }
}

/* check an exchange that supports the given kernel radius
 */
static void check_exchange(const Radius &radius, const Method methods) {

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank);

  typedef float Q1;

  INFO("ctor");
  DistributedDomain dd(10, 10, 10);

  dd.set_radius(radius);
  auto dh1 = dd.add_data<Q1>("d0");
  dd.set_methods(methods);

  INFO("realize");
  dd.realize();

  INFO("device sync");
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  INFO("barrier");
  MPI_Barrier(MPI_COMM_WORLD);

  INFO("init");
  dim3 dimGrid(10, 10, 10);
  dim3 dimBlock(8, 8, 8);
  for (auto &d : dd.domains()) {
    REQUIRE(d.get_curr(dh1) != PitchedPtr<Q1>());
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    auto acc = d.get_curr_accessor(dh1);

    Rect3 crRect = d.get_compute_region();

    init_kernel<<<dimGrid, dimBlock>>>(acc, crRect);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // test initialization
  INFO("test init");
  for (auto &d : dd.domains()) {
    const Dim3 origin = d.origin();
    const Dim3 ext = d.halo_extent(Dim3(0, 0, 0));

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.interior_to_host(qi);

      // make sure we can access data as a Q1
      std::vector<Q1> interior(ext.flatten());
      REQUIRE(vec.size() == interior.size() * sizeof(Q1));
      std::memcpy(interior.data(), vec.data(), vec.size());

      // create an accessor for the CPU data
      Accessor<Q1> acc(interior.data(), origin, ext);
      Rect3 rect = d.get_compute_region();

      for (int64_t z = rect.lo.z; z < rect.hi.z; ++z) {
        for (int64_t y = rect.lo.y; y < rect.hi.y; ++y) {
          for (int64_t x = rect.lo.x; x < rect.hi.x; ++x) {
            Dim3 p(x, y, z);
            const Q1 ripple[4] = {0, 0.25, 0, -0.25};
            const size_t period = sizeof(ripple) / sizeof(ripple[0]);
            Q1 val = acc[p];
            REQUIRE(val == p.x + ripple[p.x % period] + p.y + ripple[p.y % period] + p.z + ripple[p.z % period]);
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  INFO("exchange");
  dd.exchange();
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }
  MPI_Barrier(MPI_COMM_WORLD);

  INFO("check whole region after exchange");
  for (auto &d : dd.domains()) {

    const Rect3 region = d.get_full_region();
    const Dim3 ext = region.hi - region.lo;

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.quantity_to_host(qi);
      // access quantity data as a Q1
      std::vector<Q1> quantity(ext.flatten());
      REQUIRE(vec.size() == quantity.size() * sizeof(Q1));
      std::memcpy(quantity.data(), vec.data(), vec.size());

      // create an accessor for the CPU data
      // the subdomain origin does not include the halo, but the access origin
      // is the point for the 0th offset in data
      Dim3 origin = d.origin();
      origin.x -= radius.x(-1);
      origin.y -= radius.y(-1);
      origin.z -= radius.z(-1);
      Accessor<Q1> acc(quantity.data(), origin, ext);
      Rect3 rect = d.get_full_region();

      std::cerr << "full region: " << rect << "\n";
      std::cerr << "compute region: " << d.get_compute_region() << "\n";
      std::cerr << "acc.origin()=" << acc.origin() << "\n";

      for (int64_t z = rect.lo.z; z < rect.hi.z; ++z) {
        for (int64_t y = rect.lo.y; y < rect.hi.y; ++y) {
          for (int64_t x = rect.lo.x; x < rect.hi.x; ++x) {
            Dim3 p(x, y, z);
            const Q1 ripple[4] = {0, 0.25, 0, -0.25};
            const size_t period = sizeof(ripple) / sizeof(ripple[0]);
            Q1 val = acc[p];

            // std::cerr << p;

            // if p is on one of the outside shell, should have recieved a value
            // from the interior of the opposite side
            if (p.x < 0) {
              p.x += dd.size().x;
            }
            if (p.y < 0) {
              p.y += dd.size().y;
            }
            if (p.z < 0) {
              p.z += dd.size().z;
            }
            if (p.x >= dd.size().x) {
              p.x -= dd.size().x;
            }
            if (p.y >= dd.size().y) {
              p.y -= dd.size().y;
            }
            if (p.z >= dd.size().z) {
              p.z -= dd.size().z;
            }

            // std::cerr << "->" << p << "\n";

            REQUIRE(val == p.x + ripple[p.x % period] + p.y + ripple[p.y % period] + p.z + ripple[p.z % period]);
          }
        }
      }
    }
  }
}

TEST_CASE("exchange2") {

  // no transfers, anything should work
  SECTION("r=0,cmpi") { check_exchange(Radius::constant(0), Method::CudaMpi); }
  SECTION("r=0,pmu") { check_exchange(Radius::constant(0), Method::ColoPackMemcpyUnpack); }
  SECTION("r=0,q") { check_exchange(Radius::constant(0), Method::ColoQuantityKernel); }
  SECTION("r=0,r") { check_exchange(Radius::constant(0), Method::ColoRegionKernel); }
  SECTION("r=0,m3") { check_exchange(Radius::constant(0), Method::ColoMemcpy3d); }
  SECTION("r=0,d") { check_exchange(Radius::constant(0), Method::ColoDomainKernel); }
  SECTION("r=0,cmp") { check_exchange(Radius::constant(0), Method::CudaMemcpyPeer); }
  SECTION("r=0,k") { check_exchange(Radius::constant(0), Method::CudaKernel); }

  // CudaMpi works for all cases
  SECTION("r=1,cmpi") { check_exchange(Radius::constant(1), Method::CudaMpi); }
  SECTION("r=1,pmu") { check_exchange(Radius::constant(1), Method::CudaMpi | Method::ColoPackMemcpyUnpack); }
  SECTION("r=1,q") { check_exchange(Radius::constant(1), Method::CudaMpi | Method::ColoQuantityKernel); }
  SECTION("r=1,r") { check_exchange(Radius::constant(1), Method::CudaMpi | Method::ColoRegionKernel); }
  SECTION("r=1,m3") { check_exchange(Radius::constant(1), Method::CudaMpi | Method::ColoMemcpy3d); }
  SECTION("r=1,d") { check_exchange(Radius::constant(1), Method::CudaMpi | Method::ColoDomainKernel); }
  SECTION("r=1,cmp") { check_exchange(Radius::constant(1), Method::CudaMpi | Method::CudaMemcpyPeer); }

  SECTION("r=2") { check_exchange(Radius::constant(2), Method::CudaMpi); }

  SECTION("+x=2") {
    Radius r = Radius::constant(0);
    r.dir(1, 0, 0) = 2;
    check_exchange(r, Method::CudaMpi);
  }

  SECTION("mx=1") { // -x doesnt work as a section on CLI
    Radius r = Radius::constant(0);
    r.dir(-1, 0, 0) = 1;
    check_exchange(r, Method::CudaMpi);
  }

  SECTION("+x=2, mx=1") { // -x doesnt work as a section on CLI
    Radius r = Radius::constant(0);
    r.dir(1, 0, 0) = 2;
    r.dir(-1, 0, 0) = 1;
    check_exchange(r, Method::CudaMpi);
  }
}

#if 0
TEST_CASE("exchange3") {

  SECTION("+x=2, mx=1") { // -x doesnt work as a section on CLI
    Radius r = Radius::constant(0);
    r.dir(1, 0, 0) = 2;
    r.dir(-1, 0, 0) = 1;
    check_exchange(r, Method::ColoQuantityKernel);
  }
}
#endif
