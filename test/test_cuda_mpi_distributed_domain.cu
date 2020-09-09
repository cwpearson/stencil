#include "catch2/catch.hpp"

#include <cstring> // std::memcpy

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/rect3.hpp"
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

/*! Set the compute region to a packed version of the logical coordinate, and the halo to -1
 */
template <typename T>
static __global__ void init_kernel(Accessor<T> dst, //<! [out] pointer to beginning of dst allocation
                                   Rect3 crExt      //<! coordinates of the compute region
                                                    //,
                                                    // bool loud
) {
  constexpr int64_t radius = 1;

  const int gdz = gridDim.z;
  const int biz = blockIdx.z;
  const int bdz = blockDim.z;
  const int tiz = threadIdx.z;

  const int gdy = gridDim.y;
  const int biy = blockIdx.y;
  const int bdy = blockDim.y;
  const int tiy = threadIdx.y;

  const int gdx = gridDim.x;
  const int bix = blockIdx.x;
  const int bdx = blockDim.x;
  const int tix = threadIdx.x;

  for (int64_t z = crExt.lo.z - radius + biz * bdz + tiz; z < crExt.hi.z + radius; z += gdz * bdz) {
    for (int64_t y = crExt.lo.y - radius + biy * bdy + tiy; y < crExt.hi.y + radius; y += gdy * bdy) {
      for (int64_t x = crExt.lo.x - radius + bix * bdx + tix; x < crExt.hi.x + radius; x += gdx * bdx) {

        Dim3 p(x, y, z);

        if (z >= crExt.lo.z && y >= crExt.lo.y && x >= crExt.lo.x && z < crExt.hi.z && y < crExt.hi.y &&
            x < crExt.hi.x) {
          int val = pack_xyz(x, y, z);
          dst[p] = val;
        } else {
          dst[p] = -1;
        }
      }
    }
  }
}

TEST_CASE("exchange1") {

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank);

  size_t radius = 1;
  typedef float Q1;

  INFO("ctor");
  DistributedDomain dd(10, 10, 10);
  dd.set_radius(radius);
  auto dh1 = dd.add_data<Q1>("d0");
  dd.set_methods(MethodFlags::CudaMpi);

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
  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    REQUIRE(d.get_curr(dh1) != PitchedPtr<Q1>());

    std::cerr << "org=" << d.get_curr_accessor(dh1).origin() << " cr=" << d.get_compute_region() << "\n";
    std::cerr << "xsize=" << d.get_curr_accessor(dh1).ptr().xsize << " pitch=" << d.get_curr_accessor(dh1).ptr().pitch
              << "\n";
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    init_kernel<<<dimGrid, dimBlock>>>(d.get_curr_accessor(dh1), d.get_compute_region() /*, mpi::world_rank() == 0*/);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  INFO("barrier");
  MPI_Barrier(MPI_COMM_WORLD);

  // test initialization
  INFO("test compute region");
  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    const Dim3 origin = d.origin();
    const Dim3 ext = d.halo_extent(Dim3(0, 0, 0));

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.interior_to_host(qi);

      // make sure we can access data as a Q1
      std::vector<Q1> interior(ext.flatten());
      REQUIRE(vec.size() == interior.size() * sizeof(Q1));
      std::memcpy(interior.data(), vec.data(), vec.size());

      for (int64_t z = 0; z < ext.z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            Q1 val = interior[z * (ext.y * ext.x) + y * (ext.x) + x];
            REQUIRE(unpack_x(val) == x + origin.x);
            REQUIRE(unpack_y(val) == y + origin.y);
            REQUIRE(unpack_z(val) == z + origin.z);
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  INFO("exchange");

  dd.exchange();
  CUDA_RUNTIME(cudaDeviceSynchronize());

  INFO("interior should be unchanged");
  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    const Dim3 origin = d.origin();
    const Dim3 ext = d.halo_extent(Dim3(0, 0, 0));

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.interior_to_host(qi);

      // make sure we can access data as a Q1
      std::vector<Q1> interior(ext.flatten());
      REQUIRE(vec.size() == interior.size() * sizeof(Q1));
      std::memcpy(interior.data(), vec.data(), vec.size());

      for (int64_t z = 0; z < ext.z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            Q1 val = interior[z * (ext.y * ext.x) + y * (ext.x) + x];
            REQUIRE(unpack_x(val) == x + origin.x);
            REQUIRE(unpack_y(val) == y + origin.y);
            REQUIRE(unpack_z(val) == z + origin.z);
          }
        }
      }
    }
  }

  INFO("check halo regions");

  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    const Dim3 origin = d.origin();

    Dim3 ext = d.size();
    ext.x += 2 * radius;
    ext.y += 2 * radius;
    ext.z += 2 * radius;

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.quantity_to_host(qi);
      // access quantity data as a Q1
      std::vector<Q1> quantity(ext.flatten());
      REQUIRE(vec.size() == quantity.size() * sizeof(Q1));
      std::memcpy(quantity.data(), vec.data(), vec.size());

      for (int64_t z = 0; z < ext.z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            Dim3 xyz = Dim3(x, y, z);
            Dim3 coord = xyz - Dim3(radius, radius, radius) + origin;
            coord = coord.wrap(Dim3(10, 10, 10));

            Q1 val = quantity[z * (ext.y * ext.x) + y * (ext.x) + x];
            REQUIRE(unpack_x(val) == coord.x);
            REQUIRE(unpack_y(val) == coord.y);
            REQUIRE(unpack_z(val) == coord.z);
          }
        }
      }
    }
  }
}

TEST_CASE("swap") {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank);

  size_t radius = 1;
  typedef float Q1;

  INFO("ctor");
  DistributedDomain dd(10, 10, 10);
  dd.set_radius(radius);
  auto dh1 = dd.add_data<Q1>("d0");
  dd.set_methods(MethodFlags::CudaMpi);

  INFO("realize");
  dd.realize();

  INFO("device sync");
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  INFO("barrier");
  MPI_Barrier(MPI_COMM_WORLD);

  INFO("swap");
  dd.swap();
}