#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "stencil/stencil.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numSubdoms;
  {
    MpiTopology topo(MPI_COMM_WORLD);
    numSubdoms = size / topo.colocated_size() * devCount;
  }

  size_t x = 512 * pow(numSubdoms, 0.33333) + 0.5; // round to nearest
  size_t y = 512 * pow(numSubdoms, 0.33333) + 0.5;
  size_t z = 512 * pow(numSubdoms, 0.33333) + 0.5;

  MethodFlags methods = MethodFlags::All;

  if (0 == rank) {
    std::cout << numSubdoms << " subdomains " << size << " ranks: " << x << "," << y << "," << z << "=" << x*y*z << "\n";
    if ((methods & MethodFlags::CudaMpi) != MethodFlags::None) {
      std::cout << "CudaMpi enabled\n";
    }
    if ((methods & MethodFlags::CudaAwareMpi) != MethodFlags::None) {
      std::cout << "CudaAwareMpi enabled\n";
    }
    if ((methods & MethodFlags::CudaMpiColocated) != MethodFlags::None) {
      std::cout << "CudaMpiColocated enabled\n";
    }
    if ((methods & MethodFlags::CudaMemcpyPeer) != MethodFlags::None) {
      std::cout << "CudaMemcpyPeer enabled\n";
    }
    if ((methods & MethodFlags::CudaKernel) != MethodFlags::None) {
      std::cout << "CudaKernel enabled\n";
    }
  }

  {
    size_t radius = 3;

    DistributedDomain dd(x, y, z);

    dd.set_methods(methods);
    dd.set_radius(radius);

    dd.add_data<float>();
    dd.add_data<float>();
    dd.add_data<float>();
    dd.add_data<float>();

    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t iter = 0; iter < 5; ++iter) {
      std::cerr << "exchange\n";
      nvtxRangePush("exchange");
      dd.exchange();
      nvtxRangePop();

    }
  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
