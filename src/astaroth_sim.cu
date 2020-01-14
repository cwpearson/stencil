#include <chrono>
#include <thread>

#include <nvToolsExt.h>

#include "stencil/stencil.hpp"

int main(int argc, char **argv) {





  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(provided == MPI_THREAD_MULTIPLE);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t x = 64 * size;
  size_t y = 64;
  size_t z = 64;
  size_t radius = 3;

  size_t kernelMillis = 500;

  DistributedDomain dd(x, y, z);

  dd.set_radius(radius);

  dd.add_data<float>();
  dd.add_data<float>();
  dd.add_data<float>();
  dd.add_data<float>();

  dd.realize();

  MPI_Barrier(MPI_COMM_WORLD);

  nvtxRangePush("exchange");
  dd.exchange();
  nvtxRangePop();

  nvtxRangePush("kernels");
  for (auto &d : dd.domains()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kernelMillis));
  }
  nvtxRangePop();

  MPI_Finalize();

  return 0;
}