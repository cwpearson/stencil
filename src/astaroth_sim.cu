#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "stencil/stencil.hpp"

int main(int argc, char **argv) {

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(provided == MPI_THREAD_MULTIPLE);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*
  Table 5
  512^3 on Pascal 34.1ms
  512^3 on Volta  20.1ms
  */

  size_t x = 64 * pow(size, 0.333);
  size_t y = 64 * pow(size, 0.333);
  size_t z = 64 * pow(size, 0.333);
  size_t kernelMillis = 34;

  size_t radius = 3;

  DistributedDomain dd(x, y, z);

  dd.set_radius(radius);

  dd.add_data<float>();
  dd.add_data<float>();
  dd.add_data<float>();
  dd.add_data<float>();

  dd.realize();

  MPI_Barrier(MPI_COMM_WORLD);

  for (size_t iter = 0; iter < 3; ++iter) {
    nvtxRangePush("exchange");
    dd.exchange();
    nvtxRangePop();

    nvtxRangePush("kernels");
    std::this_thread::sleep_for(std::chrono::milliseconds(kernelMillis));
    nvtxRangePop();
  }

  MPI_Finalize();

  return 0;
}
