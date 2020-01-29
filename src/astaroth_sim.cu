#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "stencil/stencil.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double kernelMillis = 50;
  size_t x = 64;
  size_t y = 64;
  size_t z = 64;

  cudaDeviceProp prop;
  CUDA_RUNTIME(cudaGetDeviceProperties ( &prop, 0 ));
  if (std::string("Tesla V100-SXM2-32GB") == prop.name) {
    kernelMillis = 20.1;
    x = 512 * pow(size, 0.333);
    y = 512 * pow(size, 0.333);
    z = 512 * pow(size, 0.333);
  } else if (std::string("Tesla P100-SXM2-16GB") == prop.name) {
    kernelMillis = 34.1;
    x = 512 * pow(size, 0.333);
    y = 512 * pow(size, 0.333);
    z = 512 * pow(size, 0.333);
  } else {
    std::cerr << "WARN: unknown GPU " << prop.name << ", using " << kernelMillis << "ms for kernel\n";
  }


  /*
  Table 5
  512^3
  512^3 on Pascal 34.1ms
  512^3 on Volta  20.1ms
  */

  {
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
    auto dur = std::chrono::duration<double, std::milli>(kernelMillis);
    std::this_thread::sleep_for(dur);
    nvtxRangePop();
  }
} // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
