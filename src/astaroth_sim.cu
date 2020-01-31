#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include <cxxopts/cxxopts.hpp>

#include "stencil/stencil.hpp"

int main(int argc, char **argv) {

  cxxopts::Options options("MyProgram", "One line description of MyProgram");
  // clang-format off
  options.add_options()
  ("h,help", "Show help")
  ("remote", "Enable RemoteSender/Recver")
  ("cuda-aware-mpi", "Enable CudaAwareMpiSender/Recver")
  ("colocated", "Enable ColocatedHaloSender/Recver")
  ("peer", "Enable PeerAccessSender")
  ("kernel", "Enable PeerCopySender")
  ("f,file", "File name", cxxopts::value<std::string>());
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result["help"].as<bool>()) {
    std::cerr << options.help();
    exit(EXIT_SUCCESS);
  }

  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double kernelMillis = 50;
  size_t x = 64;
  size_t y = 64;
  size_t z = 64;

  cudaDeviceProp prop;
  CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0));
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
    std::cerr << "WARN: unknown GPU " << prop.name << ", using " << kernelMillis
              << "ms for kernel\n";
  }

  /*
  Table 5
  512^3
  512^3 on Pascal 34.1ms
  512^3 on Volta  20.1ms
  */

  MethodFlags methods = MethodFlags::None;
  if (result["remote"].as<bool>()) {
    methods |= MethodFlags::CudaMpi;
  }
  if (result["cuda-aware-mpi"].as<bool>()) {
    methods |= MethodFlags::CudaMpi;
  }
  if (result["colocated"].as<bool>()) {
    methods |= MethodFlags::CudaMpiColocated;
  }
  if (result["peer"].as<bool>()) {
    methods |= MethodFlags::CudaMemcpyPeer;
  }
  if (result["kernel"].as<bool>()) {
    methods |= MethodFlags::CudaKernel;
  }
  if (MethodFlags::None == methods) {
    methods = MethodFlags::All;
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

    for (size_t iter = 0; iter < 3; ++iter) {
      std::cerr << "exchange\n";
      nvtxRangePush("exchange");
      dd.exchange();
      nvtxRangePop();

      std::cerr << "kernels\n";
      nvtxRangePush("kernels");
      auto dur = std::chrono::duration<double, std::milli>(kernelMillis);
      std::this_thread::sleep_for(dur);
      nvtxRangePop();
    }
  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
