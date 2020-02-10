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
  ("trivial", "Skip node-aware placement")
  ("x", "x dim", cxxopts::value<int>()->default_value("512"))
  ("y", "y dim", cxxopts::value<int>()->default_value("512"))
  ("z", "z dim", cxxopts::value<int>()->default_value("512"))
  ("f,file", "File name", cxxopts::value<std::string>());
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result["help"].as<bool>()) {
    std::cerr << options.help();
    exit(EXIT_SUCCESS);
  }

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

  if (0 == rank) {
    std::cout << "assuming " << numSubdoms << " subdomains\n";
  }

  double kernelMillis = 50;
  size_t x = result["x"].as<int>();
  size_t y = result["y"].as<int>(); 
  size_t z = result["z"].as<int>();

  cudaDeviceProp prop;
  CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0));
  if (std::string("Tesla V100-SXM2-32GB") == prop.name) {
    kernelMillis = 20.1;
  } else if (std::string("Tesla P100-SXM2-16GB") == prop.name) {
    kernelMillis = 34.1;
  } else {
    if (0 == rank) {
      std::cerr << "WARN: unknown GPU " << prop.name << ", using "
                << kernelMillis << "ms for kernel\n";
    }
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

  PlacementStrategy strategy = PlacementStrategy::NodeAware;
  if (result["trivial"].as<bool>()) {
    strategy = PlacementStrategy::Trivial;
  }

  if (0 == rank) {
    std::cout << "domain: " << x << "," << y << "," << z << "\n";
  }

  {
    size_t radius = 3;

    DistributedDomain dd(x, y, z);

    dd.set_methods(methods);
    dd.set_radius(radius);
    dd.set_placement(strategy);

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
