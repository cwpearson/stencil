#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "stencil/argparse.hpp"
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

  size_t x = 512;
  size_t y = 512;
  size_t z = 512;

  int nIters = 30;
  bool useKernel = false;
  bool usePeer = false;
  bool useColo = false;
  #if STENCIL_USE_CUDA_AWARE_MPI == 1
  bool useCudaAware = false;
  #endif
  bool useStaged = false;

  Parser p;
  p.add_positional(x)->required();
  p.add_positional(y)->required();
  p.add_positional(z)->required();
  p.add_positional(nIters)->required();
  p.add_flag(useKernel, "--kernel");
  p.add_flag(usePeer, "--peer");
  p.add_flag(useColo, "--colo");
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  p.add_flag(useCudaAware, "--cuda-aware");
#endif
  p.add_flag(useStaged, "--staged");
  p.parse(argc, argv);

  MethodFlags methods = MethodFlags::None;
  if (useStaged) {
    methods = MethodFlags::CudaMpi;
  }
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  if (useCudaAware) {
    methods = MethodFlags::CudaAwareMpi;
  }
#endif
  if (useColo) {
    methods |= MethodFlags::CudaMpiColocated;
  }
  if (usePeer) {
    methods |= MethodFlags::CudaMemcpyPeer;
  }
  if (useKernel) {
    methods |= MethodFlags::CudaKernel;
  }
  if (methods == MethodFlags::None) {
    methods = MethodFlags::All;
  }

  if (0 == rank) {
#ifndef NDEBUG
    std::cout << "WARN: not release mode\n";
    std::cerr << "WARN: not release mode\n";
#endif

    std::cout << numSubdoms << " subdomains " << size << " ranks: " << x << ","
              << y << "," << z << "=" << x * y * z << "\n";
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
    dd.set_placement(PlacementStrategy::NodeAware);

    dd.add_data<float>();
    dd.add_data<float>();
    dd.add_data<float>();
    dd.add_data<float>();

    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t iter = 0; iter < nIters; ++iter) {
      std::cerr << "exchange\n";
      nvtxRangePush("exchange");
      dd.exchange();
      nvtxRangePop();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    #if STENCIL_TIME == 1
        if (0 == rank) {
          printf("mpi_topo %f node_gpus %f peer_en %f placement %f realize %f plan "
                 "%f create %f exchange %f\n",
                 dd.timeMpiTopo_, dd.timeNodeGpus_, dd.timePeerEn_, dd.timePlacement_, dd.timeRealize_, dd.timePlan_, dd.timeCreate_, dd.timeExchange_);
        }
    #endif // STENCIL_TIME


  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
