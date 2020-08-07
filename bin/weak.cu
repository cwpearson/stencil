#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"
#include "stencil/stencil.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numGpus;
  int numNodes;
  {
    MpiTopology topo(MPI_COMM_WORLD);
    numNodes = topo.size() / topo.colocated_size();
    numGpus = size / topo.colocated_size() * devCount;
  }

  size_t x = 512;
  size_t y = 512;
  size_t z = 512;

  int nIters = 30;
  bool useNaivePlacement = false;
  bool useKernel = false;
  bool usePeer = false;
  bool useColo = false;
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  bool useCudaAware = false;
#endif
  bool useStaged = false;

  argparse::Parser p;
  p.no_unrecognized();
  p.add_positional(x)->required();
  p.add_positional(y)->required();
  p.add_positional(z)->required();
  p.add_positional(nIters)->required();
  p.add_flag(useKernel, "--kernel");
  p.add_flag(usePeer, "--peer");
  p.add_flag(useColo, "--colo");
  p.add_flag(useNaivePlacement, "--naive");
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  p.add_flag(useCudaAware, "--cuda-aware");
#endif
  p.add_flag(useStaged, "--staged");
  if (!p.parse(argc, argv)) {
    std::cout << p.help() << "\n";
    exit(EXIT_FAILURE);
  }

  x = size_t(double(x) * pow(double(numGpus), 0.33333) + 0.5); // round to nearest
  y = size_t(double(y) * pow(double(numGpus), 0.33333) + 0.5);
  z = size_t(double(z) * pow(double(numGpus), 0.33333) + 0.5);

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

    std::cout << numGpus << " subdomains " << size << " ranks: " << x << "," << y << "," << z << "=" << x * y * z
              << "\n";
    if (useNaivePlacement) {
      std::cerr << "naive placement\n";
    } else {
      std::cerr << "node-aware placement\n";
    }
    if ((methods & MethodFlags::CudaMpi) != MethodFlags::None) {
      std::cerr << "CudaMpi enabled\n";
    }
    if ((methods & MethodFlags::CudaAwareMpi) != MethodFlags::None) {
      std::cerr << "CudaAwareMpi enabled\n";
    }
    if ((methods & MethodFlags::CudaMpiColocated) != MethodFlags::None) {
      std::cerr << "CudaMpiColocated enabled\n";
    }
    if ((methods & MethodFlags::CudaMemcpyPeer) != MethodFlags::None) {
      std::cerr << "CudaMemcpyPeer enabled\n";
    }
    if ((methods & MethodFlags::CudaKernel) != MethodFlags::None) {
      std::cerr << "CudaKernel enabled\n";
    }
  }

  {
    size_t radius = 3;

    DistributedDomain dd(x, y, z);

    dd.set_methods(methods);
    dd.set_radius(radius);
    if (useNaivePlacement) {
      dd.set_placement(PlacementStrategy::Trivial);
    } else {
      dd.set_placement(PlacementStrategy::NodeAware);
    }

    dd.add_data<float>("d0");
    dd.add_data<float>("d1");
    dd.add_data<float>("d2");
    dd.add_data<float>("d3");

    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < nIters; ++iter) {
      if (0 == rank) {
        std::cerr << "exchange " << iter << "\n";
      }
      nvtxRangePush("exchange");
      dd.exchange();
      nvtxRangePop();
      dd.swap();
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if STENCIL_MEASURE_TIME == 1
    if (0 == rank) {
      std::string methodStr;
      if (methods && MethodFlags::CudaMpi) {
        methodStr += methodStr.empty() ? "" : ",";
        methodStr += "staged";
      }
      if (methods && MethodFlags::CudaAwareMpi) {
        methodStr += methodStr.empty() ? "" : "/";
        methodStr += "cuda-aware";
      }
      if (methods && MethodFlags::CudaMpiColocated) {
        methodStr += methodStr.empty() ? "" : "/";
        methodStr += "colo";
      }
      if (methods && MethodFlags::CudaMemcpyPeer) {
        methodStr += methodStr.empty() ? "" : "/";
        methodStr += "peer";
      }
      if (methods && MethodFlags::CudaKernel) {
        methodStr += methodStr.empty() ? "" : "/";
        methodStr += "kernel";
      }
      if (methods == MethodFlags::All) {
        methodStr += methodStr.empty() ? "" : "/";
        methodStr += "all";
      }

      // clang-format off
      // same as strong.cu
      // header should be
      // bin,config,x,y,z,s,MPI (B),Colocated (B),cudaMemcpyPeer (B),direct (B)iters,gpus,nodes,ranks,mpi_topo,node_gpus,peer_en,placement,realize,plan,create,exchange,swap,
      // clang-format on
      printf("weak,%s,%lu,%lu,%lu,%lu," // s
             "%lu,%lu,%lu,%lu,"         // <- exchange bytes
             "%d,%d,%d,%d,%e,%e,%e,%e,%e,%e,%e,%e,%e\n",
             methodStr.c_str(), x, y, z, x * y * z, dd.exchange_bytes_for_method(MethodFlags::CudaMpi),
             dd.exchange_bytes_for_method(MethodFlags::CudaMpiColocated),
             dd.exchange_bytes_for_method(MethodFlags::CudaMemcpyPeer),
             dd.exchange_bytes_for_method(MethodFlags::CudaKernel), nIters, numGpus, numNodes, size, dd.timeMpiTopo_,
             dd.timeNodeGpus_, dd.timePeerEn_, dd.timePlacement_, dd.timeRealize_, dd.timePlan_, dd.timeCreate_,
             dd.timeExchange_, dd.timeSwap_);
    }
#endif // STENCIL_MEASURE_TIME

  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
