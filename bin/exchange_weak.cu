/* measure purely total exchange time
 */

#include <cmath>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"
#include "stencil/stencil.hpp"

#include "statistics.hpp"

typedef std::chrono::duration<double> Dur;

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size = mpi::world_size();
  int rank = mpi::world_rank();
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numSubdoms;
  int numNodes;
  {
    MpiTopology topo(MPI_COMM_WORLD);

    numNodes = size / topo.colocated_size();
    int ranksPerNode = topo.colocated_size();
    int subdomsPerRank = topo.colocated_size() > devCount ? 1 : devCount / topo.colocated_size();

    if (0 == rank) {
      std::cerr << numNodes << " nodes\n";
      std::cerr << ranksPerNode << " ranks / node\n";
      std::cerr << subdomsPerRank << " sd / rank\n";
    }

    numSubdoms = numNodes * ranksPerNode * subdomsPerRank;
  }

  if (0 == rank) {
    std::cerr << "assuming " << numSubdoms << " subdomains\n";
  }

  size_t x = 512;
  size_t y = 512;
  size_t z = 512;

  int nIters = 30;
  bool useNaivePlacement = false;
  std::string prefix;
  bool useKernel = false;
  bool usePeer = false;
  bool useColoPmu = false;
  bool useColoDa = false;
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
  p.add_option(prefix, "--prefix");
  p.add_flag(useKernel, "--kernel");
  p.add_flag(usePeer, "--peer");
  p.add_flag(useColoPmu, "--colo-pmu");
  p.add_flag(useColoDa, "--colo-da");
  p.add_flag(useStaged, "--staged");
  p.add_flag(useNaivePlacement, "--naive");
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  p.add_flag(useCudaAware, "--cuda-aware");
#endif

  if (!p.parse(argc, argv)) {
    std::cout << p.help() << "\n";
    exit(EXIT_FAILURE);
  }

  /*
  For a single node, scaling the compute domain as a cube while trying to keep gridpoints/GPU constant
    cause weird-to-understand scaling performance for GPU kernels because the aspect ratio affects the GPU kernel
    performance.
  So, we just recursively mulitply the prime factors into the smallest dimensions to keep the shape constant.

  For multiple nodes, it's generally impossible to guarantee a particular size of each subdomain due to the hierarchical
  partitioning, so we just scale the whole compute region to keep the gridpoints / subdomain roughly constant.
  */
  if (1 == numNodes) {
    std::vector<int64_t> pfs = prime_factors(numSubdoms);
    for (auto pf : pfs) {
      if (x <= y && x <= z) {
        x *= pf;
      } else if (y <= z) {
        y *= pf;
      } else {
        z *= pf;
      }
    }
  } else {
    x *= std::pow(double(numSubdoms), 1.0 / 3);
    y *= std::pow(double(numSubdoms), 1.0 / 3);
    z *= std::pow(double(numSubdoms), 1.0 / 3);
  }

  Method methods = Method::None;
  if (useStaged) {
    methods = Method::CudaMpi;
  }
  if (useColoPmu) {
    methods |= Method::ColoPackMemcpyUnpack;
  }
  if (useColoDa) {
    methods |= Method::ColoDirectAccess;
  }
  if (usePeer) {
    methods |= Method::CudaMemcpyPeer;
  }
  if (useKernel) {
    methods |= Method::CudaKernel;
  }
  if (methods == Method::None) {
    methods = Method::Default;
  }

  if (0 == rank) {
#ifndef NDEBUG
    std::cout << "ERR: not release mode\n";
    std::cerr << "ERR: not release mode\n";
    exit(-1);
#endif
#ifdef STENCIL_EXCHANGE_STATS
    std::cout << "WARN: detailed time measurement\n";
    std::cerr << "WARN: detailed time measurement\n";
#endif
#ifndef STENCIL_SETUP_STATS
    std::cout << "ERR: not tracking stats\n";
    std::cerr << "ERR: not tracking stats\n";
    exit(-1);
#endif
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

    dd.set_output_prefix(prefix);

    dd.realize();

    Statistics stats;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < nIters; ++iter) {
      if (0 == rank) {
        std::cerr << "exchange " << iter << "\n";
      }
      double elapsed = MPI_Wtime();
      dd.exchange();
      elapsed = MPI_Wtime() - elapsed;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      stats.insert(elapsed);
    }

#ifdef STENCIL_SETUP_STATS
    if (0 == rank) {
      const std::string methodStr = to_string(methods);

      // clang-format off
      // same as strong.cu
      // header should be
      // bin,config,naive,x,y,z,s,ldx,ldy,ldz,MPI (B),Colocated (B),cudaMemcpyPeer (B),direct (B),iters,gpus,nodes,ranks,trimean (s)
      // clang-format on
      printf("exchange,%s,%d,%lu,%lu,%lu,%lu," // s
             "%lu,%lu,%lu,"                    // ldx ldy ldz
             "%lu,%lu,%lu,%lu,"                // <- exchange bytes
             "%d,%d,%d,%d,%e\n",
             methodStr.c_str(), useNaivePlacement, x, y, z, x * y * z, dd.domains()[0].size().x,
             dd.domains()[0].size().y, dd.domains()[0].size().z, dd.exchange_bytes_for_method(Method::CudaMpi),
             dd.exchange_bytes_for_method(Method::ColoPackMemcpyUnpack),
             dd.exchange_bytes_for_method(Method::CudaMemcpyPeer), dd.exchange_bytes_for_method(Method::CudaKernel),
             nIters, numSubdoms, numNodes, size, stats.trimean());
    }
#endif // STENCIL_SETUP_STATS

  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  std::cerr << "cuda=" << timers::cudaRuntime.get_elapsed() / nIters << "\n";
  std::cerr << "mpi=" << timers::mpi.get_elapsed() / nIters << "\n";

  return 0;
}
