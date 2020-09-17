/* measure purely total exchange time
 */

#include <cmath>

#include <nvToolsExt.h>

#include "statistics.hpp"

#include "argparse/argparse.hpp"
#include "stencil/stencil.hpp"

typedef std::chrono::duration<double> Dur;

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size;
  int rank;
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

  std::string prefix;

  int nIters = 30;
  bool useNaivePlacement = false;
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
  p.add_flag(useKernel, "--kernel");
  p.add_flag(usePeer, "--peer");
  p.add_flag(useColoPmu, "--colo-pmu");
  p.add_flag(useColoDa, "--colo-da");
  p.add_flag(useNaivePlacement, "--naive");
  p.add_option(prefix, "--prefix");
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  p.add_flag(useCudaAware, "--cuda-aware");
#endif
  p.add_flag(useStaged, "--staged");
  if (!p.parse(argc, argv)) {
    std::cout << p.help() << "\n";
    exit(EXIT_FAILURE);
  }

  Method methods = Method::None;
  if (useStaged) {
    methods = Method::CudaMpi;
  }
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  if (useCudaAware) {
    methods = Method::CudaAwareMpi;
  }
#endif
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
    methods = Method::All;
  }

  if (0 == rank) {
#ifndef NDEBUG
    std::cout << "WARN: not release mode\n";
    std::cerr << "WARN: not release mode\n";
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
    dd.set_output_prefix(prefix);
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
      // bin,config,naive,x,y,z,s,ldx,ldy,ldz,MPI (B),Colocated (B),cudaMemcpyPeer (B),direct (B),iters,sds,nodes,ranks,exchange trimean (s)
      // clang-format on
      printf("exchange,%s,%d,%lu,%lu,%lu,%lu," // s
             "%lu,%lu,%lu,"                    // ldx,ldy,ldz
             "%lu,%lu,%lu,%lu,"                // different exchange bytes
             "%d,%d,%d,%d,%e\n",
             methodStr.c_str(), useNaivePlacement, x, y, z, x * y * z, dd.domains()[0].size().x,
             dd.domains()[0].size().y, dd.domains()[0].size().z, dd.exchange_bytes_for_method(Method::CudaMpi),
             dd.exchange_bytes_for_method(Method::ColoPackMemcpyUnpack),
             dd.exchange_bytes_for_method(Method::CudaMemcpyPeer),
             dd.exchange_bytes_for_method(Method::CudaKernel), nIters, numSubdoms, numNodes, size,
             stats.trimean());
    }
#endif // STENCIL_EXCHANGE_STATS

  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
