#include <chrono>
#include <sstream>

#include "argparse/argparse.hpp"
#include "statistics.hpp"
#include "stencil/mpi.hpp"
#include "stencil/stencil.hpp"

/* return a summary of nIters exchanges to rank 0
 */
Statistics bench(const size_t nIters, const Dim3 &extent, const Radius &radius,
                 const size_t gpusPerRank) {

  int rank = mpi::world_rank();

  // stencil quantities
  typedef float Q1;

  // configure distributed stencil
  DistributedDomain dd(extent.x, extent.y, extent.z);
  dd.set_radius(radius);
  dd.add_data<Q1>("d0");
  dd.set_methods(MethodFlags::CudaMpi);

  // create distributed stencil
  dd.realize();

  // track execution time statistics
  Statistics stats;

  for (size_t i = 0; i < nIters; ++i) {

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    // to measure
    dd.exchange();
    dd.swap();
    // done measure
    double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank) {
      stats.insert(elapsed);
    }
  }
  return stats;
}


void report(const std::string &cfg, Statistics &stats) {
  std::cout << cfg << " " << stats.count() << " runs: "
  << stats.trimean() << "+-" << stats.stddev() << " (min/avg/max=" << stats.min() << "/" << stats.avg() << "/"
  << stats.max() << ")\n";
}

int main(int argc, char **argv) {

// Initialize MPI
#if STENCIL_USE_MPI == 1
  MPI_Init(&argc, &argv);
#endif // STENCIL_USE_MPI == 1

  int rank = mpi::world_rank();
  int size = mpi::world_size();

#if STENCIL_MEASURE_TIME != 0
  std::cerr << "WARN: compiled with STENCIL_MEASURE_TIME != 0. Extra overhead "
               "may be present.\n";
#endif

  // CLI parameters
  int nIters = 30;
  Dim3 ext(128,128,128);

  // parse CLI arguments
  argparse::Parser p("benchmark stencil library exchange");
  p.add_option(nIters, "--iters")->help("number of iterations to measure");
  p.add_option(ext.x, "--x")->help("x extent of compute domain");
  p.add_option(ext.y, "--y")->help("y extent of compute domain");
  p.add_option(ext.z, "--z")->help("z extent of compute domain");
  if (!p.parse(argc, argv)) {
    if (0 == rank) {
      std::cout << p.help();
    }
    exit(EXIT_FAILURE);
  }
  if (p.need_help()) {
    if (0 == rank) {
      std::cout << p.help();
    }
    exit(EXIT_SUCCESS);
  }

  srand(time(NULL) + rank);

  // benchmark parameters
  Radius radius;   // stencil radius
  int gpusPerRank; // the number of GPUs per rank

  // benchmark results
  Statistics stats;



  // positive x-leaning, radius = 2
  radius = Radius::constant(0);
  radius.dir(1,0,0) = 2;
  stats = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    report(ss.str(), stats);
  }

  // x-only, radius = 2
  radius = Radius::constant(0);
  radius.dir(1,0,0) = 2;
  radius.dir(-1,0,0) = 2;
  stats = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    report(ss.str(), stats);
  }

  // faces only, radius = 2
  radius = Radius::constant(0);
  radius.dir(1,0,0) = 2;
  radius.dir(-1,0,0) = 2;
  radius.dir(0,1,0) = 2;
  radius.dir(0,-1,0) = 2;
  radius.dir(0,0,1) = 2;
  radius.dir(0,0,-1) = 2;
  stats = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    report(ss.str(), stats);
  }

  // faces & edges, radius = 2
  radius = Radius::constant(2);
  radius.dir(1,1,1) = 0;
  radius.dir(1,1,-1) = 0;
  radius.dir(1,-1,1) = 0;
  radius.dir(1,-1,-1) = 0;
  radius.dir(-1,1,1) = 0;
  radius.dir(-1,1,-1) = 0;
  radius.dir(-1,-1,1) = 0;
  radius.dir(-1,-1,-1) = 0;
  stats = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    report(ss.str(), stats);
  }

  // uniform, radius = 2
  radius = Radius::constant(2);
  stats = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    report(ss.str(), stats);
  }

#if STENCIL_USE_MPI == 1
  MPI_Finalize();
#endif // STENCIL_USE_MPI == 1

  return 0;
}
