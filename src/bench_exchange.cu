#include <chrono>
#include <sstream>

#include "argparse/argparse.hpp"
#include "statistics.hpp"
#include "stencil/mpi.hpp"
#include "stencil/stencil.hpp"

/* return a summary of nIters exchanges to rank 0, and the number of bytes
 * exchanged
 */
std::pair<Statistics, uint64_t> bench(const size_t nIters, const Dim3 &extent,
                                      const Radius &radius,
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
  return std::make_pair(stats, dd.halo_exchange_bytes());
}

void report_header() {
  std::cout << "name,count,trimean S,trimean B/sstddev,min,avg,max\n";
}

void report(const std::string &cfg, uint64_t bytes, Statistics &stats) {
  std::cout << std::scientific;
  std::cout << cfg << "," << stats.count() << "," << stats.trimean() << ","
            << bytes / stats.trimean() << "," << stats.stddev() << ","
            << stats.min() << "," << stats.avg() << "," << stats.max() << "\n";
  std::cout << std::defaultfloat;
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
  Dim3 ext(128, 128, 128);

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
  uint64_t bytes;

  if (0 == rank) {
    report_header();
  }

  // positive x-leaning, radius = 2
  radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 2;
  std::tie(stats, bytes) = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    ss << "/px/r2";
    report(ss.str(), bytes, stats);
  }

  // x-only, radius = 2
  radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 2;
  radius.dir(-1, 0, 0) = 2;
  std::tie(stats, bytes) = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    ss << "/x/2";
    report(ss.str(), bytes, stats);
  }

  // faces only, radius = 2
  radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 2;
  radius.dir(-1, 0, 0) = 2;
  radius.dir(0, 1, 0) = 2;
  radius.dir(0, -1, 0) = 2;
  radius.dir(0, 0, 1) = 2;
  radius.dir(0, 0, -1) = 2;
  std::tie(stats, bytes) = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    ss << "/faces/2";
    report(ss.str(), bytes, stats);
  }

  // faces & edges, radius = 2
  radius = Radius::constant(2);
  radius.dir(1, 1, 1) = 0;
  radius.dir(1, 1, -1) = 0;
  radius.dir(1, -1, 1) = 0;
  radius.dir(1, -1, -1) = 0;
  radius.dir(-1, 1, 1) = 0;
  radius.dir(-1, 1, -1) = 0;
  radius.dir(-1, -1, 1) = 0;
  radius.dir(-1, -1, -1) = 0;
  std::tie(stats, bytes) = bench(nIters, ext, radius, gpusPerRank);
  if (0 == rank) {
    std::stringstream ss;
    ss << ext;
    ss << "/"
       << "face&edge/2";
    report(ss.str(), bytes, stats);
  }

  // uniform, radius = 2
  {
    std::stringstream ss;
    ss << ext;
    ss << "/uniform/2";
    nvtxRangePush(ss.str().c_str());
    radius = Radius::constant(2);
    std::tie(stats, bytes) = bench(nIters, ext, radius, gpusPerRank);
    nvtxRangePop();
    if (0 == rank) {
      report(ss.str(), bytes, stats);
    }
  }

#if STENCIL_USE_MPI == 1
  MPI_Finalize();
#endif // STENCIL_USE_MPI == 1

  return 0;
}
