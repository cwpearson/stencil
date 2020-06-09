#include <chrono>
#include <sstream>

#define STENCIL_OUTPUT_LEVEL 2

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

int main(int argc, char **argv) {

// Initialize MPI
#if STENCIL_USE_MPI == 1
  MPI_Init(&argc, &argv);
#endif // STENCIL_USE_MPI == 1

  int rank = mpi::world_rank();
  int size = mpi::world_size();

#if STENCIL_MEASURE_TIME != 0
  std::cerr << "WARN: compiled with STENCIL_MEASURE_TIME != 0. Extra overhead may be present.\n";
#endif

  // CLI parameters
  int nIters = 30;

  // parse CLI arguments
  Parser p;
  p.add_option(nIters, "--iters");
  if (!p.parse(argc, argv)) {
    if (0 == rank) {
      std::cout << p.help();
    }
    exit(EXIT_FAILURE);
  }
  srand(time(NULL) + rank);

  // benchmark parameters
  Radius radius;   // stencil radius
  Dim3 ext;        // compute domain size
  int gpusPerRank; // the number of GPUs per rank

  // benchmark results
  Statistics stats;

  ext = Dim3(64, 64, 64);
  radius = Radius::constant(1);
  stats = bench(nIters, ext, radius, gpusPerRank);
  std::cout << stats.count() << " runs: " << "min/avg/max=" << stats.min() << "/" << stats.avg() << "/" << stats.max() << "\n";

#if STENCIL_USE_MPI == 1
  MPI_Finalize();
#endif // STENCIL_USE_MPI == 1

  return 0;

}
