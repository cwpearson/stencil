#include <chrono>
#include <cmath>

#include <mpi.h>
#include <nvToolsExt.h>

#include "stencil/argparse.hpp"
#include "stencil/cuda_runtime.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int minN = 0;
  int maxN = 27;
  int nIters = 30;
  std::string outpath;

  Parser p;
  p.add_positional(outpath)->required();
  p.add_option(minN, "--min");
  p.add_option(maxN, "--max");
  p.add_option(nIters, "--iters");
  if (!p.parse(argc, argv)) {
    std::cout << p.help() << "\n";
    exit(EXIT_FAILURE);
  }

  char hostname[MPI_MAX_PROCESSOR_NAME] = {0};
  int nameLen;
  MPI_Get_processor_name(hostname, &nameLen);

  std::vector<char> hostnames;
  if (0 == rank) {
    hostnames = std::vector<char>(MPI_MAX_PROCESSOR_NAME * size, 0);
  }

  MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, hostnames.data(),
             MPI_MAX_PROCESSOR_NAME, MPI_BYTE, 0, MPI_COMM_WORLD);

  double elapsed;

  char *buf = new char[1ull << maxN];
  for (int src = 0; src < size; ++src) {
    for (int dst = src + 1; dst < size; ++dst) {
      if (0 == rank) {
        std::cout << &hostnames[MPI_MAX_PROCESSOR_NAME * src] << "-"
                  << &hostnames[MPI_MAX_PROCESSOR_NAME * dst];
      }

      for (int64_t n = minN; n <= maxN; ++n) {
        size_t numBytes = 1ull << n;

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        for (size_t i = 0; i < nIters; ++i) {
          if (src == rank) {
            MPI_Send(buf, numBytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD);
            MPI_Recv(buf, numBytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
          } else if (dst == rank) {
            MPI_Recv(buf, numBytes, MPI_BYTE, src, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buf, numBytes, MPI_BYTE, src, 0, MPI_COMM_WORLD);
          }
        }
        elapsed = MPI_Wtime() - start;

        // need to send elapsed time back to src
        if (src != 0) {
          if (src == rank && 0 != rank) {
            MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
          }
          if (0 == rank) {
            MPI_Recv(&elapsed, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
          }
        } else {
          // src is 0, so elapsed has the right value
        }
        if (0 == rank) {
          std::cout << " " << elapsed;
        }
      }
      if (0 == rank) {
        std::cout << "\n";
      }
    }
  }

  delete[] buf;

  MPI_Finalize();

  return 0;
}
