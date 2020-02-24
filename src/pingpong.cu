#include <chrono>
#include <cmath>

#include <mpi.h>
#include <nvToolsExt.h>

#include "stencil/argparse.hpp"
#include "stencil/cuda_runtime.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);


  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int minN = 0;
  int maxN = 27;
  int nIters = 30;
  int ranksPerNode = 1;
  std::string outpath;

  Parser p;
  p.add_positional(ranksPerNode)->required();
  p.add_option(minN, "--min");
  p.add_option(maxN, "--max");
  p.add_option(nIters, "--iters");
  if (!p.parse(argc, argv)) {
    std::cout << p.help() << "\n";
    exit(EXIT_FAILURE);
  }

  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int node = rank / ranksPerNode;
  int ri = rank % ranksPerNode;

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

  char *buf = new char[(1ull << maxN) / ranksPerNode];
  for (int srcNode = 0; srcNode < size / ranksPerNode; ++srcNode) {
    for (int dstNode = srcNode + 1; dstNode < size / ranksPerNode; ++dstNode) {
      if (0 == rank) {
        std::cout << &hostnames[MPI_MAX_PROCESSOR_NAME * srcNode * ranksPerNode] << "-"
                  << &hostnames[MPI_MAX_PROCESSOR_NAME * dstNode * ranksPerNode] << "-" << ranksPerNode;
      }

      for (int64_t n = minN; n <= maxN; ++n) {
        size_t numBytes = (1ull << n) / ranksPerNode;

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        // all ranks on src and dst node pingpong
        for (size_t i = 0; i < nIters; ++i) {
          if (srcNode == node) {
            MPI_Send(buf, numBytes, MPI_BYTE, dstNode * ranksPerNode + ri, 0, MPI_COMM_WORLD);
            MPI_Recv(buf, numBytes, MPI_BYTE, dstNode * ranksPerNode + ri, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
          } else if (dstNode == node) {
            MPI_Recv(buf, numBytes, MPI_BYTE, srcNode * ranksPerNode + ri, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buf, numBytes, MPI_BYTE, srcNode * ranksPerNode + ri, 0, MPI_COMM_WORLD);
          }
        }
        elapsed = MPI_Wtime() - start;

        // get the max time at node 0
	if (0 == rank) {
          MPI_Reduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
          double scratch;
          MPI_Reduce(&elapsed, &scratch, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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
