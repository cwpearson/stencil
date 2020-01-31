#include "catch2/catch.hpp"

#include <mpi.h>

#include "stencil/cuda_runtime.hpp"
#include "stencil/tx_cuda.cuh"

TEMPLATE_TEST_CASE("cuda-aware-mpi", "[mpi][cuda][template]", int32_t, int64_t) {

#if STENCIL_USE_CUDA_AWARE_MPI == 1
  int myRank;
  int worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // recv from left
  int srcRank = myRank - 1;
  if (srcRank < 0) {
    srcRank = worldSize - 1;
  }
  // send right
  int dstRank = (myRank + 1) % worldSize;

  const size_t n = 100;

  TestType *buf0 = nullptr;
  TestType *buf1 = nullptr;

  size_t numBytes = n * sizeof(TestType);

  INFO("allocate bufs");
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&buf0, numBytes));
  CUDA_RUNTIME(cudaMemset(buf0, 0x1, numBytes));
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&buf1, numBytes));
  CUDA_RUNTIME(cudaMemset(buf1, 0x00, numBytes));

  MPI_Request sreq, rreq;
  MPI_Isend(buf0, numBytes, MPI_BYTE, dstRank, 0, MPI_COMM_WORLD, &sreq);
  MPI_Irecv(buf1, numBytes, MPI_BYTE, srcRank, 0, MPI_COMM_WORLD, &rreq);

  MPI_Wait(&sreq, MPI_STATUS_IGNORE);
  MPI_Wait(&rreq, MPI_STATUS_IGNORE);

  std::vector<char> host(numBytes);
  CUDA_RUNTIME(cudaMemcpy(host.data(), buf1, numBytes, cudaMemcpyDeviceToHost));
  REQUIRE(host[0] == char(0x1));

  CUDA_RUNTIME(cudaFree(buf0));
  CUDA_RUNTIME(cudaFree(buf1));
#endif
}