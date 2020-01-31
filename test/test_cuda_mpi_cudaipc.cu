#include "catch2/catch.hpp"

#include <mpi.h>

#include "stencil/cuda_runtime.hpp"
#include "stencil/tx_cuda.cuh"

TEST_CASE("cudaipc", "[mpi][cuda]") {

  int myRank;
  int worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  if (myRank == 0) {
    // create event
    cudaEvent_t event;
    CUDA_RUNTIME(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));

    // create handle
    cudaIpcEventHandle_t handle;
    CUDA_RUNTIME(cudaIpcGetEventHandle(&handle, event));

    // send handle to rank 1
    MPI_Send(&handle, sizeof(handle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

  } else if (1 == myRank) {

    // recv handle from rank 0
    cudaIpcEventHandle_t handle;
    MPI_Recv(&handle, sizeof(handle), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Create event from handle
    cudaEvent_t event;
    CUDA_RUNTIME(cudaIpcOpenEventHandle(&event, handle));

    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_RUNTIME(cudaEventDestroy(event));
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
}