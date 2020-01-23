#include "catch2/catch.hpp"

#include <mpi.h>

#include "stencil/cuda_runtime.hpp"
#include "stencil/tx_cuda.cuh"

TEMPLATE_TEST_CASE("colocated", "[mpi][cuda][template]", int32_t, int64_t) {

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
  int srcGPU = 0;
  int dstGPU = 0;
  int srcDev = 0;
  int dstDev = 0;

  const size_t n = 100;

  TestType *buf0 = nullptr;
  TestType *buf1 = nullptr;

  INFO("allocate bufs");
  CUDA_RUNTIME(cudaSetDevice(srcDev));
  CUDA_RUNTIME(cudaMalloc(&buf0, n * sizeof(TestType)));
  CUDA_RUNTIME(cudaSetDevice(dstDev));
  CUDA_RUNTIME(cudaMalloc(&buf1, n * sizeof(TestType)));

  INFO("ctors");
  ColocatedDeviceSender sender(myRank, srcGPU, dstRank, dstGPU, srcDev);
  ColocatedDeviceRecver recver(srcRank, srcGPU, myRank, dstGPU, dstDev);

  INFO("sender.start_prepare");
  sender.start_prepare(n * sizeof(TestType));
  INFO("recver.start_prepare");
  recver.start_prepare(buf1, n * sizeof(TestType));

  INFO("sender.finish_prepare");
  sender.finish_prepare();
  INFO("recver.finish_prepare");
  recver.finish_prepare();

  INFO("streams");
  RcStream sendStream(srcDev);
  RcStream recvStream(dstDev);

  INFO("send");
  sender.send(buf0, sendStream);

  INFO("wait");
  sender.wait();
  recver.wait(recvStream);

  INFO("cuda sync");
  CUDA_RUNTIME(cudaStreamSynchronize(sendStream));
  CUDA_RUNTIME(cudaStreamSynchronize(recvStream));

  CUDA_RUNTIME(cudaFree(buf0));
  CUDA_RUNTIME(cudaFree(buf1));
}