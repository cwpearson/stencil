#include "catch2/catch.hpp"

#include <mpi.h>

#include "stencil/mpi.hpp"
#include "stencil/cuda_runtime.hpp"
#include "stencil/tx_ipc.hpp"

TEMPLATE_TEST_CASE("tx_ipc", "[mpi][cuda][template]", int32_t, int64_t) {

  const int myRank = mpi::world_rank();
  const int worldSize = mpi::world_size();

  // two colocated senders cannot have all of the same
  // srcRank, dstRank, and dstGPU. MPI tag is only generated from dstGPU
  REQUIRE(worldSize >= 2);

  // recv from left
  int srcRank = myRank - 1;
  if (srcRank < 0) {
    srcRank = worldSize - 1;
  }
  // send right
  int dstRank = (myRank + 1) % worldSize;
  int srcDom = 0;
  int dstDom = 0;
  int srcDev = 0;
  int dstDev = 0;

  INFO("ctors");
  IpcSender sender(myRank, srcDom, dstRank, dstDom, srcDev);
  IpcRecver recver(srcRank, srcDom, myRank, dstDom, dstDev);

  INFO("sender.async_prepare");
  sender.async_prepare();
  INFO("recver.async_prepare");
  recver.async_prepare();

  INFO("sender.wait_prepare");
  sender.wait_prepare();
  INFO("recver.wait_prepare");
  recver.wait_prepare();

  INFO("sender.async_notify");
  sender.async_notify();
  INFO("recver.async_listen");
  recver.async_listen();

  INFO("sender.wait_notify");
  sender.wait_notify();
  INFO("recver.wait_listen");
  recver.wait_listen();
}