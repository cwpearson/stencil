#include "stencil/tx_ipc.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"
#include "stencil/tx_common.hpp"

#include <cassert>

IpcSender::IpcSender() : event_(0) {}
IpcSender::IpcSender(int srcRank, int srcDom, int dstRank, int dstDom, int srcDev)
    : srcRank_(srcRank), dstRank_(dstRank), srcDom_(srcDom), dstDom_(dstDom), srcDev_(srcDev), dstDev_(-1), event_(0) {}

IpcSender::~IpcSender() {
  if (event_) {
    CUDA_RUNTIME(cudaEventDestroy(event_));
  }
}

void IpcSender::async_prepare() {

  // create an event and associated handle
  CUDA_RUNTIME(cudaSetDevice(srcDev_));
  CUDA_RUNTIME(cudaEventCreate(&event_, cudaEventInterprocess | cudaEventDisableTiming));
  CUDA_RUNTIME(cudaIpcGetEventHandle(&eventHandle_, event_));

  // send the event handle
  LOG_SPEW("Isend handle");
  const int evtTag = make_tag<MsgKind::ColocatedEvt>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Isend(&eventHandle_, sizeof(eventHandle_), MPI_BYTE, dstRank_, evtTag, MPI_COMM_WORLD, &evtReq_);

  // Retrieve the destination device id
  LOG_SPEW("Irecv destination device");
  const int devTag = make_tag<MsgKind::ColocatedDev>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Irecv(&dstDev_, 1, MPI_INT, dstRank_, devTag, MPI_COMM_WORLD, &dstReq_);

  LOG_SPEW("done IpcSender::async_prepare");
}

void IpcSender::wait_prepare() {
  // block until we have recved the device ID
  LOG_SPEW("sender: wait for destination device");
  MPI_Wait(&dstReq_, MPI_STATUS_IGNORE);

  // block until we have sent event handle
  LOG_SPEW("sender: wait to send event");
  MPI_Wait(&evtReq_, MPI_STATUS_IGNORE);

  LOG_SPEW("done IpcSender::wait_prepare");
}

void IpcSender::async_notify() {
  const int notTag = make_tag<MsgKind::ColocatedNotify>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Isend(&junk_, 1, MPI_BYTE, dstRank_, notTag, MPI_COMM_WORLD, &notReq_);
}

void IpcSender::wait_notify() { MPI_Wait(&notReq_, MPI_STATUS_IGNORE); }

IpcRecver::IpcRecver() : event_(0) {}
IpcRecver::IpcRecver(int srcRank, int srcDom, int dstRank, int dstDom, int dstDev)
    : srcRank_(srcRank), dstRank_(dstRank), srcDom_(srcDom), dstDom_(dstDom), srcDev_(-1), dstDev_(dstDev), event_(0) {}

IpcRecver::~IpcRecver() {
  if (event_) {
    CUDA_RUNTIME(cudaEventDestroy(event_));
  }
}

void IpcRecver::async_prepare() {
  const int payload = ((srcDom_ & 0xFF) << 8) | (dstDom_ & 0xFF);

  // recv the event handle
  const int evtTag = make_tag<MsgKind::ColocatedEvt>(payload);
  MPI_Irecv(&eventHandle_, sizeof(eventHandle_), MPI_BYTE, srcRank_, evtTag, MPI_COMM_WORLD, &evtReq_);

  // Send the CUDA device id to the ColocatedSender
  const int dstTag = make_tag<MsgKind::ColocatedDev>(payload);
  MPI_Isend(&dstDev_, 1, MPI_INT, srcRank_, dstTag, MPI_COMM_WORLD, &dstReq_);
}

void IpcRecver::wait_prepare() {
  // wait to recv the event
  MPI_Wait(&evtReq_, MPI_STATUS_IGNORE);

  // convert event handle to event
  CUDA_RUNTIME(cudaSetDevice(dstDev_));
  CUDA_RUNTIME(cudaIpcOpenEventHandle(&event_, eventHandle_));

  // wait to send the mem handle and the CUDA device ID
  MPI_Wait(&dstReq_, MPI_STATUS_IGNORE);
}

void IpcRecver::async_listen() {
  // wait to recv the event
  const int notTag = make_tag<MsgKind::ColocatedNotify>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Irecv(&junk_, 1, MPI_INT, srcRank_, notTag, MPI_COMM_WORLD, &notReq_);
}

bool IpcRecver::test_listen() {
  int flag;
  MPI_Test(&notReq_, &flag, MPI_STATUS_IGNORE);
  return flag;
}

void IpcRecver::wait_listen() { MPI_Wait(&notReq_, MPI_STATUS_IGNORE); }