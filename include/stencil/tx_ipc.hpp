#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

/* \file
Handle synchronization between MPI ranks on the same node.

Also, provides a CUDA event that is shared between the IpcSender and IpcRecver, but does not directly use that event.
The object that uses the IpcSender and Recver may use the shared event to help coordinate

*/

/*
 */
class IpcSender {

  MPI_Request evtReq_; // async send of event to IpcRecvter
  MPI_Request dstReq_; // async recv of destination CUDA device id
  MPI_Request notReq_; // async send of notify() message

  int srcRank_, dstRank_;
  int srcDom_, dstDom_;
  int srcDev_, dstDev_;
  cudaEvent_t event_; // event to synchronize without MPI
  cudaIpcEventHandle_t eventHandle_;

public:
  IpcSender();
  IpcSender(int srcRank, int srcDom, // domain ID
            int dstRank, int dstDom, // domain ID
            int srcDev               // cuda ID
  );
  ~IpcSender();

  // start async send of event and recv of destination device ID
  void async_prepare();

  // block until we have recved the destination device ID and sent the event
  void wait_prepare();

  // async send notify message
  void async_notify();

  // block until notify is complete
  void wait_notify();

  const cudaEvent_t &event() const noexcept { return event_; }
  int dst_dev() const noexcept { return dstDev_; }
};

/*
 */
class IpcRecver {

  MPI_Request evtReq_; // async recv of event from IpcSender
  MPI_Request dstReq_; // async send of my CUDA device id
  MPI_Request notReq_; // async recv of notify() message

  int srcRank_, dstRank_;
  int srcDom_, dstDom_; // domain ID
  int srcDev_, dstDev_; // cuda ID
  cudaEvent_t event_;   // to synchronize with sender
  cudaIpcEventHandle_t eventHandle_;

public:
  IpcRecver();
  IpcRecver(int srcRank, int srcDom, // domain ID
            int dstRank, int dstDom, // domain ID
            int dstDev               // cuda ID
  );
  ~IpcRecver();

  // start async send of destination device and recv of event
  void async_prepare();

  // block until we have recved the event and sent the destination device
  void wait_prepare();

  // start listening for notification
  void async_listen();

  // true if notification has been recieved
  bool test_listen();

  // block until got notification
  void wait_listen();

  const cudaEvent_t &event() const noexcept { return event_; }
};