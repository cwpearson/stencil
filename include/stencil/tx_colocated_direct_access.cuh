#pragma once

#include "stencil/local_domain.cuh"
#include "stencil/rcstream.hpp"
#include "stencil/tx_common.hpp"
#include "stencil/tx_ipc.hpp"

class ColocatedDirectAccessSender {
private:
  int srcRank_, dstRank_;
  int srcDom_, dstDom_;

  LocalDomain *domain_;
  RcStream stream_;
  IpcSender ipcSender_;

  /* one memory handle per quantity
  */
  MPI_Request memReq_;
  std::vector<cudaIpcMemHandle_t> handles_;
  std::vector<void*> bufs_;

  std::vector<Message> outbox_;

public:
  ColocatedDirectAccessSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain);

  void start_prepare(const std::vector<Message> &outbox);

  void finish_prepare();

  void send();

  void wait();
};

class ColocatedDirectAccessRecver {
private:
  int srcRank_;
  int srcDom_;
  int dstDom_;

  LocalDomain *domain_;

  RcStream stream_;

  IpcRecver ipcRecver_;
  
  // send destination buffers to host
  MPI_Request memReq_;
  std::vector<cudaIpcMemHandle_t> handles_;

  enum class State {
    NONE,        // ready to recv
    WAIT_NOTIFY, // waiting on Irecv from ColocatedDirectAccessSender
    WAIT_KERNEL  // waiting on sender kernel to complete
  };
  State state_;

  char junk_; // to recv data into

public:
  ColocatedDirectAccessRecver(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain);

  void start_prepare(const std::vector<Message> &inbox);

  void finish_prepare();

  void recv();

  // once we are in the WAIT_KERNEL state, there's nothing else we need to do
  bool active() { return state_ == State::WAIT_NOTIFY; }

  bool next_ready();

  void next();

  void wait();
};