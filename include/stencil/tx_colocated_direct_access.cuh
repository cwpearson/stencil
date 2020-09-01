#pragma once

#include "stencil/local_domain.cuh"
#include "stencil/partition.hpp"
#include "stencil/rcstream.hpp"
#include "stencil/tx_common.hpp"
#include "stencil/tx_ipc.hpp"

class ColocatedDirectAccessSender {
private:
  int srcRank_, dstRank_;
  int srcDom_, dstDom_;

  LocalDomain *domain_;
  Placement *placement_;
  RcStream stream_;
  IpcSender ipcSender_;

  /* one memory handle per quantity
   */
  MPI_Request memReq_;
  std::vector<cudaIpcMemHandle_t> handles_;

  // pointers to the destination domain buffers
  std::vector<void *> dstDomCurrDatas_;
  void **dstDomCurrDatasDev_;

public:
  ColocatedDirectAccessSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                              Placement *placement);
  ~ColocatedDirectAccessSender();
  void start_prepare();
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

  ~ColocatedDirectAccessRecver();
  
  void start_prepare();

  void finish_prepare();

  void recv();

  bool active();

  bool next_ready();

  void next();

  void wait();
};