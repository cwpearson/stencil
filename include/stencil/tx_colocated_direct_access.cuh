#pragma once

#include "stencil/local_domain.cuh"
#include "stencil/partition.hpp"
#include "stencil/rcstream.hpp"
#include "stencil/translate.cuh"
#include "stencil/tx_common.hpp"
#include "stencil/tx_ipc.hpp"

class ColoDirectAccessHaloSender : public StatefulSender {
private:
  int srcRank_, dstRank_;
  int srcDom_, dstDom_;

  const LocalDomain *domain_;
  Placement *placement_;
  RcStream stream_;
  IpcSender ipcSender_;
  Translate translate_;

  /* one memory handle per quantity
   */
  MPI_Request memReq_;
  std::vector<cudaIpcMemHandle_t> memHandles_;

  /* pitch information about quantities
  */
  MPI_Request ptrReq_;

  std::vector<Message> outbox_;

  // pointers to the destination domain buffers
  std::vector<cudaPitchedPtr> dstDomCurrDatas_;
  cudaPitchedPtr *dstDomCurrDatasDev_;

public:
  ColoDirectAccessHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                             Placement *placement);
  ~ColoDirectAccessHaloSender();

  void start_prepare(const std::vector<Message> &outbox) override;
  void finish_prepare() override;
  void send() override;
  void wait() override;

  // unused, but filling StatefulSender interface
  bool active() override { return false; }
  bool next_ready() override { return false; }
  void next() override{};
};

class ColoDirectAccessHaloRecver : public StatefulRecver {
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

  // pitch information about quantities
  MPI_Request ptrReq_;

  enum class State {
    NONE,        // ready to recv
    WAIT_NOTIFY, // waiting on Irecv from ColoDirectAccessHaloSender
    WAIT_KERNEL  // waiting on sender kernel to complete
  };
  State state_;

  char junk_; // to recv data into

public:
  ColoDirectAccessHaloRecver(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain);

  ~ColoDirectAccessHaloRecver();

  void start_prepare(const std::vector<Message> &inbox) override;
  void finish_prepare() override;
  void recv();
  bool active();
  bool next_ready();
  void next();
  void wait();
};