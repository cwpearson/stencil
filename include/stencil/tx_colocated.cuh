#pragma once

/*! \file tx_colocated.cuh
    \brief Tx/Rx for colocated devices where data is sent directly into destination memory

    Defines ColoHaloSender, which does of the hard work but do not specify a particular Translator.
    Use one of the derived Sender classes, which uses a specific Translator.

    The ColoHaloRecver may be used with any derived Sender.

    Since these senders/recvers write directly into the destination memory, they must track both the `current` and
   `next` data at the destination, and allow themselves to be swapped


    FIXME: Not to be confused with `ColocatedHaloSender` and `ColocatedHaloRecver`, which must be used together
    This is the pack/memcpy/unpack sender
*/

#include "stencil/local_domain.cuh"
#include "stencil/partition.hpp"
#include "stencil/rcstream.hpp"
#include "stencil/translator.cuh"
#include "stencil/tx_common.hpp"
#include "stencil/tx_ipc.hpp"

/* Interface and almost full implementation for a Halo sender between two co-located domains

   DO NOT USE DIRECTLY. Use one of the derived classes
*/
class ColoHaloSender : public StatefulSender {
protected:
  // to be set by the derived class. This class sets to nullptr
  Translator *currTranslator_; // translator for `current` data
  Translator *nextTranslator_; // translator for `next` data

private:
  int srcRank_, dstRank_;
  int srcDom_, dstDom_;

  const LocalDomain *domain_;
  Placement *placement_;
  RcStream stream_;
  IpcSender ipcSender_;

  /* one memory handle per quantity
   */
  MPI_Request currMemReq_, nextMemReq_;
  std::vector<cudaIpcMemHandle_t> currMemHandles_, nextMemHandles_;

  /* pitch information about quantities
   */
  MPI_Request ptrReq_;

  std::vector<Message> outbox_;

  // pointers to the destination domain buffers
  std::vector<cudaPitchedPtr> dstDomCurrDatas_;
  std::vector<cudaPitchedPtr> dstDomNextDatas_;

public:
  ColoHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain, Placement *placement);
  virtual ~ColoHaloSender();

  void start_prepare(const std::vector<Message> &outbox) override;
  void finish_prepare() override;
  void send() override;
  void wait() override;

  // unused, but filling StatefulSender interface
  bool active() override { return false; }
  bool next_ready() override { return false; }
  void next() override{};

  // swap the current next destination pointers
  void swap();
};

/* Do a colocated halo send using cudaMemcpy3d
 */
class ColoMemcpy3dHaloSender : public ColoHaloSender {
public:
  ColoMemcpy3dHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain, Placement *placement);
};

/* Do a colocated halo send using direct access
 */
class ColoQuantityKernelSender : public ColoHaloSender {
public:
  ColoQuantityKernelSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain, Placement *placement);
};

/* Do a colocated halo send using direct access
 */
class ColoRegionKernelSender : public ColoHaloSender {
public:
  ColoRegionKernelSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain, Placement *placement);
};

/* to be paired on the recieving end of any ColoHaloSender
 */
class ColoHaloRecver : public StatefulRecver {
private:
  int srcRank_;
  int srcDom_;
  int dstDom_;

  LocalDomain *domain_;

  // wait for a signal on both of these to know the sender is done
  RcStream stream_;
  IpcRecver ipcRecver_;

  // send destination buffers to host
  MPI_Request currMemReq_, nextMemReq_;
  std::vector<cudaIpcMemHandle_t> currHandles_, nextHandles_;

  // pitch information about quantities
  MPI_Request ptrReq_;

  enum class State {
    NONE,        // ready to recv
    WAIT_NOTIFY, // waiting on Irecv from ColoHaloSender
    WAIT_KERNEL  // waiting on sender kernel to complete
  };
  State state_;

public:
  ColoHaloRecver(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain);

  ~ColoHaloRecver();

  void start_prepare(const std::vector<Message> &inbox) override;
  void finish_prepare() override;
  void recv();
  bool active();
  bool next_ready();
  void next();
  void wait();
};