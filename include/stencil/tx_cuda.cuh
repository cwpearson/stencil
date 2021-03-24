#pragma once

#include <algorithm>
#include <functional>
#include <future>
#include <iomanip>
#include <map>
#include <sstream>

#include <mpi.h>

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h> // nvtxNameCudaStreamA

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/logging.hpp"
#include "stencil/packer.cuh"
#include "stencil/rcstream.hpp"
#include "stencil/timer.hpp"
#include "stencil/tx_common.hpp"
#include "stencil/tx_ipc.hpp"

#include "stencil/rt.hpp"

inline void print_bytes(const char *obj, size_t n) {
  std::cerr << std::hex << std::setfill('0'); // needs to be set only once
  auto *ptr = reinterpret_cast<const unsigned char *>(obj);
  for (size_t i = 0; i < n; i++, ptr++) {
    if (i % sizeof(uint64_t) == 0) {
      std::cerr << std::endl;
    }
    std::cerr << std::setw(2) << static_cast<unsigned>(*ptr) << " ";
  }
  std::cerr << std::endl;
}

/* Send messages to local domains with a kernel
 */
class PeerAccessSender {
private:
  std::vector<Message> outbox_;

  // one stream per source device
  std::map<int, RcStream> streams_;

  std::vector<const LocalDomain *> domains_;

public:
  PeerAccessSender() {}

  ~PeerAccessSender() {}

  void prepare(std::vector<Message> &outbox, const std::vector<LocalDomain> &domains) {
    outbox_ = outbox;
    std::sort(outbox_.begin(), outbox_.end());

    for (auto &e : domains) {
      domains_.push_back(&e);
    }

    // create a stream per device
    for (auto &msg : outbox_) {
      int srcDev = domains_[msg.srcGPU_]->gpu();
      streams_.emplace(srcDev, RcStream(srcDev, RcStream::Priority::HIGH));
    }

    for (size_t i = 0; i < streams_.size(); ++i) {
      nvtxNameCudaStreamA(streams_[i], ("PeerAccessSender" + std::to_string(i)).c_str());
    }
  }

  void send() {

    nvtxRangePush("PeerSender::send");

    // translate data with kernel
    for (auto &msg : outbox_) {
      const LocalDomain *srcDomain = domains_[msg.srcGPU_];
      const LocalDomain *dstDomain = domains_[msg.dstGPU_];
      const Dim3 srcPos = srcDomain->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 dstPos = dstDomain->halo_pos(msg.dir_ * -1, true /*exterior*/);
      const Dim3 extent = srcDomain->halo_extent(msg.dir_);
      LOG_SPEW("multi_translate dir=" << msg.dir_ << " src=" << srcPos << " dst=" << dstPos << " ext" << extent);
      RcStream &stream = streams_[srcDomain->gpu()];
      const dim3 dimBlock = Dim3::make_block_dim(extent, 512 /*threads per block*/);
      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      assert(stream.device() == srcDomain->gpu());
      assert(srcDomain->num_data() == dstDomain->num_data());
      // LOG_SPEW("multi_translate grid=" << dimGrid << " block=" << dimBlock);
      CUDA_RUNTIME(rt::time(cudaSetDevice, stream.device()));
#if 0
      multi_translate<<<dimGrid, dimBlock, 0, stream>>>(dstDomain->dev_curr_datas(), dstPos,
                                                        srcDomain->dev_curr_datas(), srcPos, extent,
                                                        srcDomain->dev_elem_sizes(), srcDomain->num_data());
#endif
      rt::launch(multi_translate, dimGrid, dimBlock, 0, stream, dstDomain->dev_curr_datas(), dstPos,
                 srcDomain->dev_curr_datas(), srcPos, extent, srcDomain->dev_elem_sizes(), srcDomain->num_data());
      CUDA_RUNTIME(cudaGetLastError());
    }

    nvtxRangePop(); // PeerSender::send
  }

  void wait() {

    for (auto &kv : streams_) {
      CUDA_RUNTIME(cudaSetDevice(kv.second.device()));
      CUDA_RUNTIME(cudaStreamSynchronize(kv.second));
    }
  }
};

/* Send messages between local domains by pack, cudaMemcpyPeerAsync, unpack
 */
class PeerCopySender {
private:
  size_t srcGPU_;
  size_t dstGPU_;
  LocalDomain *srcDomain_;
  LocalDomain *dstDomain_;

  // one stream per source and destination operations
  RcStream srcStream_;
  RcStream dstStream_;

  // event to sync src and dst streams
  cudaEvent_t event_;

  // packed buffers
  DevicePacker packer_;
  DeviceUnpacker unpacker_;

public:
  PeerCopySender(size_t srcGPU, size_t dstGPU, LocalDomain &srcDomain, LocalDomain &dstDomain)
      : srcGPU_(srcGPU), dstGPU_(dstGPU), srcDomain_(&srcDomain), dstDomain_(&dstDomain),
        srcStream_(srcDomain.gpu(), RcStream::Priority::HIGH), dstStream_(dstDomain.gpu(), RcStream::Priority::HIGH),
        packer_(srcStream_), unpacker_(dstStream_) {}

  void prepare(std::vector<Message> &outbox) {
    packer_.prepare(srcDomain_, outbox);
    unpacker_.prepare(dstDomain_, outbox);

    // create event
    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    CUDA_RUNTIME(cudaEventCreate(&event_));
  }

  void send() {
    nvtxRangePush("PeerCopySender::send");
    assert(packer_.data());
    assert(unpacker_.data());
    assert(packer_.size() == unpacker_.size());

    // pack data in source stream
    packer_.pack();

    // copy from src device to dst device
    const int dstDev = dstDomain_->gpu();
    const int srcDev = srcDomain_->gpu();
    CUDA_RUNTIME(
        rt::time(cudaMemcpyPeerAsync, unpacker_.data(), dstDev, packer_.data(), srcDev, packer_.size(), srcStream_));

    // sync src and dst streams
    CUDA_RUNTIME(rt::time(cudaEventRecord, event_, srcStream_));
    CUDA_RUNTIME(rt::time(cudaSetDevice, dstDomain_->gpu()));
    CUDA_RUNTIME(rt::time(cudaStreamWaitEvent, dstStream_, event_, 0 /*flags*/));

    unpacker_.unpack();
    nvtxRangePop(); // PeerCopySender::send
  }

  void wait() {
    CUDA_RUNTIME(cudaSetDevice(srcStream_.device()));
    CUDA_RUNTIME(cudaStreamSynchronize(srcStream_));
    CUDA_RUNTIME(cudaSetDevice(dstStream_.device()));
    CUDA_RUNTIME(cudaStreamSynchronize(dstStream_));
  }
};

/*! Send data between CUDA devices in colocated ranks
    Issue copy, then record event, then send a message notifying recver that event was recorded
 */
class ColocatedDeviceSender {
private:
  int srcRank_;
  int srcGPU_; // domain ID
  int srcDev_; // cuda ID
  int dstRank_;
  int dstGPU_; // domain ID

  void *dstBuf_; // buffer on dst device in another process
  size_t bufSize_;

  IpcSender ipcSender_;

  cudaIpcMemHandle_t memHandle_;

  MPI_Request memReq_;

  int tagUb_; // largest tag value, if positive

public:
  ColocatedDeviceSender() : dstBuf_(nullptr) {}
  ColocatedDeviceSender(int srcRank, int srcGPU, // domain ID
                        int dstRank, int dstGPU, // domain ID
                        int srcDev)              // cuda ID
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU), srcDev_(srcDev), dstBuf_(nullptr),
        bufSize_(0), ipcSender_(srcRank, srcGPU, dstRank, dstGPU, srcDev) {

    tagUb_ = mpi::tag_ub(MPI_COMM_WORLD);
  }

  ~ColocatedDeviceSender() {
    if (dstBuf_) {
      CUDA_RUNTIME(rt::time(cudaSetDevice, srcDev_));
      CUDA_RUNTIME(cudaIpcCloseMemHandle(dstBuf_));
      dstBuf_ = nullptr;
    }
  }

  int payload() const noexcept { return ((srcGPU_ & 0xFF) << 8) | (dstGPU_ & 0xFF); }

  void start_prepare(size_t numBytes) {
    ipcSender_.async_prepare();

    // Recieve the IPC mem handle for the buffer
    const int memHandleTag = make_tag<MsgKind::ColocatedBuf>(payload());
    assert(memHandleTag < tagUb_);
    MPI_Irecv(&memHandle_, sizeof(memHandle_), MPI_BYTE, dstRank_, memHandleTag, MPI_COMM_WORLD, &memReq_);

    bufSize_ = numBytes;
  }

  void finish_prepare() {
    // wait for recv mem handle and convert to pointer
    MPI_Wait(&memReq_, MPI_STATUS_IGNORE);
    CUDA_RUNTIME(rt::time(cudaSetDevice, srcDev_));
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&dstBuf_, memHandle_, cudaIpcMemLazyEnablePeerAccess));

    ipcSender_.wait_prepare();
  }

  void send(const void *devPtr, RcStream &stream) {
    assert(srcDev_ == stream.device());
    assert(dstBuf_);
    assert(devPtr);
    assert(srcDev_ >= 0);
    assert(bufSize_ > 0);
    CUDA_RUNTIME(rt::time(cudaSetDevice, srcDev_));
    CUDA_RUNTIME(rt::time(cudaMemcpyPeerAsync, dstBuf_, ipcSender_.dst_dev(), devPtr, srcDev_, bufSize_, stream));
    // record the event
    CUDA_RUNTIME(rt::time(cudaEventRecord, ipcSender_.event(), stream));
    ipcSender_.async_notify();
  }

  void wait() {
    ipcSender_.wait_notify();
    CUDA_RUNTIME(cudaSetDevice(srcDev_));
    CUDA_RUNTIME(cudaEventSynchronize(ipcSender_.event()));
  }
};

class ColocatedDeviceRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_; // domain ID
  int dstDev_; // cuda ID

  cudaIpcMemHandle_t memHandle_;
  MPI_Request memReq_;

  IpcRecver ipcRecver_;

public:
  ColocatedDeviceRecver() {}
  ColocatedDeviceRecver(int srcRank, int srcGPU, int dstRank,
                        int dstGPU, // domain ID
                        int dstDev  // cuda ID
                        )
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU), dstDev_(dstDev),
        ipcRecver_(srcRank, srcGPU, dstRank, dstGPU, dstDev) {}
  ~ColocatedDeviceRecver() = default;

  /*! prepare to recieve devPtr
   */
  void start_prepare(void *devPtr) {
    LOG_SPEW("ColocatedDeviceRecver::start_prepare(): entry");
    ipcRecver_.async_prepare();

    // fprintf(stderr, "ColoDevRecv::start_prepare: send mem on %d(%d) to
    // r%dg%d\n", dstDev_, dstGPU_, srcRank_, srcGPU_);
    assert(devPtr);

    int payload = ((srcGPU_ & 0xFF) << 8) | (dstGPU_ & 0xFF);

    // get an a memory handle
    CUDA_RUNTIME(rt::time(cudaSetDevice, dstDev_));
    CUDA_RUNTIME(cudaIpcGetMemHandle(&memHandle_, devPtr));

    // Send the mem handle to the ColocatedSender
    const int memTag = make_tag<MsgKind::ColocatedBuf>(payload);
    MPI_Isend(&memHandle_, sizeof(memHandle_), MPI_BYTE, srcRank_, memTag, MPI_COMM_WORLD, &memReq_);
    LOG_SPEW("ColocatedDeviceRecver::start_prepare(): exit");
  }

  void finish_prepare() {
    ipcRecver_.wait_prepare();

    // wait to send the mem handle and the CUDA device ID
    MPI_Wait(&memReq_, MPI_STATUS_IGNORE);
  }

  // start listening for notify
  void async_listen() { ipcRecver_.async_listen(); }

  // true if we have been notified
  bool test_listen() { return ipcRecver_.test_listen(); }

  /*! have stream wait for data to arrive

     If this gets called before the associated ColocatedDeviceSender starts the copy, then it will
     not work, of course.
     Rely on whoever owns us to handle that case.
   */
  void wait(RcStream &stream) {
    assert(ipcRecver_.event());
    assert(stream.device() == dstDev_);

    // wait for ColocatedDeviceSender cudaMemcpyPeerAsync to be done
    CUDA_RUNTIME(cudaStreamWaitEvent(stream, ipcRecver_.event(), 0 /*flags*/));
  }
};

/* For colocated, either the sender or reciever has to be stateful.
   This sender implementation is async, with the sender packing the data in the stream,
   sending the data in the stream, recording an event in the stream, and then
   MPI_Isending a message to the ColocatedHaloRecver, letting it know that it can
   wait on the event.
   We defer most of the implementation to the ColocatedDeviceSender

   The other alternative would be for the sender to send a message once the copy was done.
   In that case, the recver becomes async and just blocks until it gets that message.
   Pros: the recver does not need an event to wait on.
   Cons: this send has to complete before we can start issuing recvs
*/
class ColocatedHaloSender : public StatefulSender {
private:
  LocalDomain *domain_;
  int srcRank_;

  RcStream stream_;
  DevicePacker packer_;
  ColocatedDeviceSender sender_;

public:
  ColocatedHaloSender(int srcRank, int srcGPU, int dstRank, int dstGPU, LocalDomain &domain)
      : domain_(&domain), stream_(domain.gpu(), RcStream::Priority::HIGH), packer_(stream_),
        sender_(srcRank, srcGPU, dstRank, dstGPU, domain.gpu()) {
    std::string streamName("ColocatedHaloSender_");
    streamName += "r" + std::to_string(srcRank);
    streamName += "g" + std::to_string(srcGPU);
    streamName += "->r" + std::to_string(dstRank);
    streamName += "g" + std::to_string(dstGPU);
    nvtxNameCudaStreamA(stream_, streamName.c_str());
  }

  void start_prepare(const std::vector<Message> &outbox) override {
    packer_.prepare(domain_, outbox);
    if (0 == packer_.size()) {
      std::cerr << "WARN: 0-size ColocatedHaloSender was created\n";
    }
    sender_.start_prepare(packer_.size());
  }

  void finish_prepare() override { sender_.finish_prepare(); }

  void send() noexcept override {
    LOG_SPEW("ColoHaloSender::send()");
    nvtxRangePush("ColoHaloSender: init pack");
    packer_.pack();
    nvtxRangePop();
    nvtxRangePush("ColoHaloSender: init send");
    sender_.send(packer_.data(), stream_);
    nvtxRangePop();
  }

  void wait() noexcept override { sender_.wait(); }

  // unused, but filling StatefulSender interface
  bool active() override { return false; }
  bool next_ready() override { return false; }
  void next() override{};
};

/* The receiver is stateful because it can't start to wait on the
   event until it knows the sender has recorded the event.
   So, we wait on a message saying the event has been recorded,
   then we can transition to waiting on the event itself
*/
class ColocatedHaloRecver : public StatefulRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstGPU_;

  LocalDomain *domain_;

  RcStream stream_;

  MPI_Request notifyReq_;

  ColocatedDeviceRecver recver_;
  DeviceUnpacker unpacker_;

  /* NONE: ready to recv
     WAIT_NOTIFY: waiting on Irecv from ColocatedHaloSender
     WAIT_COPY: waiting on copy
  */
  enum class State { NONE, WAIT_NOTIFY, WAIT_COPY };
  State state_;

  char junk_; // to recv data into

public:
  ColocatedHaloRecver(int srcRank, int srcGPU, int dstRank, int dstGPU, LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstGPU_(dstGPU), domain_(&domain),
        stream_(domain.gpu(), RcStream::Priority::HIGH), recver_(srcRank, srcGPU, dstRank, dstGPU, domain.gpu()),
        unpacker_(stream_), state_(State::NONE) {
    std::string streamName("ColocatedHaloRecver_");
    streamName += "r" + std::to_string(srcRank);
    streamName += "g" + std::to_string(srcGPU);
    streamName += "->r" + std::to_string(dstRank);
    streamName += "g" + std::to_string(dstGPU);
    nvtxNameCudaStreamA(stream_, streamName.c_str());
  }

  void start_prepare(const std::vector<Message> &inbox) override {
    unpacker_.prepare(domain_, inbox);
    if (0 == unpacker_.size()) {
      std::cerr << "WARN: a 0-size ColocatedHaloRecver was created\n";
    }
    recver_.start_prepare(unpacker_.data());
  }

  void finish_prepare() override { recver_.finish_prepare(); }

  void recv() override {
    assert(State::NONE == state_);
    state_ = State::WAIT_NOTIFY;

    nvtxRangePush("ColoHaloRecver: init listen");
    recver_.async_listen();
    nvtxRangePop();

    assert(stream_.device() == domain_->gpu());
  }

  // once we are in the wait_copy state, there's nothing else we need to do
  bool active() override { return state_ == State::WAIT_NOTIFY; }

  bool next_ready() override {
    if (State::WAIT_NOTIFY == state_) {
      return recver_.test_listen();
    } else { // should only be asked this in active() states
      LOG_FATAL("unexpected state");
    }
  }

  void next() override {
    if (State::WAIT_NOTIFY == state_) {
      // have device recver wait on its stream, and then unpack the data.
      // The device recver knows how to exchange with the
      state_ = State::WAIT_COPY;
      recver_.wait(stream_);
      nvtxRangePush("ColoHaloRecver: init unpack");
      unpacker_.unpack();
      nvtxRangePop();
    }
  }

  void wait() noexcept override {
    // wait on unpacker
    assert(stream_.device() == domain_->gpu());
    CUDA_RUNTIME(cudaSetDevice(stream_.device()));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    state_ = State::NONE;
  }
};

/*! Send from one domain to a remote domain
 */
class RemoteSender : public StatefulSender {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  LocalDomain *domain_;

  char *hostBuf_;

  RcStream stream_;
  MPI_Request req_;

  enum class State { Idle, D2H, Wait };
  State state_;

  DevicePacker packer_;

public:
  // RemoteSender() : hostBuf_(nullptr) {}
  RemoteSender(int srcRank, int srcGPU, int dstRank, int dstGPU, LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU), domain_(&domain), hostBuf_(nullptr),
        stream_(domain.gpu(), RcStream::Priority::HIGH), state_(State::Idle), packer_(stream_) {}

  ~RemoteSender() { CUDA_RUNTIME(cudaFreeHost(hostBuf_)); }

  /*! Prepare to send a set of messages whose direction vectors are store in
   outbox.

   If the outbox is empty, the packer may be size 0
   */
  void start_prepare(const std::vector<Message> &outbox) override {

    LOG_INFO(outbox.size() << "-size outbox provided to RemoteSender "
                           << "r" << srcRank_ << "d" << srcGPU_ << "->"
                           << "r" << dstRank_ << "d" << dstGPU_);

    packer_.prepare(domain_, outbox);

    LOG_INFO(packer_.size() << "B RemoteSender was prepared: "
                            << "r" << srcRank_ << "d" << srcGPU_ << "->"
                            << "r" << dstRank_ << "d" << dstGPU_);

    if (0 != packer_.size()) {
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

      // allocate device & host buffers
      CUDA_RUNTIME(cudaHostAlloc(&hostBuf_, packer_.size(), cudaHostAllocDefault));
      assert(hostBuf_);
    }
  }

  void finish_prepare() override {
    // no-op
  }

  virtual void send() override {
    state_ = State::D2H;
    send_d2h();
  }

  virtual bool active() override { return State::Wait != state_; }

  virtual bool next_ready() override {
    if (state_ == State::D2H) {
      return d2h_done();
    } else {
      return false;
    }
  }

  virtual void next() override {
    if (State::D2H == state_) {
      state_ = State::Wait;
      send_h2h();
    }
  }

  virtual void wait() override {
    assert(State::Wait == state_);
    if (packer_.size()) {
      MPI_Wait(&req_, MPI_STATUS_IGNORE);
    }
    state_ = State::Idle;
  }

  void send_d2h() {
    if (packer_.size()) {
      nvtxRangePush("RemoteSender::send_d2h");
      // pack data into device buffer
      assert(stream_.device() == domain_->gpu());
      packer_.pack();

      // copy to host buffer
      assert(hostBuf_);
      CUDA_RUNTIME(rt::time(cudaMemcpyAsync, hostBuf_, packer_.data(), packer_.size(), cudaMemcpyDefault, stream_));

      nvtxRangePop(); // RemoteSender::send_d2h
    }
  }

  bool is_d2h() const noexcept { return State::D2H == state_; }

  bool d2h_done() {
    assert(State::D2H == state_);
    if (packer_.size()) {
      cudaError_t err = rt::time(cudaStreamQuery, stream_);
      if (cudaSuccess == err) {
        return true;
      } else if (cudaErrorNotReady == err) {
        return false;
      } else {
        CUDA_RUNTIME(err);
        __builtin_unreachable();
      }
    } else {
      return true;
    }
  }

  void send_h2h() {
    if (packer_.size()) {
      nvtxRangePush("RemoteSender::send_h2h");
      assert(hostBuf_);
      assert(packer_.size());
      assert(srcGPU_ < 8);
      assert(dstGPU_ < 8);
      const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
      mpirt::time(MPI_Isend, hostBuf_, packer_.size(), MPI_BYTE, dstRank_, tag, MPI_COMM_WORLD, &req_);
      nvtxRangePop(); // RemoteSender::send_h2h
    }
  }
};

/*! Recv from a remote domain into a domain
 */
class RemoteRecver : public StatefulRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  LocalDomain *domain_;

  char *hostBuf_;

  RcStream stream_;

  MPI_Request req_;

  enum class State { None, H2H, H2D };
  State state_;

  DeviceUnpacker unpacker_;

public:
  RemoteRecver() = delete;
  RemoteRecver(int srcRank, int srcGPU, int dstRank, int dstGPU, LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU), domain_(&domain), hostBuf_(nullptr),
        stream_(domain.gpu(), RcStream::Priority::HIGH), state_(State::None), unpacker_(stream_) {
    CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));
  }

  ~RemoteRecver() { CUDA_RUNTIME(cudaFreeHost(hostBuf_)); }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  virtual void start_prepare(const std::vector<Message> &inbox) override {
    unpacker_.prepare(domain_, inbox);
    if (0 == unpacker_.size()) {
      LOG_INFO("0-size RemoteRecver was prepared");
    } else {
      CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));

      // allocate device & host buffers
      CUDA_RUNTIME(cudaHostAlloc(&hostBuf_, unpacker_.size(), cudaHostAllocDefault));
      assert(hostBuf_);
    }
  }

  virtual void finish_prepare() override {
    // no-op
  }

  virtual void recv() override {
    state_ = State::H2H;
    recv_h2h();
  }

  virtual bool active() override {
    assert(State::None != state_);
    return State::H2D != state_;
  }

  virtual bool next_ready() override {
    assert(State::H2H == state_);
    return h2h_done();
  }

  virtual void next() override {
    if (State::H2H == state_) {
      state_ = State::H2D;
      recv_h2d();
    } else {
      assert(0);
      __builtin_unreachable();
    }
  }

  virtual void wait() override {
    assert(State::H2D == state_);
    if (unpacker_.size()) {
      CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    }
  }

  void recv_h2d() {
    if (unpacker_.size()) {
      nvtxRangePush("RemoteRecver::recv_h2d");
      // copy to device buffer
      CUDA_RUNTIME(rt::time(cudaMemcpyAsync, unpacker_.data(), hostBuf_, unpacker_.size(), cudaMemcpyDefault, stream_));
      unpacker_.unpack();
      nvtxRangePop(); // RemoteRecver::recv_h2d
    }
  }

  bool is_h2h() const { return State::H2H == state_; }

  bool h2h_done() {
    assert(State::H2H == state_);
    if (unpacker_.size()) {
      int flag;
      mpirt::time(MPI_Test, &req_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  void recv_h2h() {
    if (unpacker_.size()) {
      nvtxRangePush("RemoteRecver::recv_h2h");
      assert(hostBuf_);
      assert(srcGPU_ < 8);
      assert(dstGPU_ < 8);
      const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
      int numBytes = unpacker_.size();
      assert(numBytes <= std::numeric_limits<int>::max());
      mpirt::time(MPI_Irecv, hostBuf_, int(numBytes), MPI_BYTE, srcRank_, tag, MPI_COMM_WORLD, &req_);
      nvtxRangePop(); // RemoteRecver::recv_h2h
    }
  }
};

/*! Send from one domain to a remote domain


    Has 3 states: None, Pack, and Send

    None->Pack, launches packer
    Pack: active
    Pack->Send, launches MPI_Isend
    Send: not active
    On wait, transition pack to None

 */
class CudaAwareMpiSender : public StatefulSender {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  LocalDomain *domain_;

  RcStream stream_;
  MPI_Request req_;

  enum class State {
    None,
    Pack,
    Send,
  };
  State state_;

  std::vector<Message> outbox_;

  DevicePacker packer_;

public:
  CudaAwareMpiSender() = delete;
  CudaAwareMpiSender(int srcRank, int srcGPU, int dstRank, int dstGPU, LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU), domain_(&domain),
        stream_(domain.gpu(), RcStream::Priority::HIGH), req_({}), state_(State::None), packer_(stream_) {}

  virtual void start_prepare(const std::vector<Message> &outbox) override {
    packer_.prepare(domain_, outbox);
    if (0 == packer_.size()) {
      LOG_FATAL("a 0-size CudaAwareMpiSender was prepared");
    }
  }

  virtual void finish_prepare() override {
    // no-op
  }

  virtual void send() override {
    assert(State::None == state_);
    state_ = State::Pack;
    send_pack();
  }

  virtual bool active() override { return State::Pack == state_; }

  virtual bool next_ready() override {
    if (State::Pack == state_) {
      return pack_done();
    } else {
      LOG_FATAL("unexpected state");
    }
  }

  virtual void next() override {
    if (State::Pack == state_) {
      state_ = State::Send;
      send_d2d();
    } else {
      LOG_FATAL("unexpected state");
    }
  }

  virtual void wait() override;

private:
  // launch pack kernel
  void send_pack();

  // true if pack kernel finished
  bool pack_done();

  // post MPI_Isend
  void send_d2d();
};

class CudaAwareMpiRecver : public StatefulRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  LocalDomain *domain_;

  RcStream stream_;
  MPI_Request req_;

  enum class State {
    None,
    Recv,
    Unpack,
  };
  State state_;

  DeviceUnpacker unpacker_;

public:
  CudaAwareMpiRecver() = delete;
  CudaAwareMpiRecver(int srcRank, int srcGPU, int dstRank, int dstGPU, LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU), domain_(&domain),
        stream_(domain.gpu(), RcStream::Priority::HIGH), state_(State::None), unpacker_(stream_) {}

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  void start_prepare(const std::vector<Message> &inbox) override {
    unpacker_.prepare(domain_, inbox);
    if (0 == unpacker_.size()) {
      LOG_FATAL("a 0-size CudaAwareMpiRecver was created");
    }
  }

  void finish_prepare() override {
    // no-op
  }

  virtual void recv() override {
    state_ = State::Recv;
    recv_d2d();
  }

  virtual bool active() override {
    assert(State::None != state_);
    return State::Unpack != state_;
  }

  virtual bool next_ready() override {
    if (State::Recv == state_) {
      return d2d_done();
    } else {
      LOG_FATAL("unexpected state");
    }
  }

  virtual void next() override {
    if (State::Recv == state_) {
      state_ = State::Unpack;
      recv_unpack();
    } else {
      LOG_FATAL("unreachable");
    }
  }

  virtual void wait() override {
    assert(unpacker_.size());
    assert(State::Unpack == state_);
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

private:
  // launch unpack kernel
  void recv_unpack();

  // true if MPI_Irecv is done
  bool d2d_done();

  // post MPI_Irecv
  void recv_d2d();
};
