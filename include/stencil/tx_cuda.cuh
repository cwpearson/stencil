#pragma once

#include <functional>
#include <future>
#include <iomanip>
#include <sstream>

#include <mpi.h>

#include <nvToolsExt.h>

// getpid()
#include <sys/types.h>
#include <unistd.h>

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/packer.cuh"
#include "stencil/rcstream.hpp"

#include "tx_common.hpp"

// #define ANY_LOUD
// #define REGION_LOUD
// #define REMOTE_LOUD

inline void print_bytes(const char *obj, size_t n) {
  std::cerr << std::hex << std::setfill('0'); // needs to be set only once
  auto *ptr = reinterpret_cast<const unsigned char *>(obj);
  for (int i = 0; i < n; i++, ptr++) {
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

  void prepare(std::vector<Message> &outbox,
               const std::vector<LocalDomain> &domains) {
    outbox_ = outbox;

    for (auto &e : domains) {
      domains_.push_back(&e);
    }

    // create a stream per device
    for (auto &msg : outbox_) {
      int srcDev = domains_[msg.srcGPU_]->gpu();
      streams_.emplace(srcDev, RcStream(srcDev));
    }
  }

  void send() {

    nvtxRangePush("PeerSender::send");

    // translate data with kernel
    for (auto &msg : outbox_) {
      const LocalDomain *srcDomain = domains_[msg.srcGPU_];
      const LocalDomain *dstDomain = domains_[msg.dstGPU_];
      const Dim3 dstSz = dstDomain->raw_size();
      const Dim3 srcSz = srcDomain->raw_size();
      const Dim3 srcPos = srcDomain->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 dstPos = dstDomain->halo_pos(msg.dir_, true /*exterior*/);
      const Dim3 extent = srcDomain->halo_extent(msg.dir_);
      RcStream &stream = streams_[srcDomain->gpu()];
      const dim3 dimBlock = make_block_dim(extent, 512 /*threads per block*/);
      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      assert(stream.device() == srcDomain->gpu());
      CUDA_RUNTIME(cudaSetDevice(stream.device()));
      assert(srcDomain->num_data() == dstDomain->num_data());

      multi_translate<<<dimGrid, dimBlock, 0, stream>>>(
          dstDomain->dev_curr_datas(), dstPos, dstSz,
          srcDomain->dev_curr_datas(), srcPos, srcSz, extent,
          srcDomain->dev_elem_sizes(), srcDomain->num_data());
      CUDA_RUNTIME(cudaGetLastError());
    }

    nvtxRangePop(); // PeerSender::send
  }

  void wait() {
    for (auto &kv : streams_) {
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
  PeerCopySender(size_t srcGPU, size_t dstGPU, LocalDomain &srcDomain,
                 LocalDomain &dstDomain)
      : srcGPU_(srcGPU), dstGPU_(dstGPU), srcDomain_(&srcDomain),
        dstDomain_(&dstDomain), srcStream_(srcDomain.gpu()),
        dstStream_(dstDomain.gpu()) {}

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
    packer_.pack(srcStream_);

    // copy from src device to dst device
    const int dstDev = dstDomain_->gpu();
    const int srcDev = srcDomain_->gpu();
    CUDA_RUNTIME(cudaMemcpyPeerAsync(unpacker_.data(), dstDev, packer_.data(),
                                     srcDev, packer_.size(), srcStream_));

    // sync src and dst streams
    CUDA_RUNTIME(cudaEventRecord(event_, srcStream_));
    CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
    CUDA_RUNTIME(cudaStreamWaitEvent(dstStream_, event_, 0 /*flags*/));

    unpacker_.unpack(dstStream_);
    nvtxRangePop(); // PeerCopySender::send
  }

  void wait() {
    cudaStreamSynchronize(srcStream_);
    cudaStreamSynchronize(dstStream_);
  }
};

/*! Send data between CUDA devices in colocated ranks
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

  int dstDev_; // cuda ID

  cudaEvent_t event_;
  cudaIpcMemHandle_t memHandle_;
  cudaIpcEventHandle_t evtHandle_;

  MPI_Request evtReq_;
  MPI_Request memReq_;
  MPI_Request idReq_;

  int tagUb_; // largest tag value, if positive

public:
  ColocatedDeviceSender() : dstBuf_(nullptr), event_(0) {}
  ColocatedDeviceSender(int srcRank, int srcGPU, // domain ID
                        int dstRank, int dstGPU, // domain ID
                        int srcDev)              // cuda ID
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        srcDev_(srcDev), dstBuf_(nullptr), bufSize_(0), event_(0) {

    int tag_ub = -1;
    int flag;
    int *tag_ub_ptr;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);
    assert(flag);
    if (flag)
      tag_ub = *tag_ub_ptr;
    tagUb_ = tag_ub;
  }

  ~ColocatedDeviceSender() {
    if (dstBuf_) {
      CUDA_RUNTIME(cudaSetDevice(srcDev_));
      CUDA_RUNTIME(cudaIpcCloseMemHandle(dstBuf_));
    }
    dstBuf_ = nullptr;
    if (event_) {
      CUDA_RUNTIME(cudaEventDestroy(event_));
    }
  }

  void start_prepare(size_t numBytes) {
    const int payload = ((srcGPU_ & 0xFF) << 8) | (dstGPU_ & 0xFF);

    // fprintf(stderr, "ColoDevSend::start_prepare: srcDev=%d (r%dg%d to
    // r%dg%d)\n", srcDev_, srcRank_, srcGPU_, dstRank_, dstGPU_);

    // create an event and associated handle
    CUDA_RUNTIME(cudaSetDevice(srcDev_));
    CUDA_RUNTIME(cudaEventCreate(&event_, cudaEventInterprocess |
                                              cudaEventDisableTiming));
    CUDA_RUNTIME(cudaIpcGetEventHandle(&evtHandle_, event_));

    MPI_Isend(&evtHandle_, sizeof(evtHandle_), MPI_BYTE, dstRank_,
              make_tag<MsgKind::ColocatedEvt>(payload), MPI_COMM_WORLD,
              &evtReq_);

    // Recieve the IPC mem handle
    const int memHandleTag = make_tag<MsgKind::ColocatedMem>(payload);
    // fprintf(stderr, "ColoDevSend::start_prepare: mem handle tag=%d\n",
    // memHandleTag);
    assert(memHandleTag < tagUb_);
    MPI_Irecv(&memHandle_, sizeof(memHandle_), MPI_BYTE, dstRank_, memHandleTag,
              MPI_COMM_WORLD, &memReq_);
    // Retrieve the destination device id
    const int devTag = make_tag<MsgKind::ColocatedDev>(payload);
    // fprintf(stderr, "ColoDevSend::start_prepare: dev tag=%d\n", devTag);
    assert(devTag < tagUb_);
    MPI_Irecv(&dstDev_, 1, MPI_INT, dstRank_, devTag, MPI_COMM_WORLD, &idReq_);

    // compute the required buffer size
    bufSize_ = numBytes;
  }

  void finish_prepare() {
    // block until we have recved the device ID
    MPI_Wait(&idReq_, MPI_STATUS_IGNORE);
    // fprintf(stderr, "ColoDevSend::finish_prepare: srcDev=%d dstDev=%d (r%dg%d
    // to r%dg%d)\n", srcDev_, dstDev_, srcRank_, srcGPU_, dstRank_, dstGPU_);

    // wait for recv mem handle
    MPI_Wait(&memReq_, MPI_STATUS_IGNORE);

    // convert to a pointer
    CUDA_RUNTIME(cudaSetDevice(srcDev_));
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&dstBuf_, memHandle_,
                                      cudaIpcMemLazyEnablePeerAccess));

    // block until we have sent event handle
    MPI_Wait(&evtReq_, MPI_STATUS_IGNORE);
  }

  void send(const void *devPtr, RcStream &stream) {
    assert(srcDev_ == stream.device());
    assert(dstBuf_);
    assert(devPtr);
    assert(dstDev_ >= 0);
    assert(srcDev_ >= 0);
    assert(bufSize_ > 0);
    CUDA_RUNTIME(cudaMemcpyPeerAsync(dstBuf_, dstDev_, devPtr, srcDev_,
                                     bufSize_, stream));
    // record the event
    CUDA_RUNTIME(cudaEventRecord(event_, stream));
  }

  void wait() { CUDA_RUNTIME(cudaEventSynchronize(event_)); }
};

class ColocatedDeviceRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstGPU_; // domain ID
  int dstDev_; // cuda ID

  cudaEvent_t event_;

  cudaIpcMemHandle_t memHandle_;
  cudaIpcEventHandle_t evtHandle_;
  MPI_Request memReq_;
  MPI_Request idReq_;
  MPI_Request evtReq_;

public:
  ColocatedDeviceRecver() : event_(0) {}
  ColocatedDeviceRecver(int srcRank, int srcGPU, int dstRank,
                        int dstGPU, // domain ID
                        int dstDev  // cuda ID
                        )
      : srcRank_(srcRank), srcGPU_(srcGPU), dstGPU_(dstGPU), dstDev_(dstDev),
        event_(0) {}
  ~ColocatedDeviceRecver() {
    if (event_) {
      CUDA_RUNTIME(cudaEventDestroy(event_));
    }
  }

  /*! prepare to recieve devPtr
   */
  void start_prepare(void *devPtr, const size_t numBytes) {
    // fprintf(stderr, "ColoDevRecv::start_prepare: send mem on %d(%d) to
    // r%dg%d\n", dstDev_, dstGPU_, srcRank_, srcGPU_);
    assert(devPtr);

    int payload = ((srcGPU_ & 0xFF) << 8) | (dstGPU_ & 0xFF);

    // recv the event handle
    MPI_Irecv(&evtHandle_, sizeof(evtHandle_), MPI_BYTE, srcRank_,
              make_tag<MsgKind::ColocatedEvt>(payload), MPI_COMM_WORLD,
              &evtReq_);

    // get an a memory handle
    CUDA_RUNTIME(cudaSetDevice(dstDev_));
    CUDA_RUNTIME(cudaIpcGetMemHandle(&memHandle_, devPtr));

    // Send the mem handle to the ColocatedSender
    MPI_Isend(&memHandle_, sizeof(memHandle_), MPI_BYTE, srcRank_,
              make_tag<MsgKind::ColocatedMem>(payload), MPI_COMM_WORLD,
              &memReq_);
    // Send the CUDA device id to the ColocatedSender
    MPI_Isend(&dstDev_, 1, MPI_INT, srcRank_,
              make_tag<MsgKind::ColocatedDev>(payload), MPI_COMM_WORLD,
              &idReq_);
  }

  void finish_prepare() {

    // wait to recv the event
    MPI_Wait(&evtReq_, MPI_STATUS_IGNORE);

    // convert event handle to event
    CUDA_RUNTIME(cudaSetDevice(dstDev_));
    CUDA_RUNTIME(cudaIpcOpenEventHandle(&event_, evtHandle_));

    // wait to send the mem handle and the CUDA device ID
    MPI_Wait(&memReq_, MPI_STATUS_IGNORE);
    MPI_Wait(&idReq_, MPI_STATUS_IGNORE);
  }

  /*! have stream wait for data to arrive
   */
  void wait(RcStream &stream) {
    assert(event_);
    assert(stream.device() == dstDev_);

    // wait for ColocatedDeviceSender cudaMemcpyPeerAsync to be done
    CUDA_RUNTIME(cudaStreamWaitEvent(stream, event_, 0 /*flags*/));
  }
};

class ColocatedHaloSender {
private:
  LocalDomain *domain_;
  int srcRank_;

  RcStream stream_;
  DevicePacker packer_;
  ColocatedDeviceSender sender_;

public:
  ColocatedHaloSender() {}
  ColocatedHaloSender(int srcRank, int srcGPU, int dstRank, int dstGPU,
                      LocalDomain &domain)
      : domain_(&domain), stream_(domain.gpu()),
        sender_(srcRank, srcGPU, dstRank, dstGPU, domain.gpu()) {}

  void start_prepare(const std::vector<Message> &outbox) {
    packer_.prepare(domain_, outbox);
    sender_.start_prepare(packer_.size());
  }

  void finish_prepare() { sender_.finish_prepare(); }

  void send() noexcept {
    packer_.pack(stream_);
    sender_.send(packer_.data(), stream_);
  }

  void wait() noexcept { sender_.wait(); }
};

class ColocatedHaloRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstGPU_;

  LocalDomain *domain_;

  RcStream stream_;
  ColocatedDeviceRecver recver_;
  DeviceUnpacker unpacker_;

public:
  ColocatedHaloRecver() {}
  ColocatedHaloRecver(int srcRank, int srcGPU, int dstRank, int dstGPU,
                      LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstGPU_(dstGPU), domain_(&domain),
        stream_(domain.gpu()),
        recver_(srcRank, srcGPU, dstRank, dstGPU, domain.gpu()) {}

  void start_prepare(const std::vector<Message> &inbox) {
    unpacker_.prepare(domain_, inbox);
    recver_.start_prepare(unpacker_.data(), unpacker_.size());
  }

  void finish_prepare() { recver_.finish_prepare(); }

  void recv() {
    assert(stream_.device() == domain_->gpu());
    recver_.wait(stream_);
    unpacker_.unpack(stream_);
  }

  void wait() noexcept {
    assert(stream_.device() == domain_->gpu());
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
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

  DevicePacker packer_;
  char *hostBuf_;

  RcStream stream_;
  MPI_Request req_;

  enum class State { None, D2H, Wait };
  State state_;

public:
  RemoteSender() : hostBuf_(nullptr) {}
  RemoteSender(int srcRank, int srcGPU, int dstRank, int dstGPU,
               LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), hostBuf_(nullptr), stream_(domain.gpu()),
        state_(State::None) {}

  ~RemoteSender() { CUDA_RUNTIME(cudaFreeHost(hostBuf_)); }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  virtual void prepare(std::vector<Message> &outbox) override {
#ifdef REMOTE_LOUD
    std::cerr << "RemoteSender::prepare(): " << outbox.size() << " messages\n";
#endif
    packer_.prepare(domain_, outbox);
    assert(packer_.size());
    assert(packer_.data());
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

// allocate device & host buffers
#ifdef REMOTE_LOUD
    std::cerr << "RemoteSender::prepare: alloc " << packer_.size() << "\n";
#endif
    CUDA_RUNTIME(
        cudaHostAlloc(&hostBuf_, packer_.size(), cudaHostAllocDefault));
    assert(hostBuf_);

    // fprintf(stderr, "RemoteSender::prepare r%dg%d -> r%dg%d %luB\n",
    // srcRank_, srcGPU_, dstRank_, dstGPU_, bufSize_);
  }

  virtual void send() override { send_d2h(); }

  virtual bool active() override {
    assert(State::None != state_);
    return State::Wait != state_;
  }

  virtual bool next_ready() override {
    assert(State::None != state_);
    if (state_ == State::D2H) {
      return d2h_done();
    } else {
      assert(0);
      __builtin_unreachable();
    }
  }

  virtual void next() override {
    if (State::D2H == state_) {
      send_h2h();
    } else {
      assert(0);
      __builtin_unreachable();
    }
  }

  virtual void wait() override {
    assert(State::Wait == state_);
    MPI_Wait(&req_, MPI_STATUS_IGNORE);
    state_ = State::None;
  }

  void send_d2h() {
    state_ = State::D2H;
    nvtxRangePush("RemoteSender::send_d2h");

    const Dim3 rawSz = domain_->raw_size();

    // pack data into device buffer
    packer_.pack(stream_);

    // copy to host buffer
    assert(hostBuf_);
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_, packer_.data(), packer_.size(),
                                 cudaMemcpyDefault, stream_));
    nvtxRangePop(); // RemoteSender::send_d2h
  }

  bool is_d2h() const noexcept { return State::D2H == state_; }

  bool d2h_done() {
    assert(State::D2H == state_);
    cudaError_t err = cudaStreamQuery(stream_);
    if (cudaSuccess == err) {
      return true;
    } else if (cudaErrorNotReady == err) {
      return false;
    } else {
      CUDA_RUNTIME(err);
      __builtin_unreachable();
    }
  }

  void send_h2h() {
    state_ = State::Wait;
    nvtxRangePush("RemoteSender::send_h2h");
    assert(hostBuf_);
    assert(packer_.size());
    assert(srcGPU_ < 8);
    assert(dstGPU_ < 8);
    const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
    MPI_Isend(hostBuf_, packer_.size(), MPI_BYTE, dstRank_, tag, MPI_COMM_WORLD,
              &req_);
    nvtxRangePop(); // RemoteSender::send_h2h
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

  DeviceUnpacker unpacker_;
  char *hostBuf_;

  RcStream stream_;

  MPI_Request req_;

  enum class State { None, H2H, H2D };
  State state_;

public:
  RemoteRecver() : hostBuf_(nullptr) {}
  RemoteRecver(int srcRank, int srcGPU, int dstRank, int dstGPU,
               LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), hostBuf_(nullptr), stream_(domain.gpu()),
        state_(State::None) {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  }

  ~RemoteRecver() { CUDA_RUNTIME(cudaFreeHost(hostBuf_)); }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  virtual void prepare(std::vector<Message> &inbox) override {
    unpacker_.prepare(domain_, inbox);
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // allocate device & host buffers
    CUDA_RUNTIME(
        cudaHostAlloc(&hostBuf_, unpacker_.size(), cudaHostAllocDefault));
    assert(hostBuf_);
    // fprintf(stderr, "RemoteRecver::prepare r%dg%d -> r%dg%d %luB\n",
    // srcRank_, srcGPU_, dstRank_, dstGPU_, bufSize_);
  }

  virtual void recv() override { recv_h2h(); }

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
      recv_h2d();
    } else {
      assert(0);
      __builtin_unreachable();
    }
  }

  virtual void wait() override {
    assert(State::H2D == state_);
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  void recv_h2d() {
    nvtxRangePush("RemoteRecver::recv_h2d");
    state_ = State::H2D;

    // copy to device buffer
    CUDA_RUNTIME(cudaMemcpyAsync(unpacker_.data(), hostBuf_, unpacker_.size(),
                                 cudaMemcpyDefault, stream_));

    unpacker_.unpack(stream_);
    nvtxRangePop(); // RemoteRecver::recv_h2d
  }

  bool is_h2h() const { return State::H2H == state_; }

  bool h2h_done() {
    assert(State::H2H == state_);
    int flag;
    MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      return true;
    } else {
      return false;
    }
  }

  void recv_h2h() {
    nvtxRangePush("RemoteRecver::recv_h2h");
    state_ = State::H2H;
    assert(hostBuf_);
    assert(unpacker_.size());
    assert(srcGPU_ < 8);
    assert(dstGPU_ < 8);
    const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
    MPI_Irecv(hostBuf_, unpacker_.size(), MPI_BYTE, srcRank_, tag,
              MPI_COMM_WORLD, &req_);
    nvtxRangePop(); // RemoteRecver::recv_h2h
  }
};

/*! Send from one domain to a remote domain
 */
class CudaAwareMpiSender : public StatefulSender {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  LocalDomain *domain_;

  DevicePacker packer_;

  RcStream stream_;
  MPI_Request req_;

  enum class State {
    None,
    Pack,
    Send,
  };
  State state_;

  std::vector<Message> outbox_;

public:
  CudaAwareMpiSender() {}
  CudaAwareMpiSender(int srcRank, int srcGPU, int dstRank, int dstGPU,
                     LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), stream_(domain.gpu()), state_(State::None) {}

  virtual void prepare(std::vector<Message> &outbox) override {
    packer_.prepare(domain_, outbox);
  }

  virtual void send() override { send_pack(); }

  virtual bool active() override { return State::Send != state_; }

  virtual bool next_ready() override {
    assert(State::Pack == state_);
    return pack_done();
  }

  virtual void next() override {
    if (State::Pack == state_) {
      send_d2d();
    } else {
      assert(0);
      __builtin_unreachable();
    }
  }

  virtual void wait() override {
    assert(State::Send == state_);
    MPI_Wait(&req_, MPI_STATUS_IGNORE);
  }

  void send_pack() {
    nvtxRangePush("CudaAwareMpiSender::send_pack");
    state_ = State::Pack;
    assert(packer_.data());
    packer_.pack(stream_);
    nvtxRangePop(); // CudaAwareMpiSender::send_pack
  }

  bool is_pack() const noexcept { return state_ == State::Pack; }

  bool pack_done() {
    cudaError_t err = cudaStreamQuery(stream_);
    if (cudaSuccess == err) {
      return true;
    } else if (cudaErrorNotReady == err) {
      return false;
    } else {
      CUDA_RUNTIME(err);
      exit(EXIT_FAILURE);
    }
  }

  void send_d2d() {
    nvtxRangePush("CudaAwareMpiSender::send_d2d");
    state_ = State::Send;
    assert(packer_.data());
    assert(srcGPU_ < 8);
    assert(dstGPU_ < 8);
    const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
    MPI_Isend(packer_.data(), packer_.size(), MPI_BYTE, dstRank_, tag,
              MPI_COMM_WORLD, &req_);
    nvtxRangePop(); // CudaAwareMpiSender::send_d2d
  }
};

class CudaAwareMpiRecver : public StatefulRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  LocalDomain *domain_;

  DeviceUnpacker unpacker_;
  RcStream stream_;
  MPI_Request req_;

  enum class State {
    None,
    Recv,
    Unpack,
  };
  State state_; // flattening

public:
  CudaAwareMpiRecver() {}
  CudaAwareMpiRecver(int srcRank, int srcGPU, int dstRank, int dstGPU,
                     LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), stream_(domain.gpu()), state_(State::None) {}

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  void prepare(std::vector<Message> &inbox) {
    unpacker_.prepare(domain_, inbox);
  }

  virtual void recv() override { recv_d2d(); }

  virtual bool active() override {
    assert(State::None != state_);
    return State::Unpack != state_;
  }

  virtual bool next_ready() override { return d2d_done(); }

  virtual void next() override {
    if (State::Recv == state_) {
      recv_unpack();
    } else {
      assert(0);
      __builtin_unreachable();
    }
  }

  virtual void wait() override {
    assert(State::Unpack == state_);
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  void recv_unpack() {
    nvtxRangePush("CudaAwareMpiRecver::recv_unpack");
    state_ = State::Unpack;
    unpacker_.unpack(stream_);
    nvtxRangePop(); // CudaAwareMpiRecver::recv_unpack
  }

  bool is_d2d() const { return state_ == State::Recv; }

  bool d2d_done() {
    int flag;
    MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      return true;
    } else {
      return false;
    }
  }

  void recv_d2d() {
    nvtxRangePush("CudaAwareMpiRecver::recv_d2d");
    state_ = State::Recv;
    assert(unpacker_.data());
    assert(srcGPU_ < 8);
    assert(dstGPU_ < 8);
    const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
    MPI_Irecv(unpacker_.data(), unpacker_.size(), MPI_BYTE, srcRank_, tag,
              MPI_COMM_WORLD, &req_);
    nvtxRangePop(); // CudaAwareMpiRecver::recv_d2d
  }
};


#undef ANY_LOUD
#undef REGION_LOUD
