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
  std::vector<RcStream> streams_;

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
      int dstDev = domains_[msg.dstGPU_]->gpu();
      if (srcDev >= streams_.size()) {
        streams_.resize(srcDev + 1);
      }
      if (dstDev >= streams_.size()) {
        streams_.resize(dstDev + 1);
      }
    }
    for (size_t i = 0; i < streams_.size(); ++i) {
      streams_[i] = RcStream(i);
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
    for (auto &s : streams_) {
      CUDA_RUNTIME(cudaStreamSynchronize(s));
    }
  }
};

/* Send messages between local domains by pack, cudaMemcpyPeerAsync, unpack
 */
class PeerCopySender {
private:
  size_t srcGPU_;
  size_t dstGPU_;
  const LocalDomain *srcDomain_;
  const LocalDomain *dstDomain_;

  std::vector<Message> outbox_;

  // one stream per source and destination operations
  RcStream srcStream_;
  RcStream dstStream_;

  // event to sync src and dst streams
  cudaEvent_t event_;

  // packed buffers
  void *srcBuf_; // src device buffer
  void *dstBuf_; // dst device buffer

  // buffer sizes
  size_t bufSize_;

  // offsets for each multi-pack on the device
  std::vector<size_t *> msgPackOffsets_;

public:
  PeerCopySender(size_t srcGPU, size_t dstGPU, const LocalDomain &srcDomain,
                 LocalDomain &dstDomain)
      : srcGPU_(srcGPU), dstGPU_(dstGPU), srcDomain_(&srcDomain),
        dstDomain_(&dstDomain), srcStream_(srcDomain.gpu()),
        dstStream_(dstDomain.gpu()) {}

  ~PeerCopySender() {}

  void prepare(std::vector<Message> &outbox) {
    outbox_ = outbox;
    std::sort(outbox_.begin(), outbox_.end());

    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));

    // create event
    CUDA_RUNTIME(cudaEventCreate(&event_));

    // compute buffer offsets for each message, and total buffer size
    std::vector<std::vector<size_t>> msgPackOffsets;
    bufSize_ = 0;
    for (const auto &msg : outbox_) {
      assert(msg.srcGPU_ == srcGPU_);
      assert(msg.dstGPU_ == dstGPU_);
      size_t offset = 0;
      std::vector<size_t> packOffsets;
      for (size_t i = 0; i < srcDomain_->num_data(); ++i) {
        packOffsets.push_back(offset);
        size_t newBytes = srcDomain_->halo_bytes(msg.dir_, i);
        offset += newBytes;
        bufSize_ += newBytes;
      }
      msgPackOffsets.push_back(packOffsets);
    }

    // store multi_pack and multi_unpack offets on GPU
    for (auto &offsets : msgPackOffsets) {
      size_t *p = nullptr;
      CUDA_RUNTIME(cudaMalloc(&p, offsets.size() * sizeof(offsets[0])));
      CUDA_RUNTIME(cudaMemcpy(p, offsets.data(),
                              offsets.size() * sizeof(offsets[0]),
                              cudaMemcpyHostToDevice));
      msgPackOffsets_.push_back(p);
    }
    assert(msgPackOffsets_.size() == outbox_.size());

    // initialize pack buffers
    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    CUDA_RUNTIME(cudaMalloc(&srcBuf_, bufSize_));
    CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
    CUDA_RUNTIME(cudaMalloc(&dstBuf_, bufSize_));
  }

  void send() {
    nvtxRangePush("PeerCopySender::send");
    assert(srcBuf_);
    assert(dstBuf_);

    const Dim3 dstSz = dstDomain_->raw_size();
    const Dim3 srcSz = srcDomain_->raw_size();

    // pack the data for each message into the src streams
    for (size_t mi = 0; mi < outbox_.size(); ++mi) {
      const auto &msg = outbox_[mi];
      const Dim3 pos = srcDomain_->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 extent = srcDomain_->halo_extent(msg.dir_);

      const dim3 dimBlock = make_block_dim(extent, 512 /*threads per block*/);
      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));

      // insert packs
      assert(srcStream_.device() == srcDomain_->gpu());
      assert(srcDomain_->num_data() == dstDomain_->num_data());
      CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));

      multi_pack<<<dimGrid, dimBlock, 0, srcStream_>>>(
          srcBuf_, msgPackOffsets_[mi], srcDomain_->dev_curr_datas(), srcSz,
          pos, extent, srcDomain_->dev_elem_sizes(), srcDomain_->num_data());
      CUDA_RUNTIME(cudaGetLastError());
    }

    // copy from src device to dst device
    const int dstDev = dstDomain_->gpu();
    const int srcDev = srcDomain_->gpu();
    CUDA_RUNTIME(cudaMemcpyPeerAsync(dstBuf_, dstDev, srcBuf_, srcDev, bufSize_,
                                     srcStream_));
    CUDA_RUNTIME(cudaEventRecord(event_, srcStream_));

    // insert unpacks
    for (size_t mi = 0; mi < outbox_.size(); ++mi) {
      const Message &msg = outbox_[mi];
      const Dim3 pos = dstDomain_->halo_pos(msg.dir_, true /*exterior*/);
      const Dim3 extent = dstDomain_->halo_extent(msg.dir_);

      const dim3 dimBlock = make_block_dim(extent, 512 /*threads per block*/);
      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));

      CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
      CUDA_RUNTIME(cudaStreamWaitEvent(dstStream_, event_, 0 /*flags*/));
      multi_unpack<<<dimGrid, dimBlock, 0, dstStream_>>>(
          dstDomain_->dev_curr_datas(), dstSz, pos, extent, dstBuf_,
          msgPackOffsets_[mi], dstDomain_->dev_elem_sizes(),
          dstDomain_->num_data());
      CUDA_RUNTIME(cudaGetLastError());
    }

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
  size_t bufSize_;

  RcStream stream_;
  MPI_Request req_;

  enum class State { None, D2H, Wait };
  State state_;

public:
  RemoteSender() : hostBuf_(nullptr) {}
  RemoteSender(int srcRank, int srcGPU, int dstRank, int dstGPU,
               LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), hostBuf_(nullptr),
        stream_(domain.gpu()), state_(State::None) {}

  ~RemoteSender() {
    CUDA_RUNTIME(cudaFreeHost(hostBuf_));
  }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  virtual void prepare(std::vector<Message> &outbox) override {
    packer_.prepare(domain_, outbox);
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

#ifdef REMOTE_LOUD
    std::cerr << "RemoteSender::prepare(): " << outbox_.size() << " messages\n";
#endif

// allocate device & host buffers
#ifdef REMOTE_LOUD
    std::cerr << "RemoteSender::prepare: alloc " << bufSize_ << "\n";
#endif
    CUDA_RUNTIME(cudaHostAlloc(&hostBuf_, packer_.size(), cudaHostAllocDefault));
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
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_, packer_.data(), packer_.size(), cudaMemcpyDefault,
                                 stream_));
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
    MPI_Isend(hostBuf_, bufSize_, MPI_BYTE, dstRank_, dstGPU_, MPI_COMM_WORLD,
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
  size_t bufSize_;

  RcStream stream_;

  MPI_Request req_;

  enum class State { None, H2H, H2D };
  State state_;

public:
  RemoteRecver() : hostBuf_(nullptr) {}
  RemoteRecver(int srcRank, int srcGPU, int dstRank, int dstGPU,
               LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), hostBuf_(nullptr),
        stream_(domain.gpu()), state_(State::None) {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  }

  ~RemoteRecver() {
    CUDA_RUNTIME(cudaFreeHost(hostBuf_));
  }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  virtual void prepare(std::vector<Message> &inbox) override {
    unpacker_.prepare(domain_, inbox);
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // allocate device & host buffers
    CUDA_RUNTIME(cudaHostAlloc(&hostBuf_, unpacker_.size(), cudaHostAllocDefault));
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
    CUDA_RUNTIME(cudaMemcpyAsync(unpacker_.data(), hostBuf_, unpacker_.size(), cudaMemcpyDefault,
                                 stream_));

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
    state_ = State::H2H;
    nvtxRangePush("RemoteRecver::recv_h2h");
    MPI_Irecv(hostBuf_, bufSize_, MPI_BYTE, srcRank_, srcGPU_, MPI_COMM_WORLD,
              &req_);
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
        domain_(&domain), stream_(domain.gpu()),
        state_(State::None) {}

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
    MPI_Isend(packer_.data(), packer_.size(), MPI_BYTE, dstRank_, dstGPU_, MPI_COMM_WORLD,
              &req_);
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

  std::vector<Message> inbox_;

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
        domain_(&domain), stream_(domain.gpu()),
        state_(State::None) {}

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
    assert(devBuf_);
    MPI_Irecv(unpacker_.data(), unpacker_.size(), MPI_BYTE, srcRank_, srcGPU_, MPI_COMM_WORLD,
              &req_);
    nvtxRangePop(); // CudaAwareMpiRecver::recv_d2d
  }
};

/*! A data sender that should work as long as MPI and CUDA are installed
1) cudaMemcpy from srcGPU to srcRank
2) MPI_Send from srcRank to dstRank, tagged with dstGPU
*/
class AnySender : public Sender {
private:
  int srcRank;
  int srcGPU_;
  int dstRank;
  int dstGPU;
  Dim3 dir;       // direction vector
  size_t dataIdx; // stencil data index

  RcStream stream_;
  std::vector<char> hostBuf_;
  std::future<void> waiter;

public:
  AnySender(int srcRank, int srcGPU, int dstRank, int dstGPU, size_t dataIdx,
            Dim3 dir, RcStream stream)
      : srcRank(srcRank), srcGPU_(srcGPU), dstRank(dstRank), dstGPU(dstGPU),
        dir(dir), dataIdx(dataIdx), stream_(stream) {}
  AnySender(int srcRank, int srcGPU, int dstRank, int dstGPU, size_t dataIdx,
            Dim3 dir)
      : AnySender(srcRank, srcGPU, dstRank, dstGPU, dataIdx, dir,
                  RcStream(srcGPU)) {}

  // copy ctor
  AnySender(const AnySender &other) = default;
  // move ctor
  AnySender(AnySender &&other) = default;
  // copy assignment
  AnySender &operator=(const AnySender &) = default;
  // move assignment
  AnySender &operator=(AnySender &&) = default;

  void resize(const size_t n) override { hostBuf_.resize(n); }

  /*! blocking send of data
   */
  void send_sync(const void *data) {
    nvtxRangePush("AnySender::send_impl");
    assert(data);
    assert(hostBuf_.data());
    assert(hostBuf_.size());
#ifdef ANY_LOUD
    printf("AnySender::send_impl(): r%d,g%d: cudaMemcpy\n", srcRank, srcGPU_);
#endif
    nvtxRangePush("memcpy");
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), data, hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    nvtxRangePop();
    int tag = make_tag(dstGPU, dataIdx, dir);
#ifdef ANY_LOUD
    printf("[%d] AnySender::send_impl(): r%d,g%d,d%lu: Send %luB -> "
           "r%d,g%d,d%lu "
           "(tag=%08x)\n",
           getpid(), srcRank, srcGPU, dataIdx, hostBuf_.size(), dstRank, dstGPU,
           dataIdx, tag);
#endif
    nvtxRangePush("send");
    MPI_Send(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, dstRank, tag,
             MPI_COMM_WORLD);
    nvtxRangePop();

#ifdef ANY_LOUD
    fprintf(stderr, "[%d] AnySender::send_impl(): r%d,g%d: finished Send\n",
            getpid(), srcRank, srcGPU);
#endif
    nvtxRangePop();
  }

  /*! async send of data
   */
  void send(const void *data) override {
    assert(data);
    waiter = std::async(std::launch::async, &AnySender::send_sync, this, data);
  }

  /*! wait for send()
   */
  void wait() override {
    if (waiter.valid()) {
      waiter.wait();
#ifdef ANY_LOUD
      printf("[%d] AnySender::wait(): r%d,g%d: done\n", getpid(), srcRank,
             srcGPU);
#endif
    } else {
      assert(0 && "wait called before send?");
    }
  }
};

/*! A data recver that should work as long as MPI and CUDA are installed
1) cudaMemcpy from srcGPU to srcRank
2) MPI_Send from srcRank to dstRank, tagged with dstGPU
*/
class AnyRecver : public Recver {
private:
  int srcRank;
  int srcGPU;
  int dstRank;
  int dstGPU;
  size_t dataIdx;
  Dim3 dir;

  RcStream stream_;
  std::vector<char> hostBuf_;
  std::future<void> waiter;

  void recver(void *data) {
    assert(data && "recv into null ptr");
    int tag = make_tag(dstGPU, dataIdx, dir);
#ifdef ANY_LOUD
    fprintf(
        stderr,
        "[%d] AnyRecver::recver(): r%d,g%d,d%lu Recv %luB from r%d,g%d,d%lu "
        "(tag=%08x)\n",
        getpid(), dstRank, dstGPU, dataIdx, hostBuf_.size(), srcRank, srcGPU,
        dataIdx, tag);
#endif
    assert(hostBuf_.size() && "internal buffer size 0");
    nvtxRangePush("recv");
    MPI_Recv(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, srcRank, tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    nvtxRangePop();
#ifdef ANY_LOUD
    fprintf(stderr, "[%d] AnyRecver::recver(): r%d,g%d: cudaMemcpyAsync\n",
            getpid(), dstRank, dstGPU);
#endif
    nvtxRangePush("memcpy");
    CUDA_RUNTIME(cudaMemcpyAsync(data, hostBuf_.data(), hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
#ifdef ANY_LOUD
    fprintf(stderr, "AnyRecver::recver(): wait for cuda sync\n");
#endif
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    nvtxRangePop();
#ifdef ANY_LOUD
    std::cerr << "AnyRecver::recver(): done cuda sync\n";
#endif
  }

public:
  AnyRecver(int srcRank, int srcGPU, int dstRank, int dstGPU, size_t dataIdx,
            Dim3 dir, RcStream stream)
      : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU),
        dataIdx(dataIdx), dir(dir), stream_(stream) {}
  AnyRecver(int srcRank, int srcGPU, int dstRank, int dstGPU, size_t dataIdx,
            Dim3 dir)
      : AnyRecver(srcRank, srcGPU, dstRank, dstGPU, dataIdx, dir,
                  RcStream(dstGPU)) {}

  // copy ctor
  AnyRecver(const AnyRecver &other) = default;
  // move ctyor
  AnyRecver(AnyRecver &&other) = default;
  // copy assignment
  AnyRecver &operator=(const AnyRecver &) = default;
  // move assignment
  AnyRecver &operator=(AnyRecver &&) = default;

  void resize(const size_t n) override { hostBuf_.resize(n); }

  void recv(void *data) override {
    assert(data);
    waiter = std::async(std::launch::async, &AnyRecver::recver, this, data);
  }

  void wait() override {
    if (waiter.valid()) {
      waiter.wait();
    } else {
      assert(0 && "wait called before recv?");
    }
  }
};

/*! A data sender that should work as long as CUDA is installed the two
devices are in the same process
*/
class DirectAccessCopier : public Copier {
private:
  int srcGPU;
  int dstGPU;
  size_t size_;
  RcStream stream_;

public:
  DirectAccessCopier(int srcGPU, int dstGPU, RcStream stream)
      : srcGPU(srcGPU), dstGPU(dstGPU), size_(0), stream_(stream) {}
  DirectAccessCopier(int srcGPU, int dstGPU)
      : DirectAccessCopier(srcGPU, dstGPU, RcStream(srcGPU)) {}

  // copy ctor
  DirectAccessCopier(const DirectAccessCopier &other) = default;
  // move ctor
  DirectAccessCopier(DirectAccessCopier &&other) = default;
  // copy assignment
  DirectAccessCopier &operator=(const DirectAccessCopier &) = default;
  // move assignment
  DirectAccessCopier &operator=(DirectAccessCopier &&) = default;

  void resize(const size_t n) override { size_ = n; }

  /*! async send/recv data
   */
  void copy(void *dst, const void *src) override {
    assert(dst);
    assert(src);
    CUDA_RUNTIME(cudaMemcpyPeerAsync(dst, dstGPU, src, srcGPU, size_, stream_));
  }

  /*! wait for send_recv()
   */
  void wait() override { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
};

/*! Interface for sending any part of a halo anywhere
 */
class HaloSender {
public:
  // prepare to send the appropriate number of bytes
  virtual void allocate() = 0;

  // send the halo data
  virtual void send() = 0;

  // wait for send to be complete
  virtual void wait() = 0;
};

/*! Interface for receiving any part of a halo from anywhere
 */
class HaloRecver {
public:
  // prepare to send the appropriate number of bytes
  virtual void allocate() = 0;

  // recv the halo data
  virtual void recv() = 0;

  // wait for send to be complete
  virtual void wait() = 0;
};

/*! Send a LocalDomain region using Sender
 */
template <typename Sender> class MTRegionSender : public HaloSender {
private:
  const LocalDomain *domain_; // the domain we are sending from
  Dim3 dir_;                  // the direction vector of the send

  std::future<void> fut_; // future for asyc call to send_impl

  // one sender per domain data
  std::vector<Sender> senders_;
  // one flattened device buffer per domain data
  std::vector<char *> bufs_;
  // one stream per domain data
  std::vector<RcStream> streams_;

public:
  MTRegionSender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
                 size_t dstRank, size_t dstGPU, Dim3 dir)
      : domain_(&domain), dir_(dir) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      RcStream stream(domain_->gpu()); // associate stream with correct GPU
      senders_.push_back(
          Sender(srcRank, srcGPU, dstRank, dstGPU, di, dir_, stream));
      streams_.push_back(stream);
    }
  }

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    // preallocate each sender
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->halo_bytes(dir_, i);
      char *buf = nullptr;
      CUDA_RUNTIME(cudaMalloc(&buf, numBytes));
      bufs_.push_back(buf);
      senders_[i].resize(numBytes);
    }
  }

  void send_impl() {
    nvtxRangePush("MTRegionSender::send_impl");
    assert(bufs_.size() == senders_.size() && "was allocate called?");
    const Dim3 haloPos = domain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    // insert all packs into streams and start sends
    assert(senders_.size() == domain_->num_data());
    const Dim3 rawSz = domain_->raw_size();
    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {
      const void *src = domain_->curr_data(idx);
      const size_t elemSize = domain_->elem_size(idx);
      RcStream stream = streams_[idx];

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);

      assert(stream.device() == domain_->gpu());
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      pack<<<dimGrid, dimBlock, 0, stream>>>(
          bufs_[idx], src, rawSz, 0 /*pitch*/, haloPos, haloExtent, elemSize);
      senders_[idx].send(bufs_[idx]);
    }

// wait for sends
#ifdef REGION_LOUD
    std::cerr << "MTRegionSender:send_impl(): " << dir_
              << " wait for senders\n";
#endif
    for (auto &s : senders_) {
      s.wait();
    }

    nvtxRangePop();
  }

  void send() override {
    fut_ = std::async(std::launch::async, &MTRegionSender::send_impl, this);
  }

  // wait for send to be complete
  void wait() override {
    if (fut_.valid()) {
      fut_.wait();
#ifdef REGION_LOUD
      std::cerr << "MTRegionSender::wait(): " << dir_ << " done\n";
#endif
    } else {
      assert(0 && "wait called before send()");
    }
  }
};

/*! \brief Receive a LocalDomain face using Recver
 */
template <typename Recver> class MTRegionRecver : public HaloRecver {
private:
  const LocalDomain *domain_; // the domain we are receiving into
  Dim3 dir_;                  // the direction of the send we are recving

  // one future per domain data
  std::vector<std::future<void>> futs_;
  // one receiver per domain data
  std::vector<Recver> recvers_;
  // one device buffer per domain data
  std::vector<char *> bufs_;
  // one stream per domain data
  std::vector<RcStream> streams_;

public:
  MTRegionRecver(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
                 size_t dstRank, size_t dstGPU, const Dim3 &dir)
      : domain_(&domain), dir_(dir) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);

    futs_.resize(domain.num_data());

    // associate domain data array with a receiver and a stream
    for (size_t di = 0; di < domain.num_data(); ++di) {
      RcStream stream(domain_->gpu()); // associate stream with the cuda id
      recvers_.push_back(
          Recver(srcRank, srcGPU, dstRank, dstGPU, di, dir_, stream));
      streams_.push_back(stream);
    }
  }

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->halo_bytes(dir_, i);
      char *buf = nullptr;
      CUDA_RUNTIME(cudaMalloc(&buf, numBytes));
      bufs_.push_back(buf);
      recvers_[i].resize(numBytes);
    }
  }

  void recv_impl(size_t dataIdx) {
    auto &recver = recvers_[dataIdx];
    recver.recv(bufs_[dataIdx]);
    recver.wait();

    const Dim3 haloPos = domain_->halo_pos(dir_, true /*halo region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    const Dim3 rawSz = domain_->raw_size();

    void *dst = domain_->curr_data(dataIdx);
    RcStream stream = streams_[dataIdx];

    // unpack from buffer into halo
    dim3 dimGrid(20, 20, 20);
    dim3 dimBlock(32, 4, 4);
    size_t elemSize = domain_->elem_size(dataIdx);
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
    unpack<<<dimGrid, dimBlock, 0, stream>>>(
        dst, rawSz, 0 /*pitch*/, haloPos, haloExtent, bufs_[dataIdx], elemSize);

    // wait for unpack
    CUDA_RUNTIME(cudaStreamSynchronize(stream));
  }

  void recv() override {
    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {
      futs_[dataIdx] = std::async(std::launch::async,
                                  &MTRegionRecver::recv_impl, this, dataIdx);
    }
  }

  // wait for send to be complete
  virtual void wait() override {

    for (auto &fut : futs_) {

      if (fut.valid()) {
        fut.wait();
#ifdef REGION_LOUD
        std::cerr << "MTRegionRecver::wait(): " << dir_ << " done\n";
#endif
      } else {
        assert(0 && "wait called before recv");
      }
    }
  }
};

/*! \brief Send a LocalDomain region

  All data fields are packed into a single message
 */
class RegionSender : public HaloSender {
private:
  const LocalDomain *domain_; // the domain we are sending from
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;
  Dim3 dir_; // the direction vector of the send

  std::future<void> fut_; // future for asyc call to send_impl

  // one flattened buffer with all domain data
  char *devBuf_;
  std::vector<char *> hostBuf_;

  // stream for packs and copy
  RcStream stream_;

public:
  RegionSender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
               size_t dstRank, size_t dstGPU, Dim3 dir)
      : domain_(&domain), srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank),
        dstGPU_(dstGPU), dir_(dir), devBuf_(nullptr), stream_(domain.gpu()) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);
  }

  ~RegionSender() { CUDA_RUNTIME(cudaFree(devBuf_)); }

  virtual void allocate() override {
    assert(nullptr == devBuf_);
    size_t totalBytes = 0;
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      totalBytes += domain_->halo_bytes(dir_, i);
    }
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
    hostBuf_.resize(totalBytes);
    CUDA_RUNTIME(cudaMalloc(&devBuf_, totalBytes));
  }

  void send_impl() {

    std::stringstream ss;
    ss << "RegionSender" << dir_;
    nvtxNameOsThread(pthread_self(), ss.str().c_str());

    nvtxRangePush("RegionSender::send_impl");

    assert(devBuf_ && "not allocated");
    assert(hostBuf_.size() && "not allocated");
    assert(stream_.device() == domain_->gpu());

    const Dim3 haloPos = domain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);
    const Dim3 rawSz = domain_->raw_size();

    // pack into buffer
    const dim3 dimGrid(20, 20, 20);
    const dim3 dimBlock(32, 4, 4);

    size_t bufOffset = 0;
    // pack all regions into device buffer
    nvtxRangePush("RegionSender::send_impl() pack");
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {
      const void *src = domain_->curr_data(idx);
      const size_t elemSize = domain_->elem_size(idx);

      pack<<<dimGrid, dimBlock, 0, stream_>>>(&devBuf_[bufOffset], src, rawSz,
                                              0 /*pitch*/, haloPos, haloExtent,
                                              elemSize);
      bufOffset += domain_->halo_bytes(dir_, idx);
    }
    nvtxRangePop(); // RegionSender::send_impl() pack

    // copy to host buffer
    nvtxRangePush("RegionSender::send_impl() cudaMemcpyAsync");
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), devBuf_, hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    nvtxRangePop(); // RegionSender::send_impl() cudaMemcpyAsync

    // Send to destination rank
    const int tag = make_tag(dstGPU_, dir_);
    nvtxRangePush("RegionSender::send_impl() MPI_Send");
    MPI_Send(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, dstRank_, tag,
             MPI_COMM_WORLD);
    nvtxRangePop(); // RegionSender::send_impl() MPI_Send

    nvtxRangePop(); // RegionSender::send_impl
  }

  void send() override {
    fut_ = std::async(std::launch::async, &RegionSender::send_impl, this);
  }

  // wait for send to be complete
  void wait() override {
    if (fut_.valid()) {
      fut_.wait();
#ifdef REGION_LOUD
      std::cerr << "RegionSender::wait(): " << dir_ << " done\n";
#endif
    } else {
      assert(0 && "wait called before send()");
    }
  }
};

/*! \brief Receive a LocalDomain region

    All data fields come in a single message
 */
class RegionRecver : public HaloRecver {
private:
  const LocalDomain *domain_; // the domain we are receiving into
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;
  Dim3 dir_; // the direction vector of the send we are recving

  std::future<void> fut_; // future for asyc call to recv_impl

  // one flattened buffer with all domain data
  char *devBuf_;
  std::vector<char *> hostBuf_;

  // stream for copy and unpacks
  RcStream stream_;

public:
  RegionRecver(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
               size_t dstRank, size_t dstGPU, const Dim3 &dir)
      : domain_(&domain), srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank),
        dstGPU_(dstGPU), dir_(dir), devBuf_(nullptr), stream_(domain.gpu()) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);
  }

  ~RegionRecver() { CUDA_RUNTIME(cudaFree(devBuf_)); }

  virtual void allocate() override {
    assert(nullptr == devBuf_);
    size_t totalBytes = 0;
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      totalBytes += domain_->halo_bytes(dir_, i);
    }
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
    hostBuf_.resize(totalBytes);
    CUDA_RUNTIME(cudaMalloc(&devBuf_, totalBytes));
  }

  void recv_impl() {

    // MPI_Recv
    const int tag = make_tag(dstGPU_, dir_);
    nvtxRangePush("RegionRecver::recv_impl() MPI_Recv");
    MPI_Recv(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, srcRank_, tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    nvtxRangePop(); // RegionRecver::recv_impl() MPI_Recv

    // copy to device
    nvtxRangePush("RegionRecver::recv_impl() cudaMemcpyAsync");
    CUDA_RUNTIME(cudaMemcpyAsync(devBuf_, hostBuf_.data(), hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    nvtxRangePop(); // RegionRecver::recv_impl() cudaMemcpyAsync

    // unpack into domain data
    nvtxRangePush("RegionRecver::recv_impl() unpack");
    size_t bufOffset = 0;
    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {

      const Dim3 haloPos = domain_->halo_pos(dir_, true /*halo region*/);
      const Dim3 haloExtent = domain_->halo_extent(dir_);

      const Dim3 rawSz = domain_->raw_size();

      void *dst = domain_->curr_data(idx);

      // unpack from buffer into halo
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(idx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      unpack<<<dimGrid, dimBlock, 0, stream_>>>(dst, rawSz, 0 /*pitch*/,
                                                haloPos, haloExtent,
                                                &devBuf_[bufOffset], elemSize);
      bufOffset += domain_->halo_bytes(dir_, idx);
    }

    // wait for unpack
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    nvtxRangePop(); // RegionRecver::recv_impl() unpack
  }

  void recv() override {
    fut_ = std::async(std::launch::async, &RegionRecver::recv_impl, this);
  }

  // wait for send to be complete
  virtual void wait() override {

    if (fut_.valid()) {
      fut_.wait();
#ifdef REGION_LOUD
      std::cerr << "RegionRecver::wait(): " << dir_ << " done\n";
#endif
    } else {
      assert(0 && "wait called before recv");
    }
  }
};

/*! \brief Copy a LocalDomain region using using Copier
 */
class RegionCopier : public HaloSender {
private:
  const LocalDomain *srcDomain_, *dstDomain_;
  Dim3 dir_;
  RcStream stream_;

  /* async copy data field dataIdx
   */
  void copy_data(size_t dataIdx) {

    const Dim3 srcPos = srcDomain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 dstPos = dstDomain_->halo_pos(dir_, true /*halo region*/);

    const Dim3 srcSize = srcDomain_->raw_size();
    const Dim3 dstSize = dstDomain_->raw_size();

    const Dim3 extent = srcDomain_->halo_extent(dir_);
    assert(extent == dstDomain_->halo_extent(dir_ * -1));

    assert(srcDomain_->num_data() == dstDomain_->num_data());

    void *dst = dstDomain_->curr_data(dataIdx);
    void *src = srcDomain_->curr_data(dataIdx);

    size_t elemSize = srcDomain_->elem_size(dataIdx);
    assert(elemSize == dstDomain_->elem_size(dataIdx));

    // translate halo region to other
    const dim3 dimBlock(32, 4, 4);
    const dim3 dimGrid = (extent + (Dim3(dimBlock) - 1)) / Dim3(dimBlock);

    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    translate<<<dimGrid, dimBlock, 0, stream_>>>(
        dst, dstPos, dstSize, src, srcPos, srcSize, extent, elemSize);
  }

public:
  RegionCopier(const LocalDomain &dstDomain, const LocalDomain &srcDomain,
               const Dim3 &dir)
      : srcDomain_(&srcDomain), dstDomain_(&dstDomain), dir_(dir),
        stream_(srcDomain.gpu()) {

    assert(dstDomain.num_data() == srcDomain.num_data());
  }

  virtual void allocate() override {
    // no intermediate storage needed
  }

  /* async copy
   */
  void send() override {
    // insert copies into streams
    for (size_t i = 0; i < srcDomain_->num_data(); ++i) {
      copy_data(i);
    }
  }

  // wait for send()
  virtual void wait() override { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
};

/*! \brief Copy a LocalDomain region using pack/memcpy/unpack

For GPUs owned by the same process without peer access
 */
class PackMemcpyCopier : public HaloSender {
private:
  const LocalDomain *srcDomain_, *dstDomain_;

  Dim3 dir_;

  // one stream per domain data
  RcStream srcStream_;
  RcStream dstStream_;

  // one buffer on each src and dst GPU
  char *srcBuf_;
  char *dstBuf_;

  // dst stream wait until src stream done
  cudaEvent_t event_;

  /* async copy data field dataIdx
   */
  void copy_data() {

    const Dim3 srcPos = srcDomain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 dstPos = dstDomain_->halo_pos(dir_, true /*halo region*/);

    const Dim3 srcSize = srcDomain_->raw_size();
    const Dim3 dstSize = dstDomain_->raw_size();

    const Dim3 extent = srcDomain_->halo_extent(dir_);
    assert(extent == dstDomain_->halo_extent(dir_ * -1));

    const dim3 dimBlock(32, 4, 4);
    const dim3 dimGrid = (extent + (Dim3(dimBlock) - 1)) / Dim3(dimBlock);

    assert(srcDomain_->num_data() == dstDomain_->num_data());

    // insert packs
    size_t bufOffset = 0;
    nvtxRangePush("packs");
    for (size_t idx = 0; idx < srcDomain_->num_data(); ++idx) {
      void *src = srcDomain_->curr_data(idx);

      const size_t elemSize = srcDomain_->elem_size(idx);
      assert(elemSize == dstDomain_->elem_size(idx));

      // pack on source
      CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
      pack<<<dimGrid, dimBlock, 0, srcStream_>>>(&srcBuf_[bufOffset], src,
                                                 srcSize, 0 /*pitch*/, srcPos,
                                                 extent, elemSize);
      bufOffset += srcDomain_->halo_bytes(dir_, idx);
    }
    nvtxRangePop();

    // copy to dst
    CUDA_RUNTIME(cudaMemcpyAsync(dstBuf_, srcBuf_, bufOffset, cudaMemcpyDefault,
                                 srcStream_));
    // record event in src stream
    CUDA_RUNTIME(cudaEventRecord(event_, srcStream_));

    // cause dst stream to wait on src event
    CUDA_RUNTIME(cudaStreamWaitEvent(dstStream_, event_, 0 /*flags*/));

    // insert unpacks
    bufOffset = 0;
    nvtxRangePush("unpacks");
    for (size_t idx = 0; idx < dstDomain_->num_data(); ++idx) {
      void *dst = dstDomain_->curr_data(idx);

      const size_t elemSize = srcDomain_->elem_size(idx);

      CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
      unpack<<<dimGrid, dimBlock, 0, dstStream_>>>(
          dst, dstSize, 0 /*pitch*/, dstPos, extent, &dstBuf_[bufOffset],
          elemSize);
      bufOffset += srcDomain_->halo_bytes(dir_, idx);
    }
  }

public:
  PackMemcpyCopier(const LocalDomain &dstDomain, const LocalDomain &srcDomain,
                   const Dim3 &dir)
      : srcDomain_(&srcDomain), dstDomain_(&dstDomain), dir_(dir),
        srcStream_(srcDomain.gpu()), dstStream_(dstDomain.gpu()) {

    assert(dstDomain.num_data() == srcDomain.num_data());

    CUDA_RUNTIME(cudaSetDevice(srcDomain.gpu()));
    CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  virtual void allocate() override {
    size_t totalBytes = 0;
    for (size_t idx = 0; idx < srcDomain_->num_data(); ++idx) {
      totalBytes += srcDomain_->halo_bytes(dir_, idx);
    }
    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    CUDA_RUNTIME(cudaMalloc(&srcBuf_, totalBytes));
    CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
    CUDA_RUNTIME(cudaMalloc(&dstBuf_, totalBytes));
  }

  /* async copy
   */
  void send() override { copy_data(); }

  // wait for send()
  virtual void wait() override {
    CUDA_RUNTIME(cudaStreamSynchronize(dstStream_));
  }
};

#undef ANY_LOUD
#undef REGION_LOUD
