#pragma once

#include <functional>
#include <future>
#include <sstream>

#include <mpi.h>

#include <nvToolsExt.h>

// getpid()
#include <sys/types.h>
#include <unistd.h>

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/rcstream.hpp"

#include "tx_common.hpp"

// #define ANY_LOUD
// #define REGION_LOUD
#define REMOTE_LOUD

class Message {
private:
public:
  Dim3 dir_;
  int srcGPU_;
  int dstGPU_;
  Message(Dim3 dir, int srcGPU, int dstGPU)
      : dir_(dir), srcGPU_(srcGPU), dstGPU_(dstGPU) {}

  bool operator<(const Message &rhs) const noexcept { return dir_ < rhs.dir_; }
};

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
    const dim3 dimBlock(8, 8, 8);
    for (auto &msg : outbox_) {
      const LocalDomain *srcDomain = domains_[msg.srcGPU_];
      const LocalDomain *dstDomain = domains_[msg.dstGPU_];
      const Dim3 dstSz = dstDomain->raw_size();
      const Dim3 srcSz = srcDomain->raw_size();
      const Dim3 srcPos = srcDomain->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 dstPos = dstDomain->halo_pos(msg.dir_, true /*exterior*/);
      const Dim3 extent = srcDomain->halo_extent(msg.dir_);
      RcStream &stream = streams_[srcDomain->gpu()];
      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      assert(stream.device() == srcDomain->gpu());
      CUDA_RUNTIME(cudaSetDevice(stream.device()));
      assert(srcDomain->num_data() == dstDomain->num_data());
      for (size_t i = 0; i < srcDomain->num_data(); ++i) {
        const char *src = srcDomain->curr_data(i);
        char *dst = dstDomain->curr_data(i);
        const size_t elemSz = srcDomain->elem_size(i);
        assert(dstDomain->elem_size(i) == elemSz);
        translate<<<dimGrid, dimBlock, 0, stream>>>(
            dst, dstPos, dstSz, src, srcPos, srcSz, extent, elemSz);
      }
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
  std::vector<Message> outbox_;
  std::vector<const LocalDomain *> domains_;

  // one stream per source and destination operations
  std::vector<std::vector<RcStream>> srcStreams_;
  std::vector<std::vector<RcStream>> dstStreams_;

  // event to sync src and dst streams
  std::vector<std::vector<cudaEvent_t>> events_;

  // packed buffers per (source, destination) domain pair
  std::vector<std::vector<void *>> srcBufs_; // src device buffers
  std::vector<std::vector<void *>> dstBufs_; // dst device buffers

  // buffer sizes
  std::vector<std::vector<size_t>> bufSizes_;

public:
  PeerCopySender() {}

  ~PeerCopySender() {}

  void prepare(std::vector<Message> &outbox,
               const std::vector<LocalDomain> &domains) {
    outbox_ = outbox;
    std::sort(outbox_.begin(), outbox_.end());

    for (auto &e : domains) {
      domains_.push_back(&e);
    }

    // initialize streams
    srcStreams_.resize(domains_.size());
    dstStreams_.resize(domains_.size());
    for (size_t i = 0; i < srcStreams_.size(); ++i) {
      srcStreams_[i].resize(domains_.size());
      dstStreams_[i].resize(domains_.size());
    }

    for (size_t i = 0; i < srcStreams_.size(); ++i) {
      for (size_t j = 0; j < srcStreams_[i].size(); ++j) {
        srcStreams_[i][j] = RcStream(i);
        dstStreams_[i][j] = RcStream(j);
      }
    }

    // compute buffer sizes
    bufSizes_ = std::vector<std::vector<size_t>>(
        domains_.size(), std::vector<size_t>(domains_.size(), 0));
    for (auto &msg : outbox_) {
      const auto *domain = domains_[msg.srcGPU_];
      for (size_t n = 0; n < domain->num_data(); ++n) {
        bufSizes_[msg.srcGPU_][msg.dstGPU_] += domain->halo_bytes(msg.dir_, n);
      }
    }

    // initialize buffers
    srcBufs_ = std::vector<std::vector<void *>>(
        domains_.size(), std::vector<void *>(domains_.size(), nullptr));
    dstBufs_ = std::vector<std::vector<void *>>(
        domains_.size(), std::vector<void *>(domains_.size(), nullptr));
    for (size_t i = 0; i < domains_.size(); ++i) {
      for (size_t j = 0; j < domains_.size(); ++j) {
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaMalloc(&srcBufs_[i][j], bufSizes_[i][j]));
        CUDA_RUNTIME(cudaSetDevice(j));
        CUDA_RUNTIME(cudaMalloc(&dstBufs_[i][j], bufSizes_[i][j]));
      }
    }
  }

  void send() {

    nvtxRangePush("PeerCopySender::send");

    // offsets for pack and unpack
    std::vector<std::vector<size_t>> bufOffsets(
        domains_.size(), std::vector<size_t>(domains_.size(), 0));

    // insert packs and copies into src stream, and unpacks into dst stream
    const dim3 dimBlock(8, 8, 8);
    for (auto &msg : outbox_) {
      const int i = msg.srcGPU_;
      const int j = msg.dstGPU_;
      const LocalDomain *srcDomain = domains_[i];
      const LocalDomain *dstDomain = domains_[j];
      const int srcDev = srcDomain->gpu();
      const int dstDev = dstDomain->gpu();
      RcStream &srcStream = srcStreams_[i][j];
      RcStream &dstStream = dstStreams_[i][j];
      cudaEvent_t event = events_[i][j];
      const Dim3 dstSz = dstDomain->raw_size();
      const Dim3 srcSz = srcDomain->raw_size();
      const Dim3 srcPos = srcDomain->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 dstPos = dstDomain->halo_pos(msg.dir_, true /*exterior*/);
      const Dim3 extent = srcDomain->halo_extent(msg.dir_);
      char *srcBuf = (char *)srcBufs_[i][j];
      char *dstBuf = (char *)dstBufs_[i][j];

      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));

      // insert packs
      assert(srcStream.device() == srcDomain->gpu());
      CUDA_RUNTIME(cudaSetDevice(srcStream.device()));
      assert(srcDomain->num_data() == dstDomain->num_data());
      size_t bufOffset = bufOffsets[i][j];
      for (size_t n = 0; n < srcDomain->num_data(); ++n) {
        const size_t elemSz = srcDomain->elem_size(n);
        const char *src = srcDomain->curr_data(n);
        pack<<<dimGrid, dimBlock, 0, srcStream>>>(
            &srcBuf[bufOffset], src, srcSz, 0, srcPos, extent, elemSz);
        bufOffset += srcDomain->halo_bytes(msg.dir_, n);
      }
      // insert copy
      CUDA_RUNTIME(cudaMemcpyPeerAsync(dstBuf, dstDev, srcBuf, srcDev,
                                       bufSizes_[i][j], srcStream));
      CUDA_RUNTIME(cudaEventRecord(event, srcStream));

      // insert dsts
      CUDA_RUNTIME(cudaSetDevice(dstStream.device()));
      bufOffset = bufOffsets[i][j];
      for (size_t n = 0; n < dstDomain->num_data(); ++n) {
        const size_t elemSz = dstDomain->elem_size(n);
        char *dst = dstDomain->curr_data(n);
        CUDA_RUNTIME(cudaStreamWaitEvent(dstStream, event, 0 /*flags*/));
        unpack<<<dimGrid, dimBlock, 0, dstStream>>>(dst, dstSz, 0, dstPos, extent,
                                                    &dstBuf[bufOffset], elemSz);
        bufOffset += dstDomain->halo_bytes(msg.dir_, n);
      }
      bufOffsets[i][j] = bufOffset;
    }

    nvtxRangePop(); // PeerCopySender::send
  }

  void wait() {
    for (auto &v : srcStreams_) {
      for (auto &stream : v) {
        cudaStreamSynchronize(stream);
      }
    }
    for (auto &v : dstStreams_) {
      for (auto &stream : v) {
        cudaStreamSynchronize(stream);
      }
    }
  }
};

/*! Send from one domain to a remote domain
 */
class RemoteSender {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  const LocalDomain *domain_;

  char *devBuf_;
  char *hostBuf_;
  size_t bufSize_;

  RcStream stream_;
  MPI_Request req_;

  bool isD2h_; // in d2h phase

  std::vector<Message> outbox_;

public:
  RemoteSender() : devBuf_(nullptr) {}
  RemoteSender(int srcRank, int srcGPU, int dstRank, int dstGPU,
               const LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), devBuf_(nullptr), hostBuf_(nullptr),
        stream_(domain.gpu()), isD2h_(false) {}

  ~RemoteSender() {
    CUDA_RUNTIME(cudaFree(devBuf_));
    CUDA_RUNTIME(cudaFreeHost(hostBuf_));
  }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  void prepare(std::vector<Message> &outbox) {
    outbox_ = outbox;
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

#ifdef REMOTE_LOUD
    std::cerr << "RemoteSender::prepare(): " << outbox_.size() << " messages\n";
#endif

    // sort messages by direction vector
    std::sort(outbox_.begin(), outbox_.end());

    // compute total size
    bufSize_ = 0;
    for (auto &msg : outbox_) {
      for (size_t i = 0; i < domain_->num_data(); ++i) {
        bufSize_ += domain_->halo_bytes(msg.dir_, i);
      }
    }

// allocate device & host buffers
#ifdef REMOTE_LOUD
    std::cerr << "RemoteSender::prepare: alloc " << bufSize_ << "\n";
#endif
    CUDA_RUNTIME(cudaMalloc(&devBuf_, bufSize_));
    CUDA_RUNTIME(cudaHostAlloc(&hostBuf_, bufSize_, cudaHostAllocDefault));
  }

  void send_d2h() {
    nvtxRangePush("RemoteSender::send_d2h");
    isD2h_ = true;

    const Dim3 rawSz = domain_->raw_size();

    // pack data into device buffer
    dim3 dimBlock(8, 8, 8);
    size_t bufOffset = 0;
    for (auto &msg : outbox_) {
      const Dim3 pos = domain_->halo_pos(msg.dir_, false /*compute region*/);
      const Dim3 extent = domain_->halo_extent(msg.dir_);
      dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));

      for (size_t i = 0; i < domain_->num_data(); ++i) {
        const char *src = domain_->curr_data(i);
        const size_t elemSz = domain_->elem_size(i);
        assert(stream_.device() == domain_->gpu());
        CUDA_RUNTIME(cudaSetDevice(stream_.device()));
        pack<<<dimGrid, dimBlock, 0, stream_>>>(&devBuf_[bufOffset], src, rawSz,
                                                0, pos, extent, elemSz);
        bufOffset += domain_->halo_bytes(msg.dir_, i);
      }
    }

    // copy to host buffer
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_, devBuf_, bufSize_, cudaMemcpyDefault,
                                 stream_));
    nvtxRangePop(); // RemoteSender::send_d2h
  }

  bool is_d2h() { return isD2h_; }

  bool d2h_done() {
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

  void send_h2h() {
    nvtxRangePush("RemoteSender::send_h2h");
    isD2h_ = false;
    MPI_Isend(hostBuf_, bufSize_, MPI_BYTE, dstRank_, dstGPU_, MPI_COMM_WORLD,
              &req_);
    nvtxRangePop(); // RemoteSender::send_h2h
  }

  void wait() { MPI_Wait(&req_, MPI_STATUS_IGNORE); }
};

/*! Recv from a remote domain into a domain
 */
class RemoteRecver {
private:
  int srcRank_;
  int srcGPU_;
  int dstRank_;
  int dstGPU_;

  const LocalDomain *domain_;

  char *devBuf_;
  char *hostBuf_;
  size_t bufSize_;

  RcStream stream_;

  MPI_Request req_;

  bool isH2h_; // in d2h phase

  std::vector<Message> inbox_;

public:
  RemoteRecver() : devBuf_(nullptr) {}
  RemoteRecver(int srcRank, int srcGPU, int dstRank, int dstGPU,
               const LocalDomain &domain)
      : srcRank_(srcRank), srcGPU_(srcGPU), dstRank_(dstRank), dstGPU_(dstGPU),
        domain_(&domain), devBuf_(nullptr), hostBuf_(nullptr),
        stream_(domain.gpu()), isH2h_(false) {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  }

  ~RemoteRecver() {
    CUDA_RUNTIME(cudaFree(devBuf_));
    CUDA_RUNTIME(cudaFreeHost(hostBuf_));
  }

  /*! Prepare to send a set of messages whose direction vectors are store in
   * outbox
   */
  void prepare(std::vector<Message> &inbox) {
    inbox_ = inbox;
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // sort messages by direction vector
    std::sort(inbox_.begin(), inbox_.end());

    // compute total size
    bufSize_ = 0;
    for (auto &msg : inbox) {
      for (size_t i = 0; i < domain_->num_data(); ++i) {
        bufSize_ += domain_->halo_bytes(msg.dir_, i);
      }
    }

    // allocate device & host buffers
    CUDA_RUNTIME(cudaMalloc(&devBuf_, bufSize_));
    CUDA_RUNTIME(cudaHostAlloc(&hostBuf_, bufSize_, cudaHostAllocDefault));
  }

  void recv_h2d() {
    nvtxRangePush("RemoteRecver::recv_h2d");
    isH2h_ = false;

    // copy to device buffer
    CUDA_RUNTIME(cudaMemcpyAsync(devBuf_, hostBuf_, bufSize_, cudaMemcpyDefault,
                                 stream_));

    const Dim3 rawSz = domain_->raw_size();

    // pack data into device buffer
    dim3 dimBlock(8, 8, 8);
    size_t bufOffset = 0;
    for (auto &msg : inbox_) {
      const Dim3 pos = domain_->halo_pos(msg.dir_, true /*halo region*/);
      const Dim3 extent = domain_->halo_extent(msg.dir_);
      dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      for (size_t i = 0; i < domain_->num_data(); ++i) {
        char *dst = domain_->curr_data(i);
        const size_t elemSz = domain_->elem_size(i);
        CUDA_RUNTIME(cudaSetDevice(stream_.device()));
        assert(stream_.device() == domain_->gpu());
        unpack<<<dimGrid, dimBlock, 0, stream_>>>(dst, rawSz, 0, pos, extent,
                                                  &devBuf_[bufOffset], elemSz);
        bufOffset += domain_->halo_bytes(msg.dir_, i);
      }
    }
    nvtxRangePop(); // RemoteRecver::recv_h2d
  }

  bool is_h2h() const { return isH2h_; }

  bool h2h_done() {
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
    isH2h_ = true;
    MPI_Irecv(hostBuf_, bufSize_, MPI_BYTE, srcRank_, srcGPU_, MPI_COMM_WORLD,
              &req_);
    nvtxRangePop(); // RemoteRecver::recv_h2h
  }

  /*! wait for recv_h2d
   */
  void wait() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
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
    printf(
        "[%d] AnySender::send_impl(): r%d,g%d,d%lu: Send %luB -> r%d,g%d,d%lu "
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

/*! A data sender that should work as long as CUDA is installed the two devices
are in the same process
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
      const char *src = domain_->curr_data(idx);
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

    char *dst = domain_->curr_data(dataIdx);
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
      const char *src = domain_->curr_data(idx);
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

      char *dst = domain_->curr_data(idx);

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

    char *dst = dstDomain_->curr_data(dataIdx);
    char *src = srcDomain_->curr_data(dataIdx);

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
      char *src = srcDomain_->curr_data(idx);

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
      char *dst = dstDomain_->curr_data(idx);

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
