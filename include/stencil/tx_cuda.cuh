#pragma once

#include <functional>
#include <future>

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
        dstGPU_(dstGPU), dir_(dir), stream_(domain.gpu()) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);
  }

  virtual void allocate() override {
    size_t totalBytes = 0;
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      totalBytes += domain_->halo_bytes(dir_, i);
    }
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
    hostBuf_.resize(totalBytes);
    CUDA_RUNTIME(cudaMalloc(&devBuf_, totalBytes));
  }

  void send_impl() {
    nvtxRangePush("RegionSender::send_impl");
    assert(bufs_.size() == senders_.size() && "was allocate called?");
    const Dim3 haloPos = domain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    size_t bufOffset = 0;
    assert(senders_.size() == domain_->num_data());
    const Dim3 rawSz = domain_->raw_size();
    // pack all regions into device buffer
    nvtxRangePush("RegionSender::send_impl() pack");
    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {
      const char *src = domain_->curr_data(idx);
      const size_t elemSize = domain_->elem_size(idx);

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);

      assert(stream.device() == domain_->gpu());
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
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
        dstGPU_(dstGPU), dir_(dir), stream_(domain.gpu()) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);
  }

  virtual void allocate() override {
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

  // one stream per domain data
  std::vector<RcStream> streams_;

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
    RcStream stream = streams_[dataIdx];

    size_t elemSize = srcDomain_->elem_size(dataIdx);
    assert(elemSize == dstDomain_->elem_size(dataIdx));

    // translate halo region to other
    const dim3 dimBlock(32, 4, 4);
    const dim3 dimGrid = (extent + (Dim3(dimBlock) - 1)) / Dim3(dimBlock);

    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    translate<<<dimGrid, dimBlock, 0, stream>>>(
        dst, dstPos, dstSize, src, srcPos, srcSize, extent, elemSize);
  }

public:
  RegionCopier(const LocalDomain &dstDomain, const LocalDomain &srcDomain,
               const Dim3 &dir)
      : srcDomain_(&srcDomain), dstDomain_(&dstDomain), dir_(dir) {

    assert(dstDomain.num_data() == srcDomain.num_data());

    // associate domain data array with a receiver and a stream
    for (size_t di = 0; di < srcDomain.num_data(); ++di) {
      streams_.push_back(RcStream(srcDomain_->gpu()));
    }
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
  virtual void wait() override {
    for (auto &s : streams_) {
      CUDA_RUNTIME(cudaStreamSynchronize(s))
    }
  }
};

/*! \brief Copy a LocalDomain region using pack/memcpy/unpack

For GPUs owned by the same process without peer access
 */
class PackMemcpyCopier : public HaloSender {
private:
  const LocalDomain *srcDomain_, *dstDomain_;

  Dim3 dir_;

  // one stream per domain data
  std::vector<RcStream> srcStreams_;
  std::vector<RcStream> dstStreams_;

  // one buffer on each src and dst GPU
  std::vector<void *> dstBufs_;
  std::vector<void *> srcBufs_;

  // one event per domain data to sync src and dst GPU
  std::vector<cudaEvent_t> events_;

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

    const dim3 dimBlock(32, 4, 4);
    const dim3 dimGrid = (extent + (Dim3(dimBlock) - 1)) / Dim3(dimBlock);

    // pack on source
    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    pack<<<dimGrid, dimBlock, 0, srcStreams_[dataIdx]>>>(
        srcBufs_[dataIdx], srcDomain_->curr_data(dataIdx), srcSize, 0 /*pitch*/,
        srcPos, extent, elemSize);

    // copy to dst
    size_t numBytes = srcDomain_->halo_bytes(dir_, dataIdx);
    CUDA_RUNTIME(cudaMemcpyAsync(dstBufs_[dataIdx], srcBufs_[dataIdx], numBytes,
                                 cudaMemcpyDefault, srcStreams_[dataIdx]));

    // record event in src stream
    CUDA_RUNTIME(cudaEventRecord(events_[dataIdx], srcStreams_[dataIdx]));

    // cause dst stream to wait on src event
    CUDA_RUNTIME(cudaStreamWaitEvent(dstStreams_[dataIdx], events_[dataIdx],
                                     0 /*flags*/));

    // unpack on dst
    CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
    unpack<<<dimGrid, dimBlock, 0, dstStreams_[dataIdx]>>>(
        dstDomain_->curr_data(dataIdx), dstSize, 0 /*pitch*/, dstPos, extent,
        dstBufs_[dataIdx], elemSize);
  }

public:
  PackMemcpyCopier(const LocalDomain &dstDomain, const LocalDomain &srcDomain,
                   const Dim3 &dir)
      : srcDomain_(&srcDomain), dstDomain_(&dstDomain), dir_(dir) {

    assert(dstDomain.num_data() == srcDomain.num_data());

    // associate domain data array a src stream, a dst stream, and an event
    for (size_t di = 0; di < srcDomain.num_data(); ++di) {
      int srcGpu = srcDomain_->gpu();
      CUDA_RUNTIME(cudaSetDevice(srcGpu));
      srcStreams_.push_back(RcStream(srcGpu));
      cudaEvent_t event;
      CUDA_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      events_.push_back(event);

      int dstGpu = dstDomain_->gpu();
      CUDA_RUNTIME(cudaSetDevice(dstGpu));
      dstStreams_.push_back(RcStream(dstGpu));
    }
  }

  virtual void allocate() override {
    assert(0 && "unimplemented");
    for (size_t dataIdx = 0; dataIdx < srcDomain_->num_data(); ++dataIdx) {
      size_t numBytes = srcDomain_->halo_bytes(dir_, dataIdx);
      void *p = nullptr;
      CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
      CUDA_RUNTIME(cudaMalloc(&p, numBytes));
      srcBufs_.push_back(p);
      p = nullptr;
      CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
      CUDA_RUNTIME(cudaMalloc(&p, numBytes));
      dstBufs_.push_back(p);
    }
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
  virtual void wait() override {
    for (auto &s : dstStreams_) {
      CUDA_RUNTIME(cudaStreamSynchronize(s));
    }
  }
};

#undef ANY_LOUD
#undef REGION_LOUD
