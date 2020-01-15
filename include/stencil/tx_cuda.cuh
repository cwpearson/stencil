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
            Dim3 dir) : AnySender(srcRank, srcGPU, dstRank, dstGPU, dataIdx, dir, RcStream(srcGPU)) {}

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
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), data, hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    int tag = make_tag(dstGPU, dataIdx, dir);
#ifdef ANY_LOUD
    printf(
        "[%d] AnySender::send_impl(): r%d,g%d,d%lu: Send %luB -> r%d,g%d,d%lu "
        "(tag=%08x)\n",
        getpid(), srcRank, srcGPU, dataIdx, hostBuf_.size(), dstRank, dstGPU,
        dataIdx, tag);
#endif
    MPI_Send(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, dstRank, tag,
             MPI_COMM_WORLD);

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
    MPI_Recv(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, srcRank, tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#ifdef ANY_LOUD
    fprintf(stderr, "[%d] AnyRecver::recver(): r%d,g%d: cudaMemcpyAsync\n",
            getpid(), dstRank, dstGPU);
#endif
    CUDA_RUNTIME(cudaMemcpyAsync(data, hostBuf_.data(), hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
#ifdef ANY_LOUD
    fprintf(stderr, "AnyRecver::recver(): wait for cuda sync\n");
#endif
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
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
            Dim3 dir) : AnyRecver(srcRank, srcGPU, dstRank, dstGPU, dataIdx, dir, RcStream(dstGPU)) {}

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

/*! Interface for sending any part of a halo anywhere
 */
class HaloSender {
public:
  // prepare to send the appropriate number of bytes
  virtual void allocate() = 0;

  // send the face data
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

  // recv the face data
  virtual void recv() = 0;

  // wait for send to be complete
  virtual void wait() = 0;
};

/*! Send a LocalDomain region using Sender
 */
template <typename Sender> class RegionSender : public HaloSender {
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
  RegionSender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
               size_t dstRank, size_t dstGPU, Dim3 dir)
      : domain_(&domain), dir_(dir) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      RcStream stream(domain_->gpu()); // associate stream with correct GPU
      senders_.push_back(Sender(srcRank, srcGPU, dstRank, dstGPU, di, dir_, stream));
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
    nvtxRangePush("RegionSender::send_impl");
    assert(bufs_.size() == senders_.size() && "was allocate called?");
    const Dim3 haloPos = domain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    // insert all packs into streams
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
    }

    // insert all sends into streams
    assert(senders_.size() == domain_->num_data());
    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {
      // copy to dst rank
      senders_[dataIdx].send(bufs_[dataIdx]);
    }

// wait for sends
#ifdef REGION_LOUD
    std::cerr << "RegionSender:send_impl(): " << dir_ << " wait for senders\n";
#endif
    for (auto &s : senders_) {
      s.wait();
    }

    nvtxRangePop();
  }

  void send() override { fut_ = std::async(&RegionSender::send_impl, this); }

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

/*! \brief Receive a LocalDomain face using Recver
 */
template <typename Recver> class RegionRecver : public HaloRecver {
private:
  const LocalDomain *domain_; // the domain we are receiving into
  Dim3 dir_;                  // the direction of the send we are recving

  std::future<void> fut_;

  // one receiver per domain data
  std::vector<Recver> recvers_;
  // one device buffer per domain data
  std::vector<char *> bufs_;
  // one stream per domain data
  std::vector<RcStream> streams_;

public:
  RegionRecver(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
               size_t dstRank, size_t dstGPU, const Dim3 &dir)
      : domain_(&domain), dir_(dir) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);

    // associate domain data array with a receiver and a stream
    for (size_t di = 0; di < domain.num_data(); ++di) {
      RcStream stream(domain_->gpu()); // associate stream with the cuda id
      recvers_.push_back(Recver(srcRank, srcGPU, dstRank, dstGPU, di, dir_, stream));
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

  void recv_impl() {
    assert(bufs_.size() == recvers_.size());

    // receive all data into flat buffers
    for (size_t dataIdx = 0; dataIdx < bufs_.size(); ++dataIdx) {
      recvers_[dataIdx].recv(bufs_[dataIdx]);
    }

    // have to call recver.wait() to ensure memcpy is inserted into stream
    // FIXME: here we wait for all recvers before doing any unpack.
    // FIXME: we only need to wait for the corresponding recvr
#ifdef REGION_LOUD
    std::cerr << "RegionRecver::recv_impl(): " << dir_ << " wait for recvers\n";
#endif
    for (auto &r : recvers_) {
      r.wait();
    }

    // insert unpacks into streams
    const Dim3 haloPos = domain_->halo_pos(dir_, true /*halo region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    const Dim3 rawSz = domain_->raw_size();
    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {

      char *dst = domain_->curr_data(dataIdx);
      RcStream stream = streams_[dataIdx];

      // unpack from buffer into halo
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(dataIdx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      unpack<<<dimGrid, dimBlock, 0, stream>>>(
          dst, rawSz, 0 /*pitch*/, haloPos, haloExtent, bufs_[dataIdx],
          elemSize);

    }

    // wait for unpacks
    for (auto &stream : streams_) {
      CUDA_RUNTIME(cudaStreamSynchronize(stream));
    }

  }

  void recv() override { fut_ = std::async(&RegionRecver::recv_impl, this); }

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

#undef ANY_LOUD
#undef REGION_LOUD
