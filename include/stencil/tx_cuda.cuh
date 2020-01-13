#pragma once

#include <functional>
#include <future>

#include <mpi.h>

// getpid()
#include <sys/types.h>
#include <unistd.h>

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/rcstream.hpp"

#include "tx_common.hpp"

/*! A data sender that should work as long as MPI and CUDA are installed
1) cudaMemcpy from srcGPU to srcRank
2) MPI_Send from srcRank to dstRank, tagged with dstGPU
*/
class AnySender : public Sender {
private:
  int srcRank;
  int srcGPU;
  int dstRank;
  int dstGPU;
  Dim3 dir;       // direction vector
  size_t dataIdx; // stencil data index

  RcStream stream_;
  std::vector<char> hostBuf_;
  std::future<void> waiter;

  void sender(const void *data) {
    assert(data);
    printf("AnySender::sender(): r%d,g%d: cudaMemcpy\n", srcRank, srcGPU);
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), data, hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    int tag = make_tag(dstGPU, dataIdx, dir);
    printf("[%d] AnySender::sender(): r%d,g%d,d%lu: Isend %luB -> r%d,g%d,d%lu "
           "(tag=%08x)\n",
           getpid(), srcRank, srcGPU, dataIdx, hostBuf_.size(), dstRank, dstGPU,
           dataIdx, tag);
    MPI_Request req;
    assert(hostBuf_.data());
    MPI_Isend(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, dstRank, tag,
              MPI_COMM_WORLD, &req);
    MPI_Status stat;
    printf("[%d] AnySender::sender(): r%d,g%d: wait on Isend\n", getpid(),
           srcRank, srcGPU);
    MPI_Wait(&req, &stat);
    printf("[%d] AnySender::sender(): r%d,g%d: finished Isend\n", getpid(),
           srcRank, srcGPU);
  }

public:
  AnySender(int srcRank, int srcGPU, int dstRank, int dstGPU, size_t dataIdx,
            Dim3 dir)
      : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU),
        dir(dir), dataIdx(dataIdx), stream_(srcGPU) {}

  // copy ctor
  AnySender(const AnySender &other) = default;
  // move ctor
  AnySender(AnySender &&other) = default;
  // copy assignment
  AnySender &operator=(const AnySender &) = default;
  // move assignment
  AnySender &operator=(AnySender &&) = default;

  void resize(const size_t n) override { hostBuf_.resize(n); }

  void send(const void *data) override {
    waiter = std::async(&AnySender::sender, this, data);
  }

  void wait() override {
    if (waiter.valid()) {
      waiter.wait();
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
    MPI_Request req;
    MPI_Status stat;
    int tag = make_tag(dstGPU, dataIdx, dir);
    printf("AnyRecver::recver(): r%d,g%d,d%lu Irecv %luB from r%d,g%d,d%lu "
           "(tag=%08x)\n",
           dstRank, dstGPU, dataIdx, hostBuf_.size(), srcRank, srcGPU, dataIdx,
           tag);
    assert(hostBuf_.size() && "internal buffer size 0");
    MPI_Irecv(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, srcRank, tag,
              MPI_COMM_WORLD, &req);
    printf("[%d] AnyRecver::recver(): r%d,g%d: wait on Irecv\n", getpid(),
           dstRank, dstGPU);
    MPI_Wait(&req, &stat);
    printf("[%d] AnyRecver::recver(): r%d,g%d: got Irecv. cudaMemcpyAsync\n", getpid(),
           dstRank, dstGPU);
    CUDA_RUNTIME(cudaMemcpyAsync(data, hostBuf_.data(), hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    printf("AnyRecver::recver(): wait for cuda sync\n");
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    std::cerr << "AnyRecver::recver(): done cuda sync\n";
  }

public:
  AnyRecver(int srcRank, int srcGPU, int dstRank, int dstGPU, size_t dataIdx,
            Dim3 dir)
      : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU),
        dataIdx(dataIdx), dir(dir), stream_(dstGPU) {}

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
    waiter = std::async(&AnyRecver::recver, this, data);
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


/*! \brief Receive a LocalDomain edge using Recver
 */
template <typename Recver> class EdgeRecver : public HaloRecver {
private:
  const LocalDomain *domain_; // the domain we are receiving into
  size_t dim0_, dim1_;        // the two dimensions the edge shares
  bool dim0Pos_;              // postive dimension 0
  bool dim1Pos_;              // positive dimension 1

  std::future<void> fut_;

  // one receiver per domain data
  std::vector<Recver> recvers_;
  // one device buffer per domain data
  std::vector<char *> bufs_;

public:
  EdgeRecver(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
             size_t dstRank, size_t dstGPU, size_t dim0, size_t dim1,
             bool dim0Pos, bool dim1Pos)
      : domain_(&domain), dim0_(dim0), dim0Pos_(dim0Pos) {

    Dim3 dir(0, 0, 0);
    dir[dim0] += (dim0Pos > 0 ? 1 : -1);
    dir[dim1] += (dim1Pos > 0 ? 1 : -1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      recvers_.push_back(Recver(srcRank, srcGPU, dstRank, dstGPU, di, dir));
    }
  }

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->edge_bytes(dim0_, dim1_, i);
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

    // wait for all data
    for (auto &r : recvers_) {
      r.wait();
    }

    // unpack all data into halo region
    const Dim3 edgePos =
        domain_->edge_pos(dim0_, dim1_, dim0Pos_, dim1Pos_, true /*halo*/);
    const Dim3 edgeExtent = domain_->edge_extent(dim0_, dim1_);

    const Dim3 rawSz = domain_->raw_size();
    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {

      char *dst = domain_->curr_data(dataIdx);
      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(dataIdx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      unpack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          dst, rawSz, 0 /*pitch*/, edgePos, edgeExtent, bufs_[dataIdx],
          elemSize);
      CUDA_RUNTIME(cudaStreamSynchronize(domain_->stream()));
    }
  }

  void recv() override { fut_ = std::async(&EdgeRecver::recv_impl, this); }

  // wait for send to be complete
  virtual void wait() override {
    if (fut_.valid()) {
      fut_.wait();
    } else {
      assert(0 && "wait called before recv");
    }
  }
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

public:
  RegionSender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
               size_t dstRank, size_t dstGPU, Dim3 dir)
      : domain_(&domain), dir_(dir) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      senders_.push_back(Sender(srcRank, srcGPU, dstRank, dstGPU, di, dir_));
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
    assert(bufs_.size() == senders_.size() && "was allocate called?");
    const Dim3 haloPos = domain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    const Dim3 rawSz = domain_->raw_size();
    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {
      const char *src = domain_->curr_data(idx);
      const size_t elemSize = domain_->elem_size(idx);

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);

      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      pack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          bufs_[idx], src, rawSz, 0 /*pitch*/, haloPos, haloExtent, elemSize);
    }
    // wait for packs
    CUDA_RUNTIME(cudaStreamSynchronize(domain_->stream()));

    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {
      // copy to dst rank
      printf("RegionSender::send_impl(): send data %lu\n", dataIdx);
      senders_[dataIdx].send(bufs_[dataIdx]);
    }

    // wait for sends
    std::cerr << "RegionSender::wait() for senders\n";
    for (auto &s : senders_) {
      s.wait();
    }
  }

  void send() override { fut_ = std::async(&RegionSender::send_impl, this); }

  // wait for send to be complete
  void wait() override {
    if (fut_.valid()) {
      std::cout << "RegionSender::wait() for fut_\n";
      fut_.wait();
      std::cout << "RegionSender::wait() done\n";
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

public:
  RegionRecver(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
               size_t dstRank, size_t dstGPU, const Dim3 &dir)
      : domain_(&domain), dir_(dir) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      recvers_.push_back(Recver(srcRank, srcGPU, dstRank, dstGPU, di, dir_));
    }
  }

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->halo_bytes(dir_, i);
      std::cerr << "allocate " << numBytes << "\n";
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

    // wait for all data
    for (auto &r : recvers_) {
      r.wait();
    }

    // unpack all data into domain's halo
    const Dim3 haloPos = domain_->halo_pos(dir_, true /*halo region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);

    const Dim3 rawSz = domain_->raw_size();
    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {

      char *dst = domain_->curr_data(dataIdx);

      // unpack from buffer into halo
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(dataIdx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

      unpack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          dst, rawSz, 0 /*pitch*/, haloPos, haloExtent, bufs_[dataIdx],
          elemSize);

      CUDA_RUNTIME(cudaStreamSynchronize(domain_->stream()));
    }
  }

  void recv() override { fut_ = std::async(&RegionRecver::recv_impl, this); }

  // wait for send to be complete
  virtual void wait() override {
    if (fut_.valid()) {
      std::cerr << "RegionRecver::wait() for fut_\n";
      fut_.wait();
      std::cerr << "RegionRecver::wait() done\n";
    } else {
      assert(0 && "wait called before recv");
    }
  }
};