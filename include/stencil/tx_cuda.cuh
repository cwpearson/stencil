#pragma once

#include <functional>
#include <future>

#include <mpi.h>

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
    printf("AnySender::sender(): r%d,g%d: cudaMemcpy\n", srcRank, srcGPU);
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), data, hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    int tag = make_tag(dstGPU, dataIdx, dir);
    printf("AnySender::sender(): r%d,g%d,d%lu: Isend %luB -> r%d,g%d,d%lu "
           "(tag=%08x)\n",
           srcRank, srcGPU, dataIdx, hostBuf_.size(), dstRank, dstGPU, dataIdx,
           tag);
    MPI_Request req;
    MPI_Isend(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, dstRank, tag,
              MPI_COMM_WORLD, &req);
    MPI_Status stat;
    printf("AnySender::wait(): r%d,g%d: wait on Isend\n", srcRank, srcGPU);
    MPI_Wait(&req, &stat);
    printf("AnySender::wait(): r%d,g%d: finished Isend\n", srcRank, srcGPU);
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
    MPI_Request req;
    MPI_Status stat;
    int tag = make_tag(dstGPU, dataIdx, dir);
    printf("AnyRecver::recver(): r%d,g%d,d%lu Irecv %luB from r%d,g%d,d%lu "
           "(tag=%08x)\n",
           dstRank, dstGPU, dataIdx, hostBuf_.size(), srcRank, srcGPU, dataIdx,
           tag);
    MPI_Irecv(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, srcRank, tag,
              MPI_COMM_WORLD, &req);
    printf("AnyRecver::recver(): r%d,g%d: wait on Irecv\n", dstRank, dstGPU);
    MPI_Wait(&req, &stat);
    printf("AnyRecver::recver(): r%d,g%d: got Irecv. cudaMemcpyAsync\n",
           dstRank, dstGPU);
    CUDA_RUNTIME(cudaMemcpyAsync(data, hostBuf_.data(), hostBuf_.size(),
                                 cudaMemcpyDefault, stream_));
    printf("AnyRecver::recver(): wait for cuda sync\n");
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
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

/*! Send a LocalDomain face using Sender
 */
template <typename Sender> class FaceSender : public HaloSender {
private:
  const LocalDomain *domain_; // the domain we are sending from
  size_t dim_;                // the face dimension we are sending
  bool pos_;                  // positive or negative face

  std::future<void> fut_; // future for asyc call to send_impl

  // one sender per domain data
  std::vector<Sender> senders_;
  // one device buffer per domain data
  std::vector<char *> bufs_;

public:
  FaceSender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
             size_t dstRank, size_t dstGPU, size_t dim, bool pos)
      : domain_(&domain), dim_(dim), pos_(pos) {

    Dim3 dir(0, 0, 0);
    dir[dim] += (pos > 0 ? 1 : -1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      senders_.push_back(Sender(srcRank, srcGPU, dstRank, dstGPU, di, dir));
    }
  }

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    // preallocate each sender
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->face_bytes(dim_, i);
      // printf("FaceSender::allocate():  alloc %lu on gpu %d\n", numBytes,
      // gpu);
      char *buf = nullptr;
      CUDA_RUNTIME(cudaMalloc(&buf, numBytes));
      bufs_.push_back(buf);
      senders_[i].resize(numBytes);
    }
  }

  void send_impl() {
    assert(bufs_.size() == senders_.size() && "was allocate called?");
    const Dim3 facePos =
        domain_->face_pos(dim_, pos_, false /*compute region*/);
    const Dim3 faceExtent = domain_->face_extent(dim_);

    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {
      const Dim3 rawSz = domain_->raw_size(idx);
      const char *src = domain_->curr_data(idx);
      const size_t elemSize = domain_->elem_size(idx);

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);

      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      pack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          bufs_[idx], src, rawSz, 0 /*pitch*/, facePos, faceExtent, elemSize);
    }
    // wait for packs
    CUDA_RUNTIME(cudaStreamSynchronize(domain_->stream()));

    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {
      // copy to dst rank
      printf("FaceSender::send_impl(): send data %lu\n", dataIdx);
      senders_[dataIdx].send(bufs_[dataIdx]);
    }
  }

  void send() override { fut_ = std::async(&FaceSender::send_impl, this); }

  // wait for send to be complete
  void wait() override {
    if (fut_.valid()) {
      std::cout << "FaceSender::wait() for fut_\n";
      fut_.wait();
      std::cout << "FaceSender::wait() for senders\n";
      for (auto &s : senders_) {
        s.wait();
      }
      std::cout << "FaceSender::wait() done\n";
    } else {
      assert(0 && "wait called before send()");
    }
  }
};

/*! \brief Receive a LocalDomain face using Recver
 */
template <typename Recver> class FaceRecver : public HaloRecver {
private:
  const LocalDomain *domain_; // the domain we are receiving into
  size_t dim_;                // the face dimension we are receiving
  bool pos_;                  // positive or negative face

  std::future<void> fut_;

  // one receiver per domain data
  std::vector<Recver> recvers_;
  // one device buffer per domain data
  std::vector<char *> bufs_;

public:
  FaceRecver(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
             size_t dstRank, size_t dstGPU, size_t dim, bool pos)
      : domain_(&domain), dim_(dim), pos_(pos) {

    Dim3 dir(0, 0, 0);
    dir[dim] += (pos > 0 ? 1 : -1);

    for (size_t di = 0; di < domain.num_data(); ++di) {
      recvers_.push_back(Recver(srcRank, srcGPU, dstRank, dstGPU, di, dir));
    }
  }

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->face_bytes(dim_, i);
      // printf("FaceRecver::allocate(): alloc %lu on %d\n", numBytes, gpu);
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

    // unpack all data into domain

    const Dim3 facePos = domain_->face_pos(dim_, pos_, true /*halo region*/);
    const Dim3 faceExtent = domain_->face_extent(dim_);

    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {
      const Dim3 rawSz = domain_->raw_size(dataIdx);
      char *dst = domain_->curr_data(dataIdx);

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(dataIdx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      unpack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          dst, rawSz, 0 /*pitch*/, facePos, faceExtent, bufs_[dataIdx],
          elemSize);
    }
  }

  void recv() override { fut_ = std::async(&FaceRecver::recv_impl, this); }

  // wait for send to be complete
  virtual void wait() override {
    if (fut_.valid()) {
      fut_.wait();
      CUDA_RUNTIME(cudaStreamSynchronize(domain_->stream()));
    } else {
      assert(0 && "wait called before recv");
    }
  }
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

    for (size_t dataIdx = 0; dataIdx < domain_->num_data(); ++dataIdx) {
      const Dim3 rawSz = domain_->raw_size(dataIdx);
      char *dst = domain_->curr_data(dataIdx);

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(dataIdx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      unpack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          dst, rawSz, 0 /*pitch*/, edgePos, edgeExtent, bufs_[dataIdx],
          elemSize);
    }
  }

  void recv() override { fut_ = std::async(&EdgeRecver::recv_impl, this); }

  // wait for send to be complete
  virtual void wait() override {
    if (fut_.valid()) {
      fut_.wait();
      CUDA_RUNTIME(cudaStreamSynchronize(domain_->stream()));
    } else {
      assert(0 && "wait called before recv");
    }
  }
};
