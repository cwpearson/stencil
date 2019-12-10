#pragma once

#include <functional>
#include <future>

#include <mpi.h>

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/local_domain.cuh"

/*! An asynchronous sender
 */
class Sender {

public:
  /*! prepare to send n bytes (allocate intermediate buffers)
   */
  virtual void resize(const size_t n) = 0;

  // send n bytes
  virtual void send(const void *data) = 0;

  /*! block until send is complete
   */
  virtual void wait() = 0;

  virtual ~Sender() {}
};


class Recver {
public:
  /*! prepare to recv n bytes (allocate intermediate buffers)
   */
  virtual void resize(const size_t n) = 0;

  // recieve into data
  virtual void recv(void *data) = 0;

  /*! block until recv is complete
   */
  virtual void wait() = 0;

  virtual ~Recver() {}
};


class NoOpSender : public Sender {
private:
  size_t srcRank;
  int64_t srcGPU;
  size_t dstRank;
  size_t dstGPU;

  size_t n_;

  void sender(const void *data) {
    fprintf(stderr, "would send %luB from rank %lu gpu %lu to rank %lu\n", n_,
            srcRank, srcGPU, dstRank);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2s);
  }

  std::future<void> waiter;

public:
  NoOpSender(size_t srcRank, int64_t srcGPU, size_t dstRank, size_t dstGPU)
      : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU) {}

  NoOpSender(const NoOpSender &other)
      : srcRank(other.srcRank), srcGPU(other.srcGPU), dstRank(other.dstRank),
        dstGPU(other.dstGPU) {}

  void resize(const size_t n) override {
    n_ = n;
    return; // no-op
  }

  void send(const void *data) override {
    waiter = std::async(&NoOpSender::sender, this, data);
    // fprintf(stderr, "would send %luB from rank %lu gpu %lu to rank %lu\n",
    // n_, srcRank, srcGPU, dstRank);
    return; // no-op
  }
  void wait() override {

    if (waiter.valid()) {
      fprintf(stderr, "NoOpSender::wait() waiting\n");
      waiter.wait();
    }
    return; // no-op
  }
};

/*! A data sender that should work as long as MPI and CUDA are installed
1) cudaMemcpy from srcGPU to srcRank
2) MPI_Send from srcRank to dstRank, tagged with dstGPU
*/
class AnySender : public Sender {
private:
  int64_t srcRank;
  int64_t srcGPU;
  int64_t dstRank;
  int64_t dstGPU;

  cudaStream_t stream_;
  std::vector<char> hostBuf_;
  MPI_Request req_;
  std::future<void> waiter;

  void sender(const void *data) {
    printf("AnySender::sender(): cudaMemcpy %ld -> %ld\n", srcGPU, srcRank);
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), data, hostBuf_.size(), cudaMemcpyDefault, stream_));
    printf("AnySender::sender(): cuda sync\n");
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    printf("AnySender::sender(): ISend %luB -> %ld (tag=%ld)\n", hostBuf_.size(), dstRank, dstGPU);
    MPI_Isend(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, dstRank, dstGPU, MPI_COMM_WORLD, &req_);
  }

public:

  AnySender(int64_t srcRank, int64_t srcGPU, int64_t dstRank, int64_t dstGPU) : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU) {
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }
  ~AnySender() {
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
  }

  void resize(const size_t n) override {hostBuf_.resize(n);}

  void send(const void *data) override {
    waiter = std::async(&AnySender::sender, this, data);
  }

  void wait() override {
    if (waiter.valid()) {
      waiter.wait();
    }
    MPI_Status stat;
    printf("AnySender::wait(): waiting on Isend\n");
    MPI_Wait(&req_, &stat);
  }
};



class NoOpRecver : public Recver {
private:
  size_t srcRank;
  int64_t srcGPU;
  size_t dstRank;
  size_t dstGPU;

  size_t n_;

public:
  NoOpRecver(size_t srcRank, int64_t srcGPU, size_t dstRank, size_t dstGPU)
      : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU) {}

  void resize(const size_t n) override {
    n_ = n;
    return; // no-op
  }
  void recv(void *data) override {
    fprintf(stderr, "would recv %luB from rank %lu into rank %lu gpu %lu\n", n_,
            srcRank, dstRank, dstGPU);
    return; // no-op
  }
  void wait() override {
    fprintf(stderr, "would wait\n");
    return; // no-op
  }
};

/*! A data recver that should work as long as MPI and CUDA are installed
1) cudaMemcpy from srcGPU to srcRank
2) MPI_Send from srcRank to dstRank, tagged with dstGPU
*/
class AnyRecver : public Recver {
private:
  int64_t srcRank;
  int64_t srcGPU;
  int64_t dstRank;
  int64_t dstGPU;

  cudaStream_t stream_;
  std::vector<char> hostBuf_;
  std::future<void> waiter;

  void recver(void *data) {
    MPI_Request req;
    MPI_Status stat;
    printf("AnyRecver::recver(): Irecv %luB from %ld (tag=%ld)\n", hostBuf_.size(), srcRank, dstGPU);
    MPI_Irecv(hostBuf_.data(), hostBuf_.size(), MPI_BYTE, srcRank, dstGPU, MPI_COMM_WORLD, &req);
    printf("AnyRecver::recver(): wait on Irecv\n");
    MPI_Wait(&req, &stat);
    printf("AnyRecver::recver(): cudaMemcpyAsync\n");
    CUDA_RUNTIME(cudaMemcpyAsync(data, hostBuf_.data(), hostBuf_.size(), cudaMemcpyDefault, stream_));
  }

public:

  AnyRecver(int64_t srcRank, int64_t srcGPU, int64_t dstRank, int64_t dstGPU) : srcRank(srcRank), srcGPU(srcGPU), dstRank(dstRank), dstGPU(dstGPU) {
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }
  ~AnyRecver() {
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
  }

  void resize(const size_t n) override {hostBuf_.resize(n);}

  void recv(void *data) override {
    waiter = std::async(&AnyRecver::recver, this, data);
  }

  void wait() override {
    if (waiter.valid()) {
      waiter.wait();
    }
    printf("AnyRecver::recver(): wait for cuda sync (stream=%lu)\n", uintptr_t(stream_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }
};

class FaceSenderBase {
public:
  // prepare to send the appropriate number of bytes
  virtual void allocate() = 0;

  // send the face data
  virtual void send() = 0;

  // wait for send to be complete
  virtual void wait() = 0;
};

class FaceRecverBase {
public:
  // prepare to send the appropriate number of bytes
  virtual void allocate() = 0;

  // recv the face data
  virtual void recv() = 0;

  // wait for send to be complete
  virtual void wait() = 0;
};



template <typename Sender> class FaceSender : public FaceSenderBase {
private:
  const LocalDomain *domain_; // the domain we are sending from
  size_t dim_;                // the face dimension we are sending
  bool pos_;                  // positive or negative face

  // one sender per domain data
  std::vector<Sender> senders_;
  // one device buffer per domain data
  std::vector<char *> bufs_;

public:
  FaceSender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
             size_t dstRank, size_t dstGPU, size_t dim, bool pos)
      : senders_(domain.num_data(), Sender(srcRank, srcGPU, dstRank, dstGPU)),
        domain_(&domain), dim_(dim), pos_(pos) {}

  virtual void allocate() override {
    // allocate space for each transfer
    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    // preallocate each sender
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->face_bytes(dim_, i);
      printf("FaceSender::allocate(): alloc %lu on %d\n", numBytes, gpu);
      char *buf = nullptr;
      CUDA_RUNTIME(cudaMalloc(&buf, numBytes));
      bufs_.push_back(buf);
      senders_[i].resize(numBytes);
    }
  }

  virtual void send() override {
    allocate();

    const Dim3 facePos = domain_->face_pos(pos_, dim_);
    const Dim3 faceExtent = domain_->face_extent(pos_, dim_);

    for (size_t idx = 0; idx < domain_->num_data(); ++idx) {
      const Dim3 rawSz = domain_->raw_size(idx);
      const char *src = domain_->curr_data(idx);

      // pack into buffer
      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size(idx);
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      pack<<<dimGrid, dimBlock, 0, domain_->stream()>>>(
          (int *)bufs_[idx], (const int *)src, rawSz, 0 /*pitch*/, facePos,
          faceExtent, elemSize);

      CUDA_RUNTIME(cudaDeviceSynchronize());

      // copy to dst rank
      senders_[idx].send(bufs_[idx]);
    }
  }

  // wait for send to be complete
  virtual void wait() override {
    for (auto &s : senders_) {
      s.wait();
    }
  }
};

