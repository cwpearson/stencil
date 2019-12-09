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
  virtual void send(const char *data) = 0;

  /*! block until send is complete
   */
  virtual void wait() = 0;
};

class NoOpSender : public Sender {
private:
  size_t srcRank;
  int64_t srcGPU;
  size_t dstRank;
  size_t dstGPU;

  size_t n_;

  void sender(const char *data) {
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

  void send(const char *data) override {
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
1) cudaMemcpy from src_gpu to src_rank
2) MPI_Send from src_rank to dst_rank
*/
class AnySender : public Sender {
private:
  size_t srcRank;
  size_t dstRank;
  size_t srcGPU;
};

class Recver {
public:
  /*! prepare to recv n bytes (allocate intermediate buffers)
   */
  virtual void resize(const size_t n) = 0;

  // recieve into data
  virtual void recv(char *data) = 0;

  /*! block until recv is complete
   */
  virtual void wait() = 0;
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
  void recv(char *data) override {
    fprintf(stderr, "would recv %luB from rank %lu into rank %lu gpu %lu\n", n_,
            srcRank, dstRank, dstGPU);
    return; // no-op
  }
  void wait() override {
    fprintf(stderr, "would wait\n");
    return; // no-op
  }
};

/*! A recvr that should work as long as MPI and CUDA are installed
1) MPI_Recv from src_rank to dst_rank
2) cudaMemcpy from dst_rank to dst_gpu
*/
class AnyRecver : public Recver {
private:
  size_t srcRank;
  size_t dstRank;
  size_t dstGPU;

  // src-to-dst intermediate data
  std::vector<char> buf;
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
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->face_bytes(dim_)[i];
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
      const char *src = domain_->data(idx);

      // pack into buffer
      dim3 dimGrid(10, 10, 10);
      dim3 dimBlock(32, 4, 4);
      size_t elemSize = domain_->elem_size()[idx];
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

class FaceRecverBase {
public:
  // prepare to send the appropriate number of bytes
  virtual void allocate() = 0;

  // recv the face data
  virtual void recv() = 0;

  // wait for send to be complete
  virtual void wait() = 0;
};