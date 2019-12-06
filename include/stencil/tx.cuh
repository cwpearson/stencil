#pragma once

#include <future>

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
  size_t dstRank;
  size_t srcGPU;

  size_t n_;

  void sender(const char *data) {
    fprintf(stderr, "would send %luB from rank %lu gpu %lu to rank %lu\n", n_, srcRank, srcGPU, dstRank);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2s);
}

  std::future<void> waiter;


public:

  NoOpSender(size_t srcRank, size_t dstRank, size_t srcGPU) : srcRank(srcRank), dstRank(dstRank), srcGPU(srcGPU) {}

  void resize(const size_t n) override {
    n_ = n;
    return; // no-op
  }



  void send(const char *data) override {
    waiter = std::async(&NoOpSender::sender, this, data);
    // fprintf(stderr, "would send %luB from rank %lu gpu %lu to rank %lu\n", n_, srcRank, srcGPU, dstRank);
    return; // no-op
  }
  void wait() override {
    
    if (waiter.valid()) {
        fprintf(stderr, "waiting\n");
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
  size_t dstRank;
  size_t dstGPU;

  size_t n_;

public:
  NoOpRecver(size_t srcRank, size_t dstRank, size_t dstGPU) : srcRank(srcRank), dstRank(dstRank), dstGPU(dstGPU) {
  }

  void resize(const size_t n) override {
    n_ = n;
    return; // no-op
  }
  void recv(char *data) override {
    fprintf(stderr, "would recv %luB from rank %lu into rank %lu gpu %lu\n", n_, srcRank, dstRank, dstGPU);
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
