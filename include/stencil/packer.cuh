#pragma once

#include <thread>
#include <vector>

#include "align.cuh"
#include "local_domain.cuh"
#include "pack_kernel.cuh"
#include "stencil/logging.hpp"
#include "tx_common.hpp"

/* Use the CUDA Graph API to accelerate repeated
   pack/unpack kernel launches
*/
#define STENCIL_USE_CUDA_GRAPH 1

inline void rand_sleep() {
  int ms = rand() % 10;
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

class Packer {
public:
  // prepare pack a domain for messages
  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages) = 0;

  // pack
  virtual void pack() = 0;

  // number of bytes
  virtual int64_t size() = 0;
  virtual void *data() = 0;

  virtual ~Packer() {}
};

class Unpacker {
public:
  // prepare pack a domain for messages
  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages) = 0;

  virtual void unpack() = 0;

  virtual int64_t size() = 0;
  virtual void *data() = 0;

  virtual ~Unpacker() {}
};


class DevicePacker : public Packer {
private:
  LocalDomain *domain_;

  std::vector<Message> dirs_;
  int64_t size_;

  char *devBuf_;

  cudaStream_t stream_; // an unowned stream
  cudaGraph_t graph_;
  cudaGraphExec_t instance_;

  void launch_pack_kernels();

public:
  DevicePacker(cudaStream_t stream)
      : domain_(nullptr), size_(-1), devBuf_(0), stream_(stream), graph_(NULL), instance_(NULL) {}
  ~DevicePacker() {
#if STENCIL_USE_CUDA_GRAPH == 1
    // TODO: these need to be guarded from ctor without prepare()?
    if (graph_) {
      CUDA_RUNTIME(cudaGraphDestroy(graph_));
    }
    if (instance_) {
      CUDA_RUNTIME(cudaGraphExecDestroy(instance_));
    }
#endif
  }

  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages);

  virtual void pack() {
    assert(size_);
#if STENCIL_USE_CUDA_GRAPH == 1
    CUDA_RUNTIME(cudaGraphLaunch(instance_, stream_));
#else
    launch_pack_kernels();
#endif
  }

  virtual int64_t size() { return size_; }

  virtual void *data() { return devBuf_; }
};

class DeviceUnpacker : public Unpacker {
private:
  LocalDomain *domain_;

  std::vector<Message> dirs_;
  int64_t size_;

  char *devBuf_;

  cudaStream_t stream_;
  cudaGraph_t graph_;
  cudaGraphExec_t instance_;

  void launch_unpack_kernels();

public:
  DeviceUnpacker(cudaStream_t stream)
      : domain_(nullptr), size_(-1), devBuf_(0), stream_(stream), graph_(NULL), instance_(NULL) {}
  ~DeviceUnpacker() {
#if STENCIL_USE_CUDA_GRAPH == 1
    // TODO: these need to be guarded from ctor without prepare()?
    if (graph_) {
      CUDA_RUNTIME(cudaGraphDestroy(graph_));
    }
    if (instance_) {
      CUDA_RUNTIME(cudaGraphExecDestroy(instance_));
    }
#endif
  }

  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages) override;

  virtual void unpack() override {
    assert(size_);
#if STENCIL_USE_CUDA_GRAPH == 1
    CUDA_RUNTIME(cudaGraphLaunch(instance_, stream_));
#else
    launch_unpack_kernels();
#endif
  }

  virtual int64_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};
