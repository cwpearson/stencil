#pragma once

#include <thread>
#include <vector>

#include "align.cuh"
#include "local_domain.cuh"
#include "stencil/logging.hpp"
#include "tx_common.hpp"

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
  DevicePacker(cudaStream_t stream);

  ~DevicePacker(); 

  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages);

  virtual void pack() override;

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
  DeviceUnpacker(cudaStream_t stream);
  ~DeviceUnpacker();

  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages) override;

  virtual void unpack() override;

  virtual int64_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};
