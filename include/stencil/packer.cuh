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

/*! pack all quantities in a single domain into dst
 */
static __global__ void dev_packer_pack_domain(void *dst,            // buffer to pack into
                                              void **srcs,          // raw pointer to each quanitity
                                              size_t *elemSizes,    // element size for each quantity
                                              const size_t nQuants, // number of quantities
                                              const Dim3 rawSz,     // domain size (elements)
                                              const Dim3 pos,       // halo position
                                              const Dim3 ext        // halo extent
) {
  size_t offset = 0;
  for (size_t qi = 0; qi < nQuants; ++qi) {
    const size_t elemSz = elemSizes[qi];
    offset = next_align_of(offset, elemSz);
    void *src = srcs[qi];
    void *dstp = &((char *)dst)[offset];
    grid_pack(dstp, src, rawSz, pos, ext, elemSz);
    offset += elemSz * ext.flatten();
  }
}

class DevicePacker : public Packer {
private:
  LocalDomain *domain_;

  std::vector<Message> dirs_;
  int64_t size_;

  char *devBuf_;

  cudaStream_t stream_; // an unowned stream
  cudaGraph_t graph_;
  cudaGraphExec_t instance_;

  void launch_pack_kernels() {
    // record packing operations
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    int64_t offset = 0;
    for (const auto &msg : dirs_) {
      // pack from from +x interior
      const Dim3 pos = domain_->halo_pos(msg.dir_, false /*interior*/);
      // send +x means recv into -x halo. +x halo size could be different
      const Dim3 ext = domain_->halo_extent(msg.dir_ * -1);

      if (ext.flatten() == 0) {
        LOG_FATAL("asked to pack for direction " << msg.dir_ << " but computed message size is 0, ext=" << ext);
      }

      LOG_SPEW("DevicePacker::pack(): dir=" << msg.dir_ << " ext=" << ext << " pos=" << pos << " @ " << offset);
      const dim3 dimBlock = Dim3::make_block_dim(ext, 512);
      const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / Dim3(dimBlock);
      assert(offset < size_);

      LOG_SPEW("DevicePacker::pack(): grid= " << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z
                                              << " block=" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z);
      dev_packer_pack_domain<<<dimGrid, dimBlock, 0, stream_>>>(&devBuf_[offset], domain_->dev_curr_datas(),
                                                                domain_->dev_elem_sizes(), domain_->num_data(),
                                                                domain_->raw_size(), pos, ext);
#if STENCIL_USE_CUDA_GRAPH == 0
      // 900: not allowed while stream is capturing
      CUDA_RUNTIME(cudaGetLastError());
#endif
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        offset = next_align_of(offset, domain_->elem_size(qi));
        // send +x means recv into -x halo. +x halo size could be different
        offset += domain_->halo_bytes(msg.dir_ * -1, qi);
      }
    }
  }

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

  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages) {

    domain_ = domain;
    dirs_ = messages;
    std::sort(dirs_.begin(), dirs_.end());

    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // compute the required buffer size for all messages
    size_ = 0;
    for (const auto &msg : dirs_) {
      LOG_SPEW("DevicePacker::prepare(): msg.dir_=" << msg.dir_);
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        size_ = next_align_of(size_, domain_->elem_size(qi));

        // if message sends in +x, we are sending to -x halo, so the size of the
        // data will be the size of the -x halo region (the +x halo region may
        // be different due to an uncentered kernel)
        size_ += domain_->halo_bytes(msg.dir_ * -1, qi);
      }

      if (0 == size_) {
        LOG_FATAL("zero-size packer was prepared");
      }
    }

    // allocate the buffer for the packing
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));

/* if we are using the graph API, record all the kernel launches here, otherwise
 * they will be done on-demand
 */
#if STENCIL_USE_CUDA_GRAPH == 1
    assert(stream_ != 0 && "can't capture the NULL stream, unless cudaStreamPerThread");
    // TODO: safer if thread-local?
    CUDA_RUNTIME(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    launch_pack_kernels();
    CUDA_RUNTIME(cudaStreamEndCapture(stream_, &graph_));
    CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
#else
    // no other prep to do
#endif
  }

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

inline __device__ void dev_unpacker_grid_unpack(void *__restrict__ dst, const Dim3 dstSize, const Dim3 dstPos,
                                                const Dim3 dstExtent, const void *__restrict__ src,
                                                const size_t elemSize) {

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  assert(dstExtent.z >= 0);
  assert(dstExtent.y >= 0);
  assert(dstExtent.x >= 0);

  for (unsigned int zi = tz; zi < dstExtent.z; zi += blockDim.z * gridDim.z) {
    unsigned int zo = zi + dstPos.z;
    for (unsigned int yi = ty; yi < dstExtent.y; yi += blockDim.y * gridDim.y) {
      unsigned int yo = yi + dstPos.y;
      for (unsigned int xi = tx; xi < dstExtent.x; xi += blockDim.x * gridDim.x) {
        unsigned int xo = xi + dstPos.x;
        unsigned int oi = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        unsigned int ii = zi * dstExtent.y * dstExtent.x + yi * dstExtent.x + xi;
        if (4 == elemSize) {
          uint32_t *pDst = reinterpret_cast<uint32_t *>(dst);
          const uint32_t *pSrc = reinterpret_cast<const uint32_t *>(src);
          uint32_t v = pSrc[ii];
          pDst[oi] = v;
        } else if (8 == elemSize) {
          uint64_t *pDst = reinterpret_cast<uint64_t *>(dst);
          const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
          pDst[oi] = pSrc[ii];
        } else {
          char *pDst = reinterpret_cast<char *>(dst);
          const char *pSrc = reinterpret_cast<const char *>(src);
          memcpy(&pDst[oi * elemSize], &pSrc[ii * elemSize], elemSize);
        }
      }
    }
  }
}

static __global__ void dev_unpacker_unpack_domain(void **dsts,          // buffer to pack into
                                                  void *src,            // raw pointer to each quanitity
                                                  size_t *elemSizes,    // element size for each quantity
                                                  const size_t nQuants, // number of quantities
                                                  const Dim3 rawSz,     // domain size (elements)
                                                  const Dim3 pos,       // halo position
                                                  const Dim3 ext        // halo extent
) {
  size_t offset = 0;
  for (unsigned int qi = 0; qi < nQuants; ++qi) {
    void *dst = dsts[qi];
    const size_t elemSz = elemSizes[qi];
    offset = next_align_of(offset, elemSz);
    void *srcp = &((char *)src)[offset];
    dev_unpacker_grid_unpack(dst, rawSz, pos, ext, srcp, elemSz);
    offset += elemSz * ext.flatten();
  }
}

class DeviceUnpacker : public Unpacker {
private:
  LocalDomain *domain_;

  std::vector<Message> dirs_;
  int64_t size_;

  char *devBuf_;

  cudaStream_t stream_;
  cudaGraph_t graph_;
  cudaGraphExec_t instance_;

  void launch_unpack_kernels() {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    int64_t offset = 0;
    for (const auto &msg : dirs_) {

      const Dim3 dir = msg.dir_ * -1; // unpack into opposite side as sent
      const Dim3 ext = domain_->halo_extent(dir);
      const Dim3 pos = domain_->halo_pos(dir, true /*exterior*/);

      LOG_SPEW("DeviceUnpacker::unpack(): dir=" << msg.dir_ << " ext=" << ext << " pos=" << pos << " @" << offset);

      const dim3 dimBlock = Dim3::make_block_dim(ext, 512);
      const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      dev_unpacker_unpack_domain<<<dimGrid, dimBlock, 0, stream_>>>(domain_->dev_curr_datas(), &devBuf_[offset],
                                                                    domain_->dev_elem_sizes(), domain_->num_data(),
                                                                    domain_->raw_size(), pos, ext);
#if STENCIL_USE_CUDA_GRAPH == 0
      // 900: operation not permitted while stream is capturing
      CUDA_RUNTIME(cudaGetLastError());
#endif
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        offset = next_align_of(offset, domain_->elem_size(qi));
        offset += domain_->halo_bytes(dir, qi);
      }
    }
  }

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

  virtual void prepare(LocalDomain *domain, const std::vector<Message> &messages) override {
    domain_ = domain;
    dirs_ = messages;

    // sort so we unpack in the same order as the sender packed
    std::sort(dirs_.begin(), dirs_.end());

    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // compute the required buffer size for all messages
    size_ = 0;
    for (const auto &msg : dirs_) {
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        size_ = next_align_of(size_, domain_->elem_size(qi));

        // if message sends in +x, we are sending to -x halo, so the size of the
        // data will be the size of the -x halo region (the +x halo region may
        // be different due to an uncentered kernel)
        size_ += domain_->halo_bytes(msg.dir_ * -1, qi);
      }

      if (0 == size_) {
        LOG_FATAL("0-size packer was prepared");
      }
    }

    // allocate the buffer that will be unpacked
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));

/* if we are using the graph API, record all the kernel launches here, otherwise
 * they will be done on-demand
 */
#if STENCIL_USE_CUDA_GRAPH == 1
    assert(stream_ != 0 && "can't capture the NULL stream, unless cudaStreamPerThread");
    // TODO: safer if thread-local?
    CUDA_RUNTIME(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    launch_unpack_kernels();
    CUDA_RUNTIME(cudaStreamEndCapture(stream_, &graph_));
    CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
#else
    // no other prep to do
#endif
  }

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

#undef STENCIL_USE_CUDA_GRAPH
