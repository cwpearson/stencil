#include "stencil/packer.cuh"

#include "stencil/pack_kernel.cuh"
#include "stencil/rt.hpp"

#include <algorithm>

/*! pack all quantities in a single domain into a destination buffer
 */
__global__ void dev_packer_pack_domain(void *dst,               // buffer to pack into
                                       cudaPitchedPtr *srcs,    // pointer to each quantity
                                       const size_t *elemSizes, // element size for each quantity
                                       const size_t nQuants,    // number of quantities
                                       const Dim3 pos,          // halo position
                                       const Dim3 ext           // halo extent
) {
  size_t offset = 0;
  for (size_t qi = 0; qi < nQuants; ++qi) {
    const size_t elemSz = elemSizes[qi];
    offset = next_align_of(offset, elemSz);
    cudaPitchedPtr src = srcs[qi];
    void *dstp = &((char *)dst)[offset];
    grid_pack(dstp, src, pos, ext, elemSz);
    offset += elemSz * ext.flatten();
  }
}

__global__ void dev_unpacker_unpack_domain(cudaPitchedPtr *dsts,    // buffers to unpack into
                                           const void *src,         // raw pointer to each quanitity
                                           const size_t *elemSizes, // element size for each quantity
                                           const size_t nQuants,    // number of quantities
                                           const Dim3 pos,          // halo position
                                           const Dim3 ext           // halo extent
) {
  size_t offset = 0;
  for (unsigned int qi = 0; qi < nQuants; ++qi) {
    cudaPitchedPtr dst = dsts[qi];
    const size_t elemSz = elemSizes[qi];
    offset = next_align_of(offset, elemSz);
    void *srcp = &((char *)src)[offset];
    grid_unpack(dst, srcp, pos, ext, elemSz);
    offset += elemSz * ext.flatten();
  }
}

DevicePacker::DevicePacker(cudaStream_t stream)
    : domain_(nullptr), size_(-1), devBuf_(0), stream_(stream), graph_(NULL), instance_(NULL) {}

DevicePacker::~DevicePacker() {
#ifdef STENCIL_USE_CUDA_GRAPH
  //  if (domain_) {
  //    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  //    domain_ = nullptr;
  //  }
  if (graph_) {
    CUDA_RUNTIME(cudaGraphDestroy(graph_));
    graph_ = 0;
  }
  if (instance_) {
    CUDA_RUNTIME(cudaGraphExecDestroy(instance_));
    instance_ = 0;
  }
#endif
}

void DevicePacker::prepare(LocalDomain *domain, const std::vector<Message> &messages) {
  domain_ = domain;
  dirs_ = messages;
  std::sort(dirs_.begin(), dirs_.end(), Message::by_size);

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
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));

/* if we are using the graph API, record all the kernel launches here, otherwise
 * they will be done on-demand
 */
#ifdef STENCIL_USE_CUDA_GRAPH
  assert(stream_ != 0 && "can't capture the NULL stream, unless cudaStreamPerThread");
  CUDA_RUNTIME(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  launch_pack_kernels();
  CUDA_RUNTIME(cudaStreamEndCapture(stream_, &graph_));
  assert(graph_);
  CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
  assert(instance_);
#else
  // no other prep to do
#endif
}

void DevicePacker::launch_pack_kernels() {
  // record packing operations
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));

  int64_t offset = 0;
  for (const auto &msg : dirs_) {
    // pack from from +x interior
    const Dim3 pos = domain_->halo_pos(msg.dir_, false /*interior*/);
    // send +x means recv into -x halo. +x halo size could be different
    const Dim3 ext = domain_->halo_extent(msg.dir_ * -1);

    if (ext.flatten() == 0) {
      LOG_FATAL("asked to pack for direction " << msg.dir_ << " but computed message size is 0, ext=" << ext);
    }

    LOG_SPEW("dir=" << msg.dir_ << " ext=" << ext << " pos=" << pos << " @ " << offset);
    const dim3 dimBlock = Dim3::make_block_dim(ext, 512);
    const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / Dim3(dimBlock);
    assert(offset < size_);

    LOG_SPEW("dev_packer_pack_domain grid=" << dimGrid << " block=" << dimBlock);
#if 0
    dev_packer_pack_domain<<<dimGrid, dimBlock, 0, stream_>>>(&devBuf_[offset], domain_->dev_curr_datas(),
                                                              domain_->dev_elem_sizes(), domain_->num_data(), pos, ext);
#endif
    rt::launch(dev_packer_pack_domain, dimGrid, dimBlock, 0, stream_, &devBuf_[offset], domain_->dev_curr_datas(),
               domain_->dev_elem_sizes(), domain_->num_data(), pos, ext);
#ifndef STENCIL_USE_CUDA_GRAPH
    // 900: not allowed while stream is capturing
    CUDA_RUNTIME(rt::time(cudaGetLastError));
#endif
    for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
      offset = next_align_of(offset, domain_->elem_size(qi));
      // send +x means recv into -x halo. +x halo size could be different
      offset += domain_->halo_bytes(msg.dir_ * -1, qi);
    }
  }
  // with cuda graph, this is called during setup so dont time it
  CUDA_RUNTIME(cudaGetLastError());
}

void DevicePacker::pack() {
  assert(size_);
#ifdef STENCIL_USE_CUDA_GRAPH
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));
  CUDA_RUNTIME(rt::time(cudaGraphLaunch, instance_, stream_));
#else
  launch_pack_kernels();
#endif
}

DeviceUnpacker::DeviceUnpacker(cudaStream_t stream)
    : domain_(nullptr), size_(-1), devBuf_(0), stream_(stream), graph_(NULL), instance_(NULL) {}

DeviceUnpacker::~DeviceUnpacker() {

#ifdef STENCIL_USE_CUDA_GRAPH
  // TODO: these need to be guarded from ctor without prepare()?
  if (graph_) {
    CUDA_RUNTIME(cudaGraphDestroy(graph_));
    graph_ = 0;
  }
  if (instance_) {
    CUDA_RUNTIME(cudaGraphExecDestroy(instance_));
    instance_ = 0;
  }
#endif
}

void DeviceUnpacker::prepare(LocalDomain *domain, const std::vector<Message> &messages) {
  domain_ = domain;
  dirs_ = messages;

  // sort so we unpack in the same order as the sender packed
  std::sort(dirs_.begin(), dirs_.end(), Message::by_size);

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
#ifdef STENCIL_USE_CUDA_GRAPH
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
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

void DeviceUnpacker::launch_unpack_kernels() {
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));

  int64_t offset = 0;
  for (const auto &msg : dirs_) {

    const Dim3 dir = msg.dir_ * -1; // unpack into opposite side as sent
    const Dim3 ext = domain_->halo_extent(dir);
    const Dim3 pos = domain_->halo_pos(dir, true /*exterior*/);

    LOG_SPEW("DeviceUnpacker::unpack(): dir=" << msg.dir_ << " ext=" << ext << " pos=" << pos << " @" << offset);

    const dim3 dimBlock = Dim3::make_block_dim(ext, 512);
    const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
#if 0
    dev_unpacker_unpack_domain<<<dimGrid, dimBlock, 0, stream_>>>(
        domain_->dev_curr_datas(), &devBuf_[offset], domain_->dev_elem_sizes(), domain_->num_data(), pos, ext);
#endif
    rt::launch(dev_unpacker_unpack_domain, dimGrid, dimBlock, 0, stream_, domain_->dev_curr_datas(), &devBuf_[offset],
               domain_->dev_elem_sizes(), domain_->num_data(), pos, ext);
#ifndef STENCIL_USE_CUDA_GRAPH
    // 900: operation not permitted while stream is capturing
    CUDA_RUNTIME(rt::time(cudaGetLastError));
#endif
    for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
      offset = next_align_of(offset, domain_->elem_size(qi));
      offset += domain_->halo_bytes(dir, qi);
    }
  }
  CUDA_RUNTIME(rt::time(cudaGetLastError));
}

void DeviceUnpacker::unpack() {
  assert(size_);
#ifdef STENCIL_USE_CUDA_GRAPH
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  CUDA_RUNTIME(rt::time(cudaGraphLaunch, instance_, stream_));
#else
  launch_unpack_kernels();
#endif
}
