#pragma once

#include <thread>
#include <vector>

#include "align.cuh"
#include "local_domain.cuh"
#include "pack_kernel.cuh"
#include "tx_common.hpp"

//#define STENCIL_PACK_LOUD
//#define STENCIL_UNPACK_LOUD

inline void rand_sleep() {
  int ms = rand() % 10;
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

class Packer {
public:
  // prepare pack a domain for messages
  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) = 0;

  // pack
  virtual void pack(cudaStream_t stream) = 0;

  // number of bytes
  virtual int64_t size() = 0;
  virtual void *data() = 0;

  virtual ~Packer() {}
};

class Unpacker {
public:
  // prepare pack a domain for messages
  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) = 0;

  virtual void unpack(cudaStream_t stream) = 0;

  virtual int64_t size() = 0;
  virtual void *data() = 0;

  virtual ~Unpacker() {}
};

/*! pack all quantities in a single domain into dst
 */
static __global__ void
dev_packer_pack_domain(void *dst,            // buffer to pack into
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

public:
  DevicePacker() : domain_(nullptr), size_(-1), devBuf_(0) {}

  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) override {
    domain_ = domain;
    dirs_ = messages;
    std::sort(dirs_.begin(), dirs_.end());

    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // compute the required buffer size for all messages
    size_ = 0;
    for (const auto &msg : dirs_) {
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        size_ = next_align_of(size_, domain_->elem_size(qi));
        size_ += domain_->halo_bytes(msg.dir_, qi);
      }
    }

    // allocate the buffer for the packing
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));
  }

  virtual void pack(cudaStream_t stream) override {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    int64_t offset = 0;
    for (const auto &msg : dirs_) {
      const Dim3 ext = domain_->halo_extent(msg.dir_);
      const Dim3 pos = domain_->halo_pos(msg.dir_, false /*interior*/);
      const dim3 dimBlock = make_block_dim(ext, 512);
      const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / Dim3(dimBlock);
      assert(offset < size_);
#ifdef STENCIL_PACK_LOUD
      rand_sleep();
      std::cerr << "DevicePacker::pack(): dir=" << msg.dir_ << " ext=" << ext
                << " pos=" << pos << " @" << offset << "\n";
      rand_sleep();
#endif
      dev_packer_pack_domain<<<dimGrid, dimBlock, 0, stream>>>(
          &devBuf_[offset], domain_->dev_curr_datas(),
          domain_->dev_elem_sizes(), domain_->num_data(), domain_->raw_size(),
          pos, ext);
      CUDA_RUNTIME(cudaGetLastError());
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        offset = next_align_of(offset, domain_->elem_size(qi));
        offset += domain_->halo_bytes(msg.dir_, qi);
      }
    }
  }

  virtual int64_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};

inline __device__ void
dev_unpacker_grid_unpack(void *__restrict__ dst, const Dim3 dstSize,
                         const Dim3 dstPos, const Dim3 dstExtent,
                         const void *__restrict__ src, const size_t elemSize) {

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
      for (unsigned int xi = tx; xi < dstExtent.x;
           xi += blockDim.x * gridDim.x) {
        unsigned int xo = xi + dstPos.x;
        unsigned int oi = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        unsigned int ii =
            zi * dstExtent.y * dstExtent.x + yi * dstExtent.x + xi;
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

static __global__ void
dev_unpacker_unpack_domain(void **dsts,       // buffer to pack into
                           void *src,         // raw pointer to each quanitity
                           size_t *elemSizes, // element size for each quantity
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

public:
  DeviceUnpacker() : domain_(nullptr), size_(-1), devBuf_(0) {}

  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) override {
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
        size_ += domain_->halo_bytes(msg.dir_, qi);
      }
    }

    // allocate the buffer that will be unpacked
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));
  }

  virtual void unpack(cudaStream_t stream) override {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    int64_t offset = 0;
    for (const auto &msg : dirs_) {

      const Dim3 dir =
          msg.dir_ * -1; // unpack into the opposite halo as was sent
      const Dim3 ext = domain_->halo_extent(dir);
      const Dim3 pos = domain_->halo_pos(dir, true /*exterior*/);

#ifdef STENCIL_UNPACK_LOUD
      rand_sleep();
      std::cerr << "DeviceUnpacker::unpack(): dir=" << msg.dir_
                << " ext=" << ext << " pos=" << pos << " @" << offset << "\n";
      rand_sleep();
#endif

      const dim3 dimBlock = make_block_dim(ext, 512);
      const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      dev_unpacker_unpack_domain<<<dimGrid, dimBlock, 0, stream>>>(
          domain_->dev_curr_datas(), &devBuf_[offset],
          domain_->dev_elem_sizes(), domain_->num_data(), domain_->raw_size(),
          pos, ext);
      CUDA_RUNTIME(cudaGetLastError());
      for (int64_t qi = 0; qi < domain_->num_data(); ++qi) {
        offset = next_align_of(offset, domain_->elem_size(qi));
        offset += domain_->halo_bytes(dir, qi);
      }
    }
  }

  virtual int64_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};
