#pragma once

#include <vector>

#include "align.cuh"
#include "local_domain.cuh"
#include "pack_kernel.cuh"
#include "tx_common.hpp"

class Packer {
public:
  // prepare pack a domain for messages
  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) = 0;

  // pack
  virtual void pack(cudaStream_t stream) = 0;

  // number of bytes
  virtual size_t size() = 0;
  virtual void *data() = 0;

  virtual ~Packer() {}
};

class Unpacker {
public:
  // prepare pack a domain for messages
  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) = 0;

  virtual void unpack(cudaStream_t stream) = 0;

  virtual size_t size() = 0;
  virtual void *data() = 0;

  virtual ~Unpacker() {}
};

/*! pack all quantities in a single domain into dst
 */
inline __global__ void
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
      for (size_t qi = 0; qi < domain_->num_data(); ++qi) {
        size_ = next_align_of(size_, domain_->elem_size(qi));
        size_ += domain_->halo_bytes(msg.dir_, qi);
      }
    }

    // allocate the buffer for the packing
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));
  }

  virtual void pack(cudaStream_t stream) override {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    size_t offset = 0;
    for (const auto &msg : dirs_) {
      const Dim3 ext = domain_->halo_extent(msg.dir_);
      const Dim3 pos = domain_->halo_pos(msg.dir_, false /*interior*/);
      const dim3 dimBlock = make_block_dim(ext, 512);
      const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      assert(offset < size_);
      dev_packer_pack_domain<<<dimGrid, dimBlock, 0, stream>>>(
          &devBuf_[offset], domain_->dev_curr_datas(),
          domain_->dev_elem_sizes(), domain_->num_data(), domain_->raw_size(),
          pos, ext);
      CUDA_RUNTIME(cudaGetLastError());
      for (size_t qi = 0; qi < domain_->num_data(); ++qi) {
        offset = next_align_of(offset, domain_->elem_size(qi));
        offset += domain_->halo_bytes(msg.dir_, qi);
      }
    }
  }

  virtual size_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};

inline __device__ void
dev_unpacker_grid_unpack(void *__restrict__ dst, const Dim3 dstSize,
                         const Dim3 dstPos, const Dim3 dstExtent,
                         const void *__restrict__ src, const size_t elemSize) {

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t zi = tz; zi < dstExtent.z; zi += blockDim.z * gridDim.z) {
    for (size_t yi = ty; yi < dstExtent.y; yi += blockDim.y * gridDim.y) {
      for (size_t xi = tx; xi < dstExtent.x; xi += blockDim.x * gridDim.x) {
        size_t zo = zi + dstPos.z;
        size_t yo = yi + dstPos.y;
        size_t xo = xi + dstPos.x;
        size_t oi = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        size_t ii = zi * dstExtent.y * dstExtent.x + yi * dstExtent.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //        yo, zo, oi);
        if (4 == elemSize) {
          uint32_t *pDst = reinterpret_cast<uint32_t *>(dst);
          const uint32_t *pSrc = reinterpret_cast<const uint32_t *>(src);
          pDst[oi] = pSrc[ii];
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

inline __global__ void
dev_unpacker_unpack_domain(void **dsts,       // buffer to pack into
                           void *src,         // raw pointer to each quanitity
                           size_t *elemSizes, // element size for each quantity
                           const size_t nQuants, // number of quantities
                           const Dim3 rawSz,     // domain size (elements)
                           const Dim3 pos,       // halo position
                           const Dim3 ext        // halo extent
) {
  size_t offset = 0;
  for (size_t qi = 0; qi < nQuants; ++qi) {
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
    std::sort(dirs_.begin(), dirs_.end());

    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // compute the required buffer size for all messages
    size_ = 0;
    for (const auto &msg : dirs_) {
      for (size_t qi = 0; qi < domain_->num_data(); ++qi) {
        size_ = next_align_of(size_, domain_->elem_size(qi));
        size_ += domain_->halo_bytes(msg.dir_, qi);
      }
    }

    // allocate the buffer that will be unpacked
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));
  }

  virtual void unpack(cudaStream_t stream) override {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    size_t offset = 0;
    for (const auto &msg : dirs_) {
      const Dim3 ext = domain_->halo_extent(msg.dir_);
      const Dim3 pos = domain_->halo_pos(msg.dir_, true /*exterior*/);

      const dim3 dimBlock = make_block_dim(ext, 512);
      const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      dev_unpacker_unpack_domain<<<dimGrid, dimBlock, 0, stream>>>(
          domain_->dev_curr_datas(), &devBuf_[offset],
          domain_->dev_elem_sizes(), domain_->num_data(), domain_->raw_size(),
          pos, ext);
      CUDA_RUNTIME(cudaGetLastError());
      for (size_t qi = 0; qi < domain_->num_data(); ++qi) {
        offset = next_align_of(offset, domain_->elem_size(qi));
        offset += domain_->halo_bytes(msg.dir_, qi);
      }
    }
  }

  virtual size_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};
