#pragma once

#include <vector>

#include "local_domain.cuh"
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

static __device__ void
dev_packer_grid_pack(void *__restrict__ dst, const void *__restrict__ src,
                     const Dim3 srcSize, const Dim3 srcPos,
                     const Dim3 srcExtent, const size_t elemSize) {

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t zo = tz; zo < srcExtent.z; zo += blockDim.z * gridDim.z) {
    size_t zi = zo + srcPos.z;
    for (size_t yo = ty; yo < srcExtent.y; yo += blockDim.y * gridDim.y) {
      size_t yi = yo + srcPos.y;
      for (size_t xo = tx; xo < srcExtent.x; xo += blockDim.x * gridDim.x) {

        size_t xi = xo + srcPos.x;
        size_t oi = zo * srcExtent.y * srcExtent.x + yo * srcExtent.x + xo;
        size_t ii = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //       yo, zo, oi);

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

__global__ static void
dev_packer_pack_domain(void *dst,             // buffer to pack into
                       void **srcs,           // raw pointer to each quanitity
                       size_t *elemSizes,     // element size for each quantity
                       const size_t nQuants,  // number of quantities
                       const Dim3 rawSz,      // domain size (elements)
                       const Dim3 *positions, // numHalos positions
                       const Dim3 *extents,   // numhalos extents
                       const size_t *offsets, // numHalos * nQuants offsets
                       const size_t numHalos) {
  size_t oi = 0; // offset index
  for (size_t mi = 0; mi < numHalos; ++mi) {
    Dim3 pos = positions[mi];
    Dim3 ext = extents[mi];
    for (size_t qi = 0; qi < nQuants; ++qi, ++oi) {
      void *src = srcs[qi];
      const size_t elemSz = elemSizes[qi];
      const size_t offset = offsets[oi];
      void *dstp = &((char *)dst)[offset];

      dev_packer_grid_pack(dstp, src, rawSz, pos, ext, elemSz);
    }
  }
}

class DevicePacker : public Packer {
private:
  LocalDomain *domain_;

  std::vector<Message> dirs_;
  size_t size_;
  Dim3 *devPositions_;
  Dim3 *devExtents_;
  size_t *devOffsets_;

  void *devBuf_;

public:
  DevicePacker()
      : devPositions_(0), devExtents_(0), devOffsets_(0), devBuf_(0) {}

  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) override {
    domain_ = domain;
    dirs_ = messages;
    std::sort(dirs_.begin(), dirs_.end());

    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // compute the buffer size, and track halo positions and extents
    std::vector<Dim3> positions;
    std::vector<Dim3> extents;
    std::vector<size_t> offsets;
    size_ = 0;
    for (const auto &msg : dirs_) {
      positions.push_back(domain_->halo_pos(msg.dir_, true /*halo*/));
      extents.push_back(domain_->halo_extent(msg.dir_));
      for (size_t i = 0; i < domain_->num_data(); ++i) {
        size_t numBytes = domain_->halo_bytes(msg.dir_, i);
        offsets.push_back(numBytes);
        size_ += numBytes;
      }
    }
    assert(positions.size() == extents.size());
    assert(offsets.size() == positions.size() * domain_->num_data());

    // copy halo info to the GPU
    size_t posBytes = positions.size() * sizeof(positions[0]);
    size_t extBytes = extents.size() * sizeof(extents[0]);
    size_t offBytes = offsets.size() * sizeof(offsets[0]);
    CUDA_RUNTIME(cudaMalloc(&devPositions_, posBytes));
    CUDA_RUNTIME(cudaMalloc(&devExtents_, extBytes));
    CUDA_RUNTIME(cudaMalloc(&devOffsets_, offBytes));
    CUDA_RUNTIME(cudaMemcpy(devPositions_, positions.data(), posBytes,
                            cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(devExtents_, extents.data(), extBytes,
                            cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(devOffsets_, offsets.data(), offBytes,
                            cudaMemcpyHostToDevice));

    // allocate the buffer for the packing
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));
  }

  virtual void pack(cudaStream_t stream) override {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    dim3 dimBlock(16, 8, 4);
    dim3 dimGrid(10, 10, 10);
    dev_packer_pack_domain<<<dimGrid, dimBlock, 0, stream>>>(
        devBuf_, domain_->dev_curr_datas(), domain_->dev_elem_sizes(),
        domain_->num_data(), domain_->raw_size(), devPositions_, devExtents_,
        devOffsets_, dirs_.size());
    CUDA_RUNTIME(cudaGetLastError());
  }

  virtual size_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};

static __device__ void
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

__global__ static void
dev_unpacker_unpack_domain(void **dsts,       // buffer to pack into
                           void *src,         // raw pointer to each quanitity
                           size_t *elemSizes, // element size for each quantity
                           const size_t nQuants,  // number of quantities
                           const Dim3 rawSz,      // domain size (elements)
                           const Dim3 *positions, // numHalos positions
                           const Dim3 *extents,   // numhalos extents
                           const size_t *offsets, // numHalos * nQuants offsets
                           const size_t numHalos) {
  size_t oi = 0; // offset index
  for (size_t mi = 0; mi < numHalos; ++mi) {
    Dim3 pos = positions[mi];
    Dim3 ext = extents[mi];
    for (size_t qi = 0; qi < nQuants; ++qi, ++oi) {
      void *dst = dsts[qi];
      const size_t elemSz = elemSizes[qi];
      const size_t offset = offsets[oi];
      void *srcp = &((char *)src)[offset];

      dev_unpacker_grid_unpack(dst, rawSz, pos, ext, srcp, elemSz);
    }
  }
}

class DeviceUnpacker : public Unpacker {
private:
  LocalDomain *domain_;

  std::vector<Message> dirs_;
  size_t size_;
  Dim3 *devPositions_;
  Dim3 *devExtents_;
  size_t *devOffsets_;

  void *devBuf_;

public:
  DeviceUnpacker()
      : devPositions_(0), devExtents_(0), devOffsets_(0), devBuf_(0) {}

  virtual void prepare(LocalDomain *domain,
                       const std::vector<Message> &messages) override {
    domain_ = domain;
    dirs_ = messages;
    std::sort(dirs_.begin(), dirs_.end());

    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    // compute the buffer size, and track halo positions and extents
    std::vector<Dim3> positions;
    std::vector<Dim3> extents;
    std::vector<size_t> offsets;
    size_ = 0;
    for (const auto &msg : dirs_) {
      positions.push_back(domain_->halo_pos(msg.dir_, false /*interior*/));
      extents.push_back(domain_->halo_extent(msg.dir_));
      for (size_t i = 0; i < domain_->num_data(); ++i) {
        size_t numBytes = domain_->halo_bytes(msg.dir_, i);
        offsets.push_back(numBytes);
        size_ += numBytes;
      }
    }
    assert(positions.size() == extents.size());
    assert(offsets.size() == positions.size() * domain_->num_data());

    // copy halo info to the GPU
    size_t posBytes = positions.size() * sizeof(positions[0]);
    size_t extBytes = extents.size() * sizeof(extents[0]);
    size_t offBytes = offsets.size() * sizeof(offsets[0]);
    CUDA_RUNTIME(cudaMalloc(&devPositions_, posBytes));
    CUDA_RUNTIME(cudaMalloc(&devExtents_, extBytes));
    CUDA_RUNTIME(cudaMalloc(&devOffsets_, offBytes));
    CUDA_RUNTIME(cudaMemcpy(devPositions_, positions.data(), posBytes,
                            cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(devExtents_, extents.data(), extBytes,
                            cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(devOffsets_, offsets.data(), offBytes,
                            cudaMemcpyHostToDevice));

    // allocate the buffer that will be unpacked
    CUDA_RUNTIME(cudaMalloc(&devBuf_, size_));
  }

  virtual void unpack(cudaStream_t stream) override {
    CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

    dim3 dimBlock(16, 8, 4);
    dim3 dimGrid(10, 10, 10);
    dev_unpacker_unpack_domain<<<dimGrid, dimBlock, 0, stream>>>(
        domain_->dev_curr_datas(), devBuf_, domain_->dev_elem_sizes(),
        domain_->num_data(), domain_->raw_size(), devPositions_, devExtents_,
        devOffsets_, dirs_.size());
    CUDA_RUNTIME(cudaGetLastError());
  }

  virtual size_t size() override { return size_; }

  virtual void *data() override { return devBuf_; }
};