#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

#include <mpi.h>

#include "cuda_runtime.hpp"

#include "stencil/dim3.cuh"
#include "stencil/tx.cuh"

class DistributedDomain;
class LocalDomain;

template <typename T> class DataHandle {
  friend class DistributedDomain;
  friend class LocalDomain;
  size_t id_;

public:
  DataHandle(size_t i) : id_(i) {}
};

// https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
std::vector<size_t> prime_factors(size_t n) {
  std::vector<size_t> result;

  while (n % 2 == 0) {
    result.push_back(2);
    n = n / 2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    // While i divides n, print i and divide n
    while (n % i == 0) {
      result.push_back(i);
      n = n / i;
    }
  }
  if (n > 2)
    result.push_back(n);

  return result;
}

/*!
  min element / max element
*/
double cubeness(double x, double y, double z) {
  double smallest = min(x, min(y, z));
  double largest = max(x, max(y, z));
  return smallest / largest;
}

/*! \brief ceil(n/d)
 */
size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }

class LocalDomain {
  friend class DistributedDomain;

private:
  // my local data size
  Dim3 sz_;

  //!< radius of stencils that will be applied
  size_t radius_;

  //!< backing info for the actual data I have
  std::vector<char *> dataPtrs_;
  std::vector<size_t> dataElemSize_;

  //!< backing info for the halo exchange buffers for each data
  std::vector<char *> pzHaloRecv_;
  std::vector<char *> mzHaloRecv_;
  std::vector<char *> pyHaloRecv_;
  std::vector<char *> myHaloRecv_;
  std::vector<char *> pxHaloRecv_;
  std::vector<char *> mxHaloRecv_;

  std::vector<char *> pzHaloSend_;
  std::vector<char *> mzHaloSend_;
  std::vector<char *> pyHaloSend_;
  std::vector<char *> myHaloSend_;
  std::vector<char *> pxHaloSend_;
  std::vector<char *> mxHaloSend_;

public:
  LocalDomain(Dim3 sz) : sz_(sz) {}

  // the sizes of the faces in bytes for each data along the requested dimension
  // x = 0, y = 1, etc
  std::vector<size_t> face_bytes(const size_t dim) const {
    std::vector<size_t> results;

    for (auto elemSz : dataElemSize_) {
      size_t bytes = elemSz;
      if (0 == dim) {
        // y * z * radius_
        bytes *= (sz_.y - 2 * radius_) * (sz_.z - 2 * radius_) * radius_;
      } else if (1 == dim) {
        // x * z * radius_
        bytes *= (sz_.x - 2 * radius_) * (sz_.z - 2 * radius_) * radius_;
      } else if (2 == dim) {
        // x * y * radius_
        bytes *= (sz_.x - 2 * radius_) * (sz_.y - 2 * radius_) * radius_;

      } else {
        assert(0);
      }
      results.push_back(bytes);
    }
    return results;
  }

  // the sizes of the edges in bytes for each data along the requested dimension
  // x = 0, y = 1, etc
  std::vector<size_t> edge_bytes(const size_t dim0, const size_t dim1) const {
    std::vector<size_t> results;

    assert(dim0 != dim1 && "no edge between matching dims");

    for (auto elemSz : dataElemSize_) {
      size_t bytes = elemSz;
      if (0 != dim0 && 0 != dim1) {
        bytes *= sz_[0];
      } else if (1 != dim0 && 1 != dim1) {
        bytes *= sz_[1];
      } else if (2 != dim0 && 2 != dim1) {
        bytes *= sz_[2];
      } else {
        assert(0);
      }
      results.push_back(bytes);
    }
    return results;
  }

  // the size of the halo corner in bytes for each data
  std::vector<size_t> corner_bytes() const {
    std::vector<size_t> results;
    for (auto elemSz : dataElemSize_) {
      size_t bytes = elemSz * radius_ * radius_ * radius_;
      results.push_back(bytes);
    }
    return results;
  }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
  }

  template <typename T> T *get_data(const DataHandle<T> handle) {
    assert(dataElemSize_.size() > handle.id_);
    assert(dataPtrs_.size() > handle.id_);
    void *ptr = dataPtrs_[handle.id_];
    assert(sizeof(T) == dataElemSize_[handle.id_]);
    return static_cast<T *>(ptr);
  }

  void realize() {

    // allocate each data region
    for (size_t i = 0; i < dataElemSize_.size(); ++i) {
      size_t elemSz = dataElemSize_[i];

      size_t elemBytes = ((sz_.x + 2 * radius_) * (sz_.y + 2 * radius_) *
                          (sz_.z + 2 * radius_)) *
                         elemSz;
      std::cerr << "Allocate " << elemBytes << "\n";
      char *p = new char[elemBytes];
      assert(uintptr_t(p) % elemSz == 0 && "allocation should be aligned");
      dataPtrs_.push_back(p);
    }

    // allocate halos
    for (auto elemSz : dataElemSize_) {
      std::vector<size_t> xBytes = face_bytes(0);
      for (auto bytes : xBytes) {
#warning unimplemented
      }
      std::vector<size_t> yBytes = face_bytes(1);
      for (auto bytes : yBytes) {
#warning unimplemented
      }
      std::vector<size_t> zBytes = face_bytes(2);
      for (auto bytes : zBytes) {
        char *data;
        CUDA_RUNTIME(cudaMalloc(&data, bytes));
        pzHaloSend_.push_back(data);
        CUDA_RUNTIME(cudaMalloc(&data, bytes));
        mzHaloSend_.push_back(data);
        CUDA_RUNTIME(cudaMalloc(&data, bytes));
        pzHaloRecv_.push_back(data);
        CUDA_RUNTIME(cudaMalloc(&data, bytes));
        mzHaloRecv_.push_back(data);
      }
    }
  }
};

class DistributedDomain {
private:
  Dim3 size_;

  int rank_;
  int worldSize_;

  // the GPUs this MPI rank will use
  std::vector<int> gpus_;

  // the stencil radius
  size_t radius_;

  // the dimension of the domain in MPI ranks and GPUs
  Dim3 rankDim_;
  Dim3 gpuDim_;

  // typically one per GPU
  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;
  // the index of the domain in the distributed domain
  std::vector<Dim3> indices_;

  // Senders / receivers for each domain
  // faces
  std::vector<Sender *> pzSenders_; // senders for +z
  std::vector<Recver *> pzRecvers_;
  std::vector<Sender *> mzSenders_; // senders for -z
  std::vector<Recver *> mzRecvers_;

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

public:
  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), rankDim_(1, 1, 1), gpuDim_(1, 1, 1) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));

    // create a communicator for this node
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    int shmrank, shmsize;
    MPI_Comm_rank(shmcomm, &shmrank);
    MPI_Comm_size(shmcomm, &shmsize);

    // if fewer ranks, round-robin GPUs to ranks
    if (shmsize <= deviceCount) {
      for (int gpu = 0; gpu < deviceCount; ++gpu) {
        if (gpu % shmsize == shmrank) {
          gpus_.push_back(gpu);
        }
      }
    } else { // if more ranks, share gpus among ranks
      gpus_.push_back(shmrank % deviceCount);
    }

    for (const auto gpu : gpus_) {
      printf("rank %d/%d local=%d using gpu %d\n", rank_, worldSize_, shmrank,
             gpu);
    }
  }

  std::vector<LocalDomain> &domains() { return domains_; }

  void set_radius(size_t r) { radius_ = r; }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
  }

  size_t get_rank(const Dim3 idx) {
    assert(idx.x >= 0);
    assert(idx.y >= 0);
    assert(idx.z >= 0);
    assert(idx.x < rankDim_.x * gpuDim_.x);
    return (idx.x / gpuDim_.x) + (idx.y / gpuDim_.y) * rankDim_.x +
           (idx.z / gpuDim_.z) * rankDim_.y * rankDim_.x;
  }

  size_t get_gpu(const Dim3 idx) {
    assert(idx.x >= 0);
    assert(idx.y >= 0);
    assert(idx.z >= 0);
    return (idx.x % gpuDim_.x) + (idx.y % gpuDim_.y) * gpuDim_.x +
           (idx.z % gpuDim_.z) * gpuDim_.y * gpuDim_.x;
  }

  Dim3 get_idx(size_t rank, size_t gpu) {
    assert(0);
#warning unimplemented
  }

  void realize() {

    // recursively split region among MPI ranks to make it ~cubical
    Dim3 splitSize = size_;
    auto factors = prime_factors(worldSize_);
    std::sort(factors.begin(), factors.end(),
              [](size_t a, size_t b) { return b < a; });
    for (size_t amt : factors) {

      if (rank_ == 0) {
        printf("split by %lu\n", amt);
      }

      double curCubeness = cubeness(size_.x, size_.y, size_.z);
      double xSplitCubeness =
          cubeness(div_ceil(splitSize.x, amt), splitSize.y, splitSize.z);
      double ySplitCubeness =
          cubeness(splitSize.x, div_ceil(splitSize.y, amt), splitSize.z);
      double zSplitCubeness =
          cubeness(splitSize.x, splitSize.y, div_ceil(splitSize.z, amt));

      if (rank_ == 0) {
        printf("%lu %lu %lu %f\n", size_.x, size_.y, size_.z, curCubeness);
      }

      if (xSplitCubeness > max(ySplitCubeness, zSplitCubeness)) { // split in x
        if (rank_ == 0) {
          printf("x split: %f\n", xSplitCubeness);
        }
        splitSize.x = div_ceil(splitSize.x, amt);
        rankDim_.x *= amt;
      } else if (ySplitCubeness >
                 max(xSplitCubeness, ySplitCubeness)) { // split in y
        if (rank_ == 0) {
          printf("y split: %f\n", ySplitCubeness);
        }
        splitSize.y = div_ceil(splitSize.y, amt);
        rankDim_.y *= amt;
      } else { // split in z
        if (rank_ == 0) {
          printf("z split: %f\n", zSplitCubeness);
        }
        splitSize.z = div_ceil(splitSize.z, amt);
        rankDim_.z *= amt;
      }
    }

    // split biggest dimension across GPUs
    if (splitSize.x > max(splitSize.y, splitSize.z)) {
      gpuDim_.x = gpus_.size();
      splitSize.x /= gpus_.size();
    } else if (splitSize.y > max(splitSize.x, splitSize.x)) {
      gpuDim_.y = gpus_.size();
      splitSize.y /= gpus_.size();
    } else {
      gpuDim_.z = gpus_.size();
      splitSize.z /= gpus_.size();
    }

    if (rank_ == 0) {
      printf("%lux%lux%lu of %lux%lux%lux (gpus %lux%lux%lu)\n", splitSize.x,
             splitSize.y, splitSize.z, rankDim_.x, rankDim_.y, rankDim_.z,
             gpuDim_.x, gpuDim_.y, gpuDim_.z);
    }

    // create local domains
    for (const auto gpu : gpus_) {

      Dim3 rankIdx;
      Dim3 gpuIdx;

      auto rank = rank_;
      rankIdx.x = rank % rankDim_.x;
      rank /= rankDim_.x;
      rankIdx.y = rank % rankDim_.y;
      rank /= rankDim_.y;
      rankIdx.z = rank;

      auto i = gpu;
      gpuIdx.x = i % gpuDim_.x;
      i /= gpuDim_.x;
      gpuIdx.y = i % gpuDim_.y;
      i /= gpuDim_.y;
      gpuIdx.z = i;

      LocalDomain ld(splitSize);
      ld.dataElemSize_ = dataElemSize_;
      ld.radius_ = radius_;

      domains_.push_back(ld);
      Dim3 idx = rankIdx * gpuDim_ + gpuIdx;
      printf("rank,gpu=%d,%d => idx %ld %ld %ld\n", rank_, gpu, idx.x, idx.y,
             idx.z);
      indices_.push_back(rankIdx * gpuDim_ + gpuIdx);
    }

    // realize local domains
    for (auto &d : domains_) {
      d.realize();
    }

    for (size_t di = 0; di < domains_.size(); ++di) {
      assert(domains_.size() == indices_.size());

      auto &d = domains_[di];
      std::vector<int> deltas{-1, 1};

      // create senders
      for (auto dx : deltas) {
      }
      for (auto dz : deltas) {
      }
      for (auto dz : deltas) {
        auto srcIdx = indices_[di];
        Dim3 dstIdx = (srcIdx + Dim3(0, 0, dz)).wrap(rankDim_ * gpuDim_);
        printf("dz=%d: %ld %ld %ld -> %ld %ld %ld\n", dz, srcIdx.x, srcIdx.y,
               srcIdx.z, dstIdx.x, dstIdx.y, dstIdx.z);
        int64_t srcRank = get_rank(srcIdx);
        int64_t srcGPU = get_gpu(srcIdx);
        int64_t dstRank = get_rank(dstIdx);
        pzSenders_.push_back(new NoOpSender(srcRank, dstRank, srcGPU));
      }

      // create recvers
      for (auto dz : deltas) {
        auto dstIdx = indices_[di];
        Dim3 srcIdx = srcIdx - Dim3(0, 0, dz);
        int64_t srcRank = get_rank(srcIdx);
        int64_t dstGPU = get_gpu(dstIdx);
        int64_t dstRank = get_rank(dstIdx);
        pzRecvers_.push_back(new NoOpRecver(srcRank, dstRank, dstGPU));
      }
      for (auto dy : deltas) {
      }
      for (auto dz : deltas) {
      }
    }
  }

  /*!
  start a halo exchange and return.
  Call sync() to block until exchange is done.
  */
  void exchange_async() {

    for (size_t di = 0; di < domains_.size(); ++di) {
      auto &d = domains_[di];
      std::vector<size_t> numBytes = d.face_bytes(2); // z
      for (size_t i = 0; i < d.dataElemSize_.size(); ++i) {
        pzRecvers_[di]->resize(numBytes[i]);
        pzSenders_[di]->resize(numBytes[i]);
        pzRecvers_[di]->recv(d.pzHaloRecv_[i]);
        pzSenders_[di]->send(d.pzHaloSend_[i]);
      }
#warning unimplemented
    }
  }

  /*!
  wait for async exchange
  */
  void sync() {

    // wait for all senders and recvers to be done
    for (auto &tx : pzSenders_) {
      tx->wait();
    }
    for (auto &tx : pzRecvers_) {
      tx->wait();
    }

    // wait for everyone else's exchanges to be done
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {
    exchange_async();
    sync();
  }
};