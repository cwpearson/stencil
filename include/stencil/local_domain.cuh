#pragma once

#include <iostream>
//#include <mpi.h>

#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/pack_kernel.cuh"

class DistributedDomain;

template <typename T> class DataHandle {
  friend class DistributedDomain;
  friend class LocalDomain;
  size_t id_;
  std::string name_;

public:
  DataHandle(size_t i, const std::string &name = "") : id_(i), name_(name) {}
};

enum class DataType {
  None,
  Float,
  Double,
};

class LocalDomain {
  friend class DistributedDomain;

private:
  // my local data size (not including halo)
  Dim3 sz_;

  //!< radius of stencils that will be applied
  size_t radius_;

  //!< backing info for the actual data I have
  // host versions
  std::vector<void *> currDataPtrs_;
  std::vector<void *> nextDataPtrs_;
  std::vector<int64_t> dataElemSize_;
  std::vector<std::string> dataName_;
  // device versions
  void **devCurrDataPtrs_;
  size_t *devDataElemSize_;

  int dev_; // CUDA device

public:
  LocalDomain(Dim3 sz, int dev)
      : sz_(sz), dev_(dev), devCurrDataPtrs_(nullptr),
        devDataElemSize_(nullptr) {}

  ~LocalDomain() {
    CUDA_RUNTIME(cudaGetLastError());

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // std::cerr << "dtor rank=" << rank << " ~LocalDomain(): device=" << dev_
    // << "\n";
    CUDA_RUNTIME(cudaSetDevice(dev_));
    for (auto p : currDataPtrs_) {
      // std::cerr << "rank=" << rank << " ~LocalDomain(): cudaFree " <<
      // uintptr_t(p) << "\n";
      if (p)
        CUDA_RUNTIME(cudaFree(p));
    }
    if (devCurrDataPtrs_)
      CUDA_RUNTIME(cudaFree(devCurrDataPtrs_));

    for (auto p : nextDataPtrs_) {
      if (p)
        CUDA_RUNTIME(cudaFree(p));
    }
    if (devDataElemSize_)
      CUDA_RUNTIME(cudaFree(devDataElemSize_));
    CUDA_RUNTIME(cudaGetLastError());
  }

  /*
   */
  int64_t num_data() const {
    assert(currDataPtrs_.size() == nextDataPtrs_.size());
    assert(dataElemSize_.size() == currDataPtrs_.size());
    return int64_t(currDataPtrs_.size());
  }

  /*! Add an untyped data field with an element size of n.

  \returns The index of the added data
  */
  int64_t add_data(size_t n, const std::string &name = "") {
    dataName_.push_back(name);
    dataElemSize_.push_back(n);
    currDataPtrs_.push_back(nullptr);
    nextDataPtrs_.push_back(nullptr);
    return int64_t(dataElemSize_.size()) - 1;
  }

  template <typename T> DataHandle<T> add_data(const std::string &name = "") {
    return DataHandle<T>(add_data(sizeof(T)), name);
  }



  /*! \brief set the radius. Should only be called by DistributedDomain
   */
  void set_radius(size_t r) { radius_ = r; }

  size_t radius() const noexcept { return radius_; }

  /*! \brief retrieve a pointer to current domain values (to read in stencil)
   */
  template <typename T> T *get_curr(const DataHandle<T> handle) {
    assert(dataElemSize_.size() > handle.id_);
    assert(currDataPtrs_.size() > handle.id_);
    void *ptr = currDataPtrs_[handle.id_];
    assert(sizeof(T) == dataElemSize_[handle.id_]);
    return static_cast<T *>(ptr);
  }

  /*! \brief retrieve a pointer to next domain values (to set in stencil)
   */
  template <typename T> T *get_next(const DataHandle<T> handle) {
    assert(dataElemSize_.size() > handle.id_);
    assert(nextDataPtrs_.size() > handle.id_);
    void *ptr = nextDataPtrs_[handle.id_];
    assert(sizeof(T) == dataElemSize_[handle.id_]);
    return static_cast<T *>(ptr);
  }

  size_t elem_size(const size_t idx) const {
    assert(idx < dataElemSize_.size());
    return dataElemSize_[idx];
  }

  size_t *dev_elem_sizes() const { return devDataElemSize_; }

  void *curr_data(size_t idx) const {
    assert(idx < currDataPtrs_.size());
    return currDataPtrs_[idx];
  }

  void **dev_curr_datas() const { return devCurrDataPtrs_; }

  void *next_data(size_t idx) const {
    assert(idx < nextDataPtrs_.size());
    return nextDataPtrs_[idx];
  }

  // return the position of the halo relative to get_data() in direction `dir`
  Dim3 halo_pos(const Dim3 &dir, const bool halo) const noexcept {
    Dim3 ret;
    assert(dir.all_gt(-2));
    assert(dir.all_lt(2));

    if (1 == dir.x) {
      ret.x = sz_.x + (halo ? radius_ : 0);
    } else if (-1 == dir.x) {
      ret.x = halo ? 0 : radius_;
    } else if (0 == dir.x) {
      ret.x = radius_;
    } else {
      __builtin_unreachable();
    }

    if (1 == dir.y) {
      ret.y = sz_.y + (halo ? radius_ : 0);
    } else if (-1 == dir.y) {
      ret.y = halo ? 0 : radius_;
    } else if (0 == dir.y) {
      ret.y = radius_;
    } else {
      __builtin_unreachable();
    }

    if (1 == dir.z) {
      ret.z = sz_.z + (halo ? radius_ : 0);
    } else if (-1 == dir.z) {
      ret.z = halo ? 0 : radius_;
    } else if (0 == dir.z) {
      ret.z = radius_;
    } else {
      __builtin_unreachable();
    }

    return ret;
  }

  // used by some placement code to compute a hypothetical communication cost
  static Dim3 halo_extent(const Dim3 &dir, const Dim3 &sz,
                          const size_t radius) {
    assert(dir.x >= -1 && dir.x <= 1);
    assert(dir.y >= -1 && dir.y <= 1);
    assert(dir.z >= -1 && dir.z <= 1);
    Dim3 ret;

    ret.x = (dir.x != 0) ? radius : sz.x;
    ret.y = (dir.y != 0) ? radius : sz.y;
    ret.z = (dir.z != 0) ? radius : sz.z;

    return ret;
  }

  // return the extent of the halo in direction `dir`
  Dim3 halo_extent(const Dim3 &dir) const noexcept {
    return halo_extent(dir, sz_, radius_);
  }

  // return the number of bytes of the halo in direction `dir`
  int64_t halo_bytes(const Dim3 &dir, const int64_t idx) const noexcept {
    return dataElemSize_[idx] * halo_extent(dir).flatten();
  }

  // return the 3d size of the compute domain, in terms of elements
  Dim3 size() const noexcept { return sz_; }

  // return the 3d size of the actual allocation, in terms of elements
  Dim3 raw_size() const noexcept {
    return Dim3(sz_.x + 2 * radius_, sz_.y + 2 * radius_, sz_.z + 2 * radius_);
  }

  // the GPU this domain is on
  int gpu() const { return dev_; }

  std::vector<unsigned char> region_to_host(const Dim3 &pos, const Dim3 &ext,
                                            const size_t qi // quantity index
                                            ) const {

    const size_t bytes = elem_size(qi) * ext.flatten();

    // pack quantity
    CUDA_RUNTIME(cudaSetDevice(gpu()));
    void *devBuf = nullptr;
    CUDA_RUNTIME(cudaMalloc(&devBuf, bytes));
    const dim3 dimBlock = make_block_dim(ext, 512);
    const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
    pack_kernel<<<dimGrid, dimBlock>>>(devBuf, curr_data(qi), raw_size(), pos,
                                       ext, elem_size(qi));

    // copy quantity to host
    std::vector<unsigned char> hostBuf(bytes);
    CUDA_RUNTIME(
        cudaMemcpy(hostBuf.data(), devBuf, hostBuf.size(), cudaMemcpyDefault));

    // free device buffer
    CUDA_RUNTIME(cudaFree(devBuf));

    return hostBuf;
  }

  /*! Copy the compute region to the host
  */
  std::vector<unsigned char> interior_to_host(const size_t qi // quantity index
                                              ) const {

    Dim3 pos = halo_pos(Dim3(0, 0, 0), true);
    Dim3 ext = halo_extent(Dim3(0, 0, 0));
    return region_to_host(pos, ext, qi);
  }

  /*! Copy an entire quantity, including halo region, to host
  */
  std::vector<unsigned char> quantity_to_host(const size_t qi // quantity index
                                              ) const {
    Dim3 allocSz = sz_;
    allocSz.x += 2 * radius_;
    allocSz.y += 2 * radius_;
    allocSz.z += 2 * radius_;
    return region_to_host(Dim3(0, 0, 0), allocSz, qi);
  }

  void realize() {
    CUDA_RUNTIME(cudaGetLastError());
    assert(currDataPtrs_.size() == nextDataPtrs_.size());
    assert(dataElemSize_.size() == nextDataPtrs_.size());

    // allocate each data region
    CUDA_RUNTIME(cudaSetDevice(dev_));
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // std::cerr << "r" << rank << " dev=" << dev_ << "\n";
    for (int64_t i = 0; i < num_data(); ++i) {
      int64_t elemSz = dataElemSize_[i];

      int64_t elemBytes = ((sz_.x + 2 * radius_) * (sz_.y + 2 * radius_) *
                          (sz_.z + 2 * radius_)) *
                         elemSz;
      char *c = nullptr;
      char *n = nullptr;
      CUDA_RUNTIME(cudaMalloc(&c, elemBytes));
      CUDA_RUNTIME(cudaMalloc(&n, elemBytes));
      assert(uintptr_t(c) % elemSz == 0 && "allocation should be aligned");
      assert(uintptr_t(n) % elemSz == 0 && "allocation should be aligned");
      currDataPtrs_[i] = c;
      nextDataPtrs_[i] = n;
    }

    CUDA_RUNTIME(cudaMalloc(&devCurrDataPtrs_,
                            currDataPtrs_.size() * sizeof(currDataPtrs_[0])));
    CUDA_RUNTIME(cudaMalloc(&devDataElemSize_,
                            dataElemSize_.size() * sizeof(dataElemSize_[0])));
    CUDA_RUNTIME(cudaMemcpy(devCurrDataPtrs_, currDataPtrs_.data(),
                            currDataPtrs_.size() * sizeof(currDataPtrs_[0]),
                            cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(devDataElemSize_, dataElemSize_.data(),
                            dataElemSize_.size() * sizeof(dataElemSize_[0]),
                            cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaGetLastError());
  }
};
