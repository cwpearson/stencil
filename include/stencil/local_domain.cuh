#pragma once

#include <iostream>

#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.cuh"

class DistributedDomain;

template <typename T> class DataHandle {
  friend class DistributedDomain;
  friend class LocalDomain;
  size_t id_;

public:
  DataHandle(size_t i) : id_(i) {}
};

class LocalDomain {
  friend class DistributedDomain;

private:
  // my local data size
  Dim3 sz_;

  //!< radius of stencils that will be applied
  size_t radius_;

  //!< backing info for the actual data I have
  std::vector<char *> currDataPtrs_;
  std::vector<char *> nextDataPtrs_;
  std::vector<size_t> dataElemSize_;

  int dev_;             // CUDA device
  cudaStream_t stream_; // CUDA stream

public:
  LocalDomain(Dim3 sz, int dev) : sz_(sz), dev_(dev), stream_(0) {}

  ~LocalDomain() {

    CUDA_RUNTIME(cudaSetDevice(dev_));
    for (auto p : currDataPtrs_) {
      CUDA_RUNTIME(cudaFree(p));
    }

    for (auto p : nextDataPtrs_) {
      CUDA_RUNTIME(cudaFree(p));
    }

    if (stream_ != 0) {
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
  }

  // the sizes of the faces in bytes for each data along the requested dimension
  // (x = 0, y = 1, etc) for data entry idx
  size_t face_bytes(const size_t dim, const size_t idx) const {
    assert(idx < dataElemSize_.size());
    size_t bytes = dataElemSize_[idx];
    if (0 == dim) { // x face = y * z * radius_
      bytes *= sz_.y * sz_.z * radius_;
    } else if (1 == dim) { // y face = x * z * radius
      bytes *= sz_.x * sz_.z * radius_;
    } else if (2 == dim) { // z face = x * y * radius_
      bytes *= sz_.x * sz_.y * radius_;
    } else {
      assert(0);
    }
    return bytes;
  }

  // the sizes of the edges in bytes for each data along the requested dimension
  // x = 0, y = 1, etc
  size_t edge_bytes(const size_t dim0, const size_t dim1,
                    const size_t idx) const {

    assert(dim0 != dim1 && "no edge between matching dims");
    assert(idx < dataElemSize_.size());
    size_t bytes = dataElemSize_[idx];
    if (0 != dim0 && 0 != dim1) {
      bytes *= sz_[0];
    } else if (1 != dim0 && 1 != dim1) {
      bytes *= sz_[1];
    } else if (2 != dim0 && 2 != dim1) {
      bytes *= sz_[2];
    } else {
      assert(0);
    }
    return bytes;
  }

  // the size of the halo corner in bytes for each data
  size_t corner_bytes(const size_t idx) const {
    assert(idx < dataElemSize_.size());
    return dataElemSize_[idx] * radius_ * radius_ * radius_;
  }

  /*
   */
  size_t num_data() const {
    assert(currDataPtrs_.size() == nextDataPtrs_.size());
    assert(dataElemSize_.size() == currDataPtrs_.size());
    return currDataPtrs_.size();
  }

  template <typename T> DataHandle<T> add_data() {
    return DataHandle<T>(add_data(sizeof(T)));
  }

  /*! Add an untyped data field with an element size of n.

  \returns The index of the added data
  */
  size_t add_data(size_t n) {
    dataElemSize_.push_back(n);
    currDataPtrs_.push_back(nullptr);
    nextDataPtrs_.push_back(nullptr);
    return dataElemSize_.size() - 1;
  }

  /*! \brief set the radius. Should only be called by DistributedDomain
   */
  void set_radius(size_t r) { radius_ = r; }

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

  size_t elem_size(const size_t idx) const { return dataElemSize_[idx]; }

  char *curr_data(size_t idx) const {
    assert(idx < currDataPtrs_.size());
    return currDataPtrs_[idx];
  }

  char *next_data(size_t idx) const {
    assert(idx < nextDataPtrs_.size());
    return nextDataPtrs_[idx];
  }

  size_t pitch() const {
#warning pitch is unimplemented
    assert(0);
    return 0;
  }

  // return the position of the face relative to get_data()
  // positive or negative
  // x=0, y=1, z=2
  Dim3 face_pos(const size_t dim, const bool pos, const bool halo) const {
    switch (pos) {
    case false: // negative-facing
      switch (dim) {
      case 0:
        return Dim3(halo ? 0 : radius_, radius_, radius_);
      case 1:
        return Dim3(radius_, halo ? 0 : radius_, radius_);
      case 2:
        return Dim3(radius_, radius_, halo ? 0 : radius_);
      }
    case true: // positive-facing
      switch (dim) {
      case 0:
        return Dim3(sz_.x + (halo ? radius_ : 0), radius_, radius_); // +x
      case 1:
        return Dim3(radius_, sz_.y + (halo ? radius_ : 0), radius_); // +y
      case 2:
        return Dim3(radius_, radius_, sz_.z + (halo ? radius_ : 0)); // +z
      }
    }

    assert(0 && "unreachable");
    return Dim3(-1, -1, -1);
  }

  // return the extent of the face in every dimension
  Dim3 face_extent(const size_t dim) const {
    switch (dim) {
    case 0:
      return Dim3(radius_, sz_.y, sz_.z);
    case 1:
      return Dim3(sz_.x, radius_, sz_.z);
    case 2:
      return Dim3(sz_.x, sz_.y, radius_);
    default:
      assert(0 && "unreachable");
      return Dim3(-1, -1, -1);
    }
  }

  /*! \brief return the position of the edge, to be used in conjunction with
    edge_extent takes the two dimensions the edge abuts, and whether or not
    those dimensions are positive
    \param halo \\![in] whether the position is in the halo, or compute region
  */
  Dim3 edge_pos(const size_t dim0, const size_t dim1, const bool dim0Pos,
                const bool dim1Pos, const bool halo) const {
    assert(dim0 != dim1);
    assert(dim0 < 3);
    assert(dim1 < 3);

    // make dim2 not dim1 or dim0
    size_t dim2 = 0;
    while (dim2 == dim1 || dim2 == dim0) {
      ++dim2;
    }

    size_t dim0Sz;
    size_t dim1Sz;

    if (dim0Pos) {
      if (0 == dim0) {
        dim0Sz = sz_.x + (halo ? radius_ : 0);
      } else if (1 == dim0) {
        dim0Sz = sz_.y + (halo ? radius_ : 0);
      } else if (2 == dim0) {
        dim0Sz = sz_.z + (halo ? radius_ : 0);
      } else {
        assert(0);
      }
    } else {
      dim0Sz = halo ? 0 : radius_;
    }
    if (dim1Pos) {
      if (dim1Pos && 0 == dim1) {
        dim1Sz = sz_.x + (halo ? radius_ : 0);
      } else if (1 == dim1) {
        dim1Sz = sz_.y + (halo ? radius_ : 0);
      } else if (2 == dim1) {
        dim1Sz = sz_.z + (halo ? radius_ : 0);
      } else {
        assert(0);
      }
    } else {
      dim1Sz = halo ? 0 : radius_;
    }

    Dim3 result;
    result[dim0] = dim0Sz;
    result[dim1] = dim1Sz;
    result[dim2] = radius_;

    return result;
  }

  /*! \brief return the extent of the edge, to be used in conjunction with
  takes the two dimensions the edge abuts
*/
  Dim3 edge_extent(const size_t dim0, const size_t dim1) const {
    assert(dim0 != dim1);
    assert(dim0 < 3);
    assert(dim1 < 3);

    if (0 != dim0 and 0 != dim1) {
      return Dim3(sz_.x, radius_, radius_);
    } else if (1 != dim0 and 1 != dim1) {
      return Dim3(radius_, sz_.y, radius_);
    } else if (2 != dim0 and 2 != dim1) {
      return Dim3(radius_, radius_, sz_.z);
    } else {
      assert(0);
      return Dim3(-1, -1, -1);
    }
  }

  // return the 3d size of the actual allocation for data idx, in terms of
  // elements
  Dim3 raw_size(size_t idx) const {
    return Dim3(sz_.x + 2 * radius_, sz_.y + 2 * radius_, sz_.z + 2 * radius_);
  }

  // the GPU this domain is on
  int gpu() const { return dev_; }

  // a stream associated with this domain
  cudaStream_t stream() const { return stream_; }

  void realize() {

    assert(currDataPtrs_.size() == nextDataPtrs_.size());
    assert(dataElemSize_.size() == nextDataPtrs_.size());

    // allocate each data region
    for (size_t i = 0; i < num_data(); ++i) {
      size_t elemSz = dataElemSize_[i];

      size_t elemBytes = ((sz_.x + 2 * radius_) * (sz_.y + 2 * radius_) *
                          (sz_.z + 2 * radius_)) *
                         elemSz;
      std::cerr << "Allocate " << elemBytes << "B on gpu " << dev_ << "\n";
      char *c = nullptr;
      char *n = nullptr;
      CUDA_RUNTIME(cudaSetDevice(dev_));
      CUDA_RUNTIME(cudaMalloc(&c, elemBytes));
      CUDA_RUNTIME(cudaMalloc(&n, elemBytes));
      assert(uintptr_t(c) % elemSz == 0 && "allocation should be aligned");
      assert(uintptr_t(n) % elemSz == 0 && "allocation should be aligned");
      currDataPtrs_[i] = c;
      nextDataPtrs_[i] = n;
    }
  }

  void realize_unified() {

    assert(currDataPtrs_.size() == nextDataPtrs_.size());
    assert(dataElemSize_.size() == nextDataPtrs_.size());

    // allocate each data region
    for (size_t i = 0; i < num_data(); ++i) {
      size_t elemSz = dataElemSize_[i];

      size_t elemBytes = ((sz_.x + 2 * radius_) * (sz_.y + 2 * radius_) *
                          (sz_.z + 2 * radius_)) *
                         elemSz;
      std::cerr << "Allocate " << elemBytes << "B on gpu " << dev_ << "\n";
      char *c = nullptr;
      char *n = nullptr;
      CUDA_RUNTIME(cudaSetDevice(dev_));
      CUDA_RUNTIME(cudaMalloc(&c, elemBytes));
      CUDA_RUNTIME(cudaMalloc(&n, elemBytes));
      assert(uintptr_t(c) % elemSz == 0 && "allocation should be aligned");
      assert(uintptr_t(n) % elemSz == 0 && "allocation should be aligned");
      currDataPtrs_[i] = c;
      nextDataPtrs_[i] = n;
    }
  }
};