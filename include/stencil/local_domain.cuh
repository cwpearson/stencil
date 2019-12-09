#pragma once

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
  std::vector<char *> dataPtrs_;
  std::vector<size_t> dataElemSize_;

  int dev_;             // CUDA device
  cudaStream_t stream_; // CUDA stream

public:
  LocalDomain(Dim3 sz, int dev) : sz_(sz), dev_(dev), stream_(0) {}

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

  /*
   */
  size_t num_data() const {
    assert(dataElemSize_.size() == dataPtrs_.size());
    return dataPtrs_.size();
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

  std::vector<size_t> elem_size() const { return dataElemSize_; }

  char *data(size_t idx) const {
    assert(idx < dataPtrs_.size());
    return dataPtrs_[idx];
  }

  size_t pitch() const {
#warning pitch is unimplemented
    assert(0);
    return 0;
  }

  // return the position of the face relative to get_data()
  // positive or negative
  // x=0, y=1, z=2
  Dim3 face_pos(bool pos, const size_t dim) const {
    switch (pos) {
    case false: // negative-facing
      switch (dim) {
      case 0:
        return Dim3(0, 0, 0);
      case 1:
        return Dim3(0, 0, 0);
      case 2:
        return Dim3(0, 0, 0);
      }
    case true: // positive-facing
      switch (dim) {
      case 0:
        return Dim3(radius_ + sz_.x, 0, 0); // +x
      case 1:
        return Dim3(0, radius_ + sz_.y, 0); // +y
      case 2:
        return Dim3(0, 0, radius_ + sz_.z); // +z
      }
    }

    assert(0 && "unreachable");
    return Dim3(-1, -1, -1);
  }

  // return the extent of the face in every dimension
  Dim3 face_extent(bool pos, const size_t dim) const {
    switch (dim) {
    case 0:
      return Dim3(0, sz_.y, sz_.z);
    case 1:
      return Dim3(sz_.x, 0, sz_.z);
    case 2:
      return Dim3(sz_.x, sz_.y, 0);
    }

    assert(0 && "unreachable");
    return Dim3(-1, -1, -1);
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
  }
};