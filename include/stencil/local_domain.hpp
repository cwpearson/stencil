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

  std::vector<size_t> elem_size() const { return dataElemSize_; }

  std::vector<char *> data() const { return dataPtrs_; }
  size_t pitch() const { assert(0); }

  // return the position of the face relative to get_data()
  Dim3 face_pos() const { assert(0); }

  // return the extent of the face in every dimension
  Dim3 face_extent() const { assert(0); }

  // return the 3d size of the actual allocation
  Dim3 raw_size() const { assert(0); }

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