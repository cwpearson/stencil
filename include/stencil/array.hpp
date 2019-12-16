#pragma once

enum class device_type {
  cpu,
#if STENCIL_USE_CUDA
  gpu,
#endif
};

template <typename T, device_type ty = device_type::cpu> class Array {
private:
  size_t size_;
  size_t pitch_;
  T *data_;

public:
  Array() : size_(0), pitch_(0), data_(nullptr) {}
  Array(size_t n) : Array() { resize(n); }
  ~Array() { resize(0); }

  void resize(size_t n) {
    if (n != size_) {
      delete[] data_;
      data_ = nullptr;
      if (n > 0) {
        data_ = new T[n];
      }
      size_ = n;
    }
  }

  size_t size() const noexcept { return size_; }

  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  T &operator[](size_t n) {
      return data_[n];
  }
  const T& operator[](size_t n) const {
      return data_[n];
  }

  friend void swap(Array &a, Array &b) {
      using std::swap;
      swap(a.size_, b.size_);
      swap(a.pitch_, b.pitch_);
      swap(a.data_, b.data_);
  }
};

#if PANGOLIN_USE_CUDA
template <typename T> class Array<T, device_type::gpu> {
private:
  size_t size_;
  size_t pitch_;
  T *data_;
  int dev_;

public:
  Array(int dev = 0) : Array<device_type::cpu>(), dev_(dev) {}
  Array(size_t n, int dev = 0) : Array(dev) { resize(n); }

  void resize(size_t n) {
    if (n != size_) {
      CUDA_RUNTIME(cudaFree(data_));
      data_ = nullptr;
      if (n > 0) {
            CUDA_RUNTIME(cudaMalloc(&data_, sizeof(T) * n);
      }
      size_ = n;
    }
  }

  friend void swap(Array &a, Array &b) {
      using std::swap;
      swap(a.size_, b.size_);
      swap(a.pitch_, b.pitch_);
      swap(a.data_, b.data_);
      swap(a.dev_, b.dev_);
  }

};
#endif