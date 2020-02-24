#pragma once

#include <cstring>
#include <iostream>
#include <vector>

struct Shape {
  int64_t x;
  int64_t y;
  Shape(int64_t x, int64_t y) : x(x), y(y) {}

  int64_t flatten() const noexcept { return x * y; }
  bool operator==(const Shape &rhs) const noexcept {
    return x == rhs.x && y == rhs.y;
  }
  bool operator!=(const Shape &rhs) const noexcept { return !((*this) == rhs); }
};

template <typename T> class Mat2D {
private:
  void swap(Mat2D &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(shape_, other.shape_);
  }

public:
  class Row {
    friend class Mat2D;
    T *data_;
    int64_t size_;

    Row(T *data, int64_t size) : data_(data), size_(size) {}

  public:
    T &operator[](int64_t i) noexcept { return data_[i]; }
    const T &operator[](int64_t i) const noexcept { return data_[i]; }

    T *begin() const { return data_; }
    T *end() const { return data_ + size_; }

    /* overwrite a row with another row */
    Row &operator=(const Row &rhs) {
      assert(rhs.size_ == size_);
      std::memcpy(data_, rhs.data_, size_ * sizeof(T));
      return *this;
    }

    /* overwrite a row with a vector */
    Row &operator=(const std::vector<T> &rhs) {
      assert(rhs.size() == size_);
      std::memcpy(data_, rhs.data(), size_ * sizeof(T));
      return *this;
    }
  };

  class ConstRow {
    friend class Mat2D;
    const T *data_;
    int64_t size_;

    ConstRow(const T *data, int64_t size) : data_(data), size_(size) {}

  public:
    const T &operator[](int64_t i) const noexcept { return data_[i]; }

    const T *begin() const { return data_; }
    const T *end() const { return data_ + size_; }
  };

  std::vector<T> data_;
  Shape shape_;

  Mat2D() : shape_(0, 0) {}
  Mat2D(int64_t x, int64_t y) : data_(x * y), shape_(x, y) {}
  Mat2D(int64_t x, int64_t y, const T &v) : data_(x * y, v), shape_(x, y) {}
  Mat2D(Shape s) : Mat2D(s.x, s.y) {}
  Mat2D(Shape s, const T &val) : Mat2D(s.x, s.y, val) {}

  Mat2D(const std::initializer_list<std::initializer_list<T>> &ll) : Mat2D() {

    if (ll.size() > 0) {
      resize(ll.begin()->size(), ll.size());
    }

    auto llit = ll.begin();
    for (size_t i = 0; i < shape_.y; ++i, ++llit) {
      assert(llit->size() == shape_.x);
      auto lit = llit->begin();
      for (size_t j = 0; j < shape_.x; ++j, ++lit) {
        at(i, j) = *lit;
      }
    }
  }

  ~Mat2D() {}

  Mat2D(const Mat2D &other) : Mat2D(other.shape_) { data_ = other.data_; }

  Mat2D &operator=(const Mat2D &rhs) = default;
  Mat2D(Mat2D &&other) = default;
  Mat2D &operator=(Mat2D &&rhs) = default;

  inline T &at(int64_t i, int64_t j) noexcept {
    assert(i < shape_.x);
    assert(j < shape_.y);
    return data_[i * shape_.x + j];
  }
  inline const T &at(int64_t i, int64_t j) const noexcept {
    assert(i < shape_.x);
    assert(j < shape_.y);
    return data_[i * shape_.x + j];
  }

  // pointer to row i
  Row operator[](int64_t i) noexcept {
    assert(i < shape_.y);
    return Row(&data_[i * shape_.x], shape_.x);
  }
  ConstRow operator[](int64_t i) const noexcept {
    assert(i < shape_.y);
    return ConstRow(&data_[i * shape_.x], shape_.x);
  }

  /* grow or shrink to [x,y], preserving top-left corner of matrix */
  void resize(int64_t x, int64_t y) {
    Mat2D mat(x, y);

    const int64_t copyRows = std::min(mat.shape_.y, shape_.y);
    const int64_t copyCols = std::min(mat.shape_.x, shape_.x);

    for (int64_t i = 0; i < copyRows; ++i) {
      std::memcpy(&mat.at(i, 0), &at(i, 0), copyCols * sizeof(T));
    }
    swap(mat);
  }

  /* add a row to the matrix */
  void push_back(const std::vector<T> &row) {
    if (shape_.y != 0) {
      assert(row.size() == shape_.x);
    }
    resize(row.size(), shape_.y + 1);

    operator[](shape_.y - 1) = row;
  }

  inline const Shape &shape() const noexcept { return shape_; }

  bool operator==(const Mat2D &rhs) const noexcept {
    if (shape_ != rhs.shape_) {
      return false;
    }
    for (int64_t i = 0; i < shape_.y; ++i) {
      for (int64_t j = 0; j < shape_.x; ++j) {
        if (data_[i * shape_.x + j] != rhs.data_[i * shape_.x + j]) {
          return false;
        }
      }
    }
    return true;
  }
};

inline Mat2D<double> make_reciprocal(const Mat2D<double> &m) {
  Mat2D<double> ret(m.shape());

  for (size_t i = 0; i < m.shape().y; ++i) {
    for (size_t j = 0; j < m.shape().x; ++j) {
      double e = m.at(i,j);
      if (0 == e) {
        ret.at(i,j) = std::numeric_limits<double>::infinity();
      } else {
        ret.at(i,j) = 1.0 / e;
      }
    }
  }

  return ret;
}

/*! return a copy of `m` with ret[i][j]=m[map[i]][map[j]]
 */
template <typename T>
Mat2D<T> permute(const Mat2D<T> &m, std::vector<size_t> map) {
  assert(m.size() == map.size());
  for (auto &r : m) {
    assert(r.size() == map.size());
  }

  Mat2D<T> result(m.size());
  for (auto &v : result) {
    v.resize(m[0].size());
  }

  for (size_t r = 0; r < m.size(); ++r) {
    for (size_t c = 0; c < m.size(); ++c) {
      assert(r < map.size());
      assert(c < map.size());
      size_t nr = map[r];
      size_t nc = map[c];
      assert(nr < result.size());
      assert(nc < result[nr].size());
      result[r][c] = m[nr][nc];
    }
  }

  return result;
}
