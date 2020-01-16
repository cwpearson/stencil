#pragma once

#include <ostream>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Dim3 {
public:
  int64_t x;
  int64_t y;
  int64_t z;

public:
  CUDA_CALLABLE_MEMBER Dim3() {}

#ifdef __CUDACC__
  /* cuda dim3 ctor */
  CUDA_CALLABLE_MEMBER Dim3(const dim3 other)
      : x(other.x), y(other.y), z(other.z) {}
#endif

  CUDA_CALLABLE_MEMBER Dim3(int64_t x, int64_t y, int64_t z)
      : x(x), y(y), z(z) {}
  /* copy ctor*/
  CUDA_CALLABLE_MEMBER Dim3(const Dim3 &d) : x(d.x), y(d.y), z(d.z) {}
  /* move ctor */
  CUDA_CALLABLE_MEMBER Dim3(Dim3 &&d)
      : x(std::move(d.x)), y(std::move(d.y)), z(std::move(d.z)) {}

  /* copy assign */
  CUDA_CALLABLE_MEMBER Dim3 &operator=(const Dim3 &d) {
    x = d.x;
    y = d.y;
    z = d.z;
    return *this;
  }

  /* move assign */
  CUDA_CALLABLE_MEMBER Dim3 &operator=(Dim3 &&d) {
    x = std::move(d.x);
    y = std::move(d.y);
    z = std::move(d.z);
    return *this;
  }

  void swap(Dim3 &d) {
    std::swap(x, d.x);
    std::swap(y, d.y);
    std::swap(z, d.z);
  }

  CUDA_CALLABLE_MEMBER int64_t &operator[](const size_t idx) {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    else if (idx == 2)
      return z;
    assert(0 && "only 3 dimensions!");
    return x;
  }
  CUDA_CALLABLE_MEMBER const int64_t &operator[](const size_t idx) const {
    return operator[](idx);
  }

  /*! \brief elementwise max
   */
  Dim3 max(const Dim3 &other) const {
    Dim3 result;
    result.x = std::max(x, other.x);
    result.y = std::max(x, other.y);
    result.z = std::max(x, other.z);
    return result;
  }

  CUDA_CALLABLE_MEMBER bool any() const { return x != 0 || x != 0 || z != 0; }
  CUDA_CALLABLE_MEMBER bool all() const { return x != 0 && x != 0 && z != 0; }

  CUDA_CALLABLE_MEMBER size_t flatten() const { return x * y * z; }

  CUDA_CALLABLE_MEMBER bool operator>(const Dim3 &rhs) const {
    return x > rhs.x && y > rhs.y && z > rhs.z;
  }
  CUDA_CALLABLE_MEMBER bool operator<(const Dim3 &rhs) const {
    return x < rhs.x && y < rhs.y && z < rhs.z;
  }
  CUDA_CALLABLE_MEMBER bool operator>=(const int64_t rhs) const {
    return x >= rhs && y >= rhs && z >= rhs;
  }

  CUDA_CALLABLE_MEMBER Dim3 &operator%=(const Dim3 &rhs) {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    return *this;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator%(const Dim3 &rhs) const {
    Dim3 result = *this;
    result %= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER Dim3 &operator+=(const Dim3 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator+(const Dim3 &rhs) const {
    Dim3 result = *this;
    result += rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER Dim3 &operator-=(const Dim3 &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator-(const Dim3 &rhs) const {
    Dim3 result = *this;
    result -= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator-(int64_t rhs) const {
    Dim3 result = *this;
    result.x -= rhs;
    result.y -= rhs;
    result.z -= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER Dim3 &operator*=(const Dim3 &rhs) {
    x *= rhs.x;
    y *= rhs.y;
    z *= rhs.z;
    return *this;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator*(const Dim3 &rhs) const {
    Dim3 result = *this;
    result *= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator*(int64_t rhs) const {
    Dim3 result = *this;
    result.x *= rhs;
    result.y *= rhs;
    result.z *= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER Dim3 &operator/=(const Dim3 &rhs) {
    x /= rhs.x;
    y /= rhs.y;
    z /= rhs.z;
    return *this;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator/(const Dim3 &rhs) const {
    Dim3 result = *this;
    result /= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER bool operator==(const Dim3 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  CUDA_CALLABLE_MEMBER bool operator!=(const Dim3 &rhs) const {
    return x != rhs.x || y != rhs.y || z == rhs.z;
  }

#ifdef __CUDACC__
  /* convertible to CUDA dim3 */
  CUDA_CALLABLE_MEMBER operator dim3() const { return dim3(x, y, z); }
#endif

  CUDA_CALLABLE_MEMBER Dim3 wrap(const Dim3 &lims) {
    if (x >= lims.x) {
      x = x % lims.x;
    }
    while (x < 0) {
      x += lims.x;
    }
    if (y >= lims.y) {
      y = y % lims.y;
    }
    while (y < 0) {
      y += lims.y;
    }
    if (z >= lims.z) {
      z = z % lims.z;
    }
    while (z < 0) {
      z += lims.z;
    }

    return *this;
  }
};

inline std::ostream &operator<<(std::ostream &os, const Dim3 &d) {
  os << '[' << d.x << ',' << d.y << ',' << d.z << ']';
  return os;
}