#pragma once

#include <cassert>
#include <ostream>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

inline int64_t nextPowerOfTwo(int64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  return x;
}

class Dim3 {
public:
  int64_t x;
  int64_t y;
  int64_t z;

public:
#ifdef __CUDACC__
  /* ctor from CUDA dim3 */
  CUDA_CALLABLE_MEMBER Dim3(const dim3 other) : x(other.x), y(other.y), z(other.z) {}
#endif

  CUDA_CALLABLE_MEMBER Dim3(int64_t _x, int64_t _y, int64_t _z) : x(_x), y(_y), z(_z) {}

  Dim3() = default;                         // default ctor
  Dim3(const Dim3 &d) = default;            // copy ctor
  Dim3(Dim3 &&d) = default;                 // move ctor
  Dim3 &operator=(const Dim3 &d) = default; // copy assign
  Dim3 &operator=(Dim3 &&d) = default;      // move assign

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
  CUDA_CALLABLE_MEMBER const int64_t &operator[](const size_t idx) const { return operator[](idx); }

  /*! \brief elementwise max
   */
  Dim3 max(const Dim3 &other) const {
    Dim3 result;
    result.x = std::max(x, other.x);
    result.y = std::max(x, other.y);
    result.z = std::max(x, other.z);
    return result;
  }

  CUDA_CALLABLE_MEMBER bool any() const { return x != 0 || y != 0 || z != 0; }
  CUDA_CALLABLE_MEMBER bool all() const { return x != 0 && y != 0 && z != 0; }

  CUDA_CALLABLE_MEMBER size_t flatten() const { return x * y * z; }

  CUDA_CALLABLE_MEMBER bool operator<(const Dim3 &rhs) const {
    if (x < rhs.x) {
      return true;
    } else if (x > rhs.x) {
      return false;
    } else {
      if (y < rhs.y) {
        return true;
      } else if (y > rhs.y) {
        return false;
      } else {
        return z < rhs.z;
      }
    }
  }

  CUDA_CALLABLE_MEMBER bool all_lt(const int64_t rhs) const { return x < rhs && y < rhs && z < rhs; }

  CUDA_CALLABLE_MEMBER bool all_lt(const Dim3 rhs) const { return x < rhs.x && y < rhs.y && z < rhs.z; }

  CUDA_CALLABLE_MEMBER bool all_gt(const int64_t rhs) const { return x > rhs && y > rhs && z > rhs; }

  CUDA_CALLABLE_MEMBER bool all_ge(const int64_t rhs) const { return x >= rhs && y >= rhs && z >= rhs; }

  CUDA_CALLABLE_MEMBER bool any_lt(const int64_t rhs) const { return x < rhs || y < rhs || z < rhs; }

  CUDA_CALLABLE_MEMBER bool any_gt(const int64_t rhs) const { return x > rhs || y > rhs || z > rhs; }

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

  CUDA_CALLABLE_MEMBER Dim3 &operator*=(const double &rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
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

  CUDA_CALLABLE_MEMBER Dim3 &operator/=(const double &rhs) {
    x /= rhs;
    y /= rhs;
    z /= rhs;
    return *this;
  }

  CUDA_CALLABLE_MEMBER Dim3 operator/(const Dim3 &rhs) const {
    Dim3 result = *this;
    result /= rhs;
    return result;
  }

  CUDA_CALLABLE_MEMBER bool operator==(const Dim3 &rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }

  CUDA_CALLABLE_MEMBER bool operator!=(const Dim3 &rhs) const { return x != rhs.x || y != rhs.y || z == rhs.z; }

#ifdef __CUDACC__
  /* convertible to CUDA dim3 */
  CUDA_CALLABLE_MEMBER operator dim3() const {
    assert(x > 0);
    assert(y > 0);
    assert(z > 0);

    return dim3((unsigned int)x, (unsigned int)y, (unsigned int)z);
  }
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

  /* map `threads` threads into a 3d shape similar to `extent`
   */
  static Dim3 make_block_dim(const Dim3 extent, int64_t threads) {
    assert(extent.x >= 0);
    assert(extent.y >= 0);
    assert(extent.z >= 0);
    threads = std::min(threads, int64_t(1024)); // max of 1024 threads in a block
    Dim3 ret;
    ret.x = std::min(threads, nextPowerOfTwo(extent.x));
    assert(ret.x);
    threads /= ret.x;
    ret.y = std::min(threads, nextPowerOfTwo(extent.y));
    assert(ret.y);
    threads /= ret.y;
    ret.z = std::min(threads, nextPowerOfTwo(extent.z));

    // cap dimensions by CUDA max size
    ret.z = std::min(ret.z, int64_t(64));
    ret.y = std::min(ret.y, int64_t(1024));
    ret.x = std::min(ret.x, int64_t(1024));

    assert(ret.x * ret.y * ret.z <= 1024);
    return ret;
  }

#if 0
// TODO: smarter shaping of blocks to extents
  static Dim3 make_block_dim2(const Dim3 extent, int64_t threads) {
    threads = std::min(threads, int64_t(1024)); // max of 1024 threads in a block
    assert(extent.x >= 0);
    assert(extent.y >= 0);
    assert(extent.z >= 0);
    double sf = 1.1;
    Dim3 scaled = extent;

    do {
      Dim3 ret;
      ret.x = nextPowerOfTwo(scaled.x);
      ret.y = nextPowerOfTwo(scaled.y);
      ret.z = nextPowerOfTwo(scaled.z);

      ret.z = std::min(ret.z, int64_t(64));
      ret.y = std::min(ret.y, int64_t(1024));
      ret.x = std::min(ret.x, int64_t(1024));

      if (ret.flatten() < threads / 2) {
        scaled.x = std::ceil(scaled.x * sf);
        scaled.y = std::ceil(scaled.y * sf);
        scaled.z = std::ceil(scaled.z * sf);
        std::cerr << scaled.x << " " << scaled.y << " " << scaled.z << "=scaled\n";
      } else if (ret.flatten() > threads) {
        scaled.x = std::floor(scaled.x / sf);
        scaled.y = std::floor(scaled.y / sf);
        scaled.z = std::floor(scaled.z / sf);
        scaled.x = std::max(scaled.x, int64_t(1));
        scaled.y = std::max(scaled.y, int64_t(1));
        scaled.z = std::max(scaled.z, int64_t(1));
        std::cerr << scaled.x << " " << scaled.y << " " << scaled.z << "=scaled\n";
      } else {
        // std::cerr << ret << "=ret\n";
        assert(ret.x * ret.y * ret.z <= 1024);
        return ret;
      }

    } while (true);


  }
  #endif

};

inline std::ostream &operator<<(std::ostream &os, const Dim3 &d) {
  os << '[' << d.x << ',' << d.y << ',' << d.z << ']';
  return os;
}

#undef CUDA_CALLABLE_MEMBER