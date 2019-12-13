#pragma once

#include <ostream>

class Dim3 {
public:
  int64_t x;
  int64_t y;
  int64_t z;

public:
  Dim3() {}
  Dim3(int64_t x, int64_t y, int64_t z) : x(x), y(y), z(z) {}

  int64_t &operator[](const size_t idx) {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    else if (idx == 2)
      return z;
    assert(0 && "only 3 dimensions!");
    return x;
  }
  const int64_t &operator[](const size_t idx) const { return operator[](idx); }

  /*! \brief elementwise max
   */
  Dim3 max(const Dim3 &other) const {
    Dim3 result;
    result.x = std::max(x, other.x);
    result.y = std::max(x, other.y);
    result.z = std::max(x, other.z);
    return result;
  }

  bool any() const { return x != 0 || x != 0 || z != 0; }

  size_t flatten() const { return x * y * z; }

  Dim3 operator>(const Dim3 &rhs) const {
    Dim3 result;
    result.x = x > rhs.x;
    result.y = y > rhs.y;
    result.z = z > rhs.z;
    return result;
  }

  Dim3 &operator%=(const Dim3 &rhs) {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    return *this;
  }

  Dim3 operator%(const Dim3 &rhs) const {
    Dim3 result = *this;
    result %= rhs;
    return result;
  }

  Dim3 &operator+=(const Dim3 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  Dim3 operator+(const Dim3 &rhs) const {
    Dim3 result = *this;
    result += rhs;
    return result;
  }

  Dim3 &operator-=(const Dim3 &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  Dim3 operator-(const Dim3 &rhs) const {
    Dim3 result = *this;
    result -= rhs;
    return result;
  }

  Dim3 &operator*=(const Dim3 &rhs) {
    x *= rhs.x;
    y *= rhs.y;
    z *= rhs.z;
    return *this;
  }

  Dim3 operator*(const Dim3 &rhs) {
    Dim3 result = *this;
    result *= rhs;
    return result;
  }

  Dim3 &operator/=(const Dim3 &rhs) {
    x /= rhs.x;
    y /= rhs.y;
    z /= rhs.z;
    return *this;
  }

  Dim3 operator/(const Dim3 &rhs) {
    Dim3 result = *this;
    result /= rhs;
    return result;
  }

  bool operator==(const Dim3 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  bool operator!=(const Dim3 &rhs) const {
    return x != rhs.x || y != rhs.y || z == rhs.z;
  }

  Dim3 wrap(const Dim3 &lims) {
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