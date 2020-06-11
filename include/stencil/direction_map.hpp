#pragma once

#include <array>

// #define SPEW(x) std::cerr << "SPEW[" << __FILE__ << ":" << __LINE__ << "] "
// <<  x << "\n";
#define SPEW(x)

/*! store a T associated with each direction vector
 */
template <typename T> class DirectionMap {

private:
  std::array<std::array<std::array<T, 3>, 3>, 3> data_;

public:
  typedef int index_type;

  DirectionMap() = default;

  /* construct filled with `val` */
  DirectionMap(const T &val) {
    for (int z = 0; z <= 2; ++z) {
      for (int y = 0; y <= 2; ++y) {
        for (int x = 0; x <= 2; ++x) {
          data_[z][y][x] = val;
        }
      }
    }
  }

  bool operator==(const DirectionMap &rhs) const noexcept {
    return data_ == rhs.data_;
  }

  T &at(index_type x, index_type y, index_type z) noexcept {
    assert(x >= 0 && x <= 2);
    assert(y >= 0 && y <= 2);
    assert(z >= 0 && z <= 2);
    return data_[z][y][x];
  }

  T &at_dir(index_type x, index_type y, index_type z) noexcept {
    assert(x >= -1 && x <= 1);
    assert(y >= -1 && y <= 1);
    assert(z >= -1 && z <= 1);
    SPEW("x,y,z=" << x << "," << y << "," << z);
    return data_[z + 1][y + 1][x + 1];
  }

  const T &at_dir(index_type x, index_type y, index_type z) const noexcept {
    assert(x >= -1 && x <= 1);
    assert(y >= -1 && y <= 1);
    assert(z >= -1 && z <= 1);
    SPEW("x,y,z=" << x << "," << y << "," << z);
    return data_[z + 1][y + 1][x + 1];
  }
};

#undef SPEW