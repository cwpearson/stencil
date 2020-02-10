#pragma once

#include <vector>

template <typename T> using Mat2D = std::vector<std::vector<T>>;

template <typename T> Mat2D<T> make_mat2d(size_t n, const T &val) {
  return std::vector<std::vector<T>>(n, std::vector<T>(n, val));
}
