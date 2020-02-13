#pragma once

#include <vector>

template <typename T> using Mat2D = std::vector<std::vector<T>>;

template <typename T> Mat2D<T> make_mat2d(size_t n, const T &val) {
  return std::vector<std::vector<T>>(n, std::vector<T>(n, val));
}

inline Mat2D<double> make_reciprocal(const Mat2D<double> &m) {
  Mat2D<double> ret;

  for (size_t i = 0; i < m.size(); ++i) {
    std::vector<double> row;

    for (double e : m[i]) {
      if (0 == e) {
        row.push_back(std::numeric_limits<double>::infinity());
      } else {
        row.push_back(1.0 / e);
      }
    }
    ret.push_back(row);
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
