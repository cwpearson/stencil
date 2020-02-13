#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mat2d.hpp"

namespace qap {

namespace detail {
double cost(const Mat2D<double> &w,      // weight
            const Mat2D<double> &d,      // distance
            const std::vector<size_t> &f // bijection
) {
  assert(w.size() == f.size());
  assert(d.size() == w.size());

  double ret = 0;

  for (size_t a = 0; a < w.size(); ++a) {
    for (size_t b = 0; b < w[a].size(); ++b) {
      double p;
      if (0 == w[a][b] || 0 == d[f[a]][f[b]]) {
        p = 0;
      } else {
        p = w[a][b] * d[f[a]][f[b]];
      }
      ret += p;
    }
  }

  return ret;
}
} // namespace detail

std::vector<size_t> solve(const Mat2D<double> &w, Mat2D<double> &d) {

  assert(w.size() == d.size());
  if (w.size() > 0) {
    assert(w[0].size() == d[0].size());
  }

  std::vector<size_t> f(w.size());
  for (size_t i = 0; i < w.size(); ++i) {
    f[i] = i;
  }

  std::vector<size_t> bestF = f;
  double bestCost = detail::cost(w, d, f);
  do {
    const double cost = detail::cost(w, d, f);
    if (bestCost > cost) {
      bestF = f;
      bestCost = cost;
    }
  } while (std::next_permutation(f.begin(), f.end()));

  return bestF;
}
} // namespace qap
