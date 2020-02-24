#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mat2d.hpp"

namespace qap {

namespace detail {
inline double cost(const Mat2D<double> &w,      // weight
                   const Mat2D<double> &d,      // distance
                   const std::vector<size_t> &f // bijection
) {
  assert(w.shape().x == w.shape().y);
  assert(d.shape() == w.shape());
  assert(w.shape().x == f.size());

  double ret = 0;

  for (size_t a = 0; a < w.shape().y; ++a) {
    for (size_t b = 0; b < w.shape().x; ++b) {
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

inline std::vector<size_t> solve(const Mat2D<double> &w, Mat2D<double> &d) {

  assert(w.shape() == d.shape());
  assert(w.shape().x == d.shape().y);

  std::vector<size_t> f(w.shape().x);
  for (size_t i = 0; i < w.shape().x; ++i) {
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

inline std::vector<size_t> solve_catch(const Mat2D<double> &w, Mat2D<double> &d) {

  assert(w.shape() == d.shape());
  assert(w.shape().x == w.shape().y);

  // initial guess
  std::vector<size_t> bestF(w.shape().x);
  for (size_t i = 0; i < w.shape().x; ++i) {
    bestF[i] = i;
  }
  double bestCost = std::numeric_limits<double>::infinity();

  bool improved;
 do {
    improved = false;

    std::vector<size_t> imprF = bestF;
    double imprCost = bestCost;

    // find the best improvement for swapping a single location
    for (size_t i = 0; i < w.shape().x; ++i) {
      for (size_t j = i+1; j < w.shape().x; ++j) {
        std::vector<size_t> f = bestF;
        auto tmp = f[i];
        f[i] = f[j];
        f[j] = tmp;
        const double cost = detail::cost(w,d,f);
        if (cost < imprCost) {
          imprF = f;
          imprCost = cost;
          improved = true;
        }
      }
    }

  if (improved) {
    bestF = imprF;
    bestCost = imprCost;
  }

  } while (improved);


  return bestF;
}



} // namespace qap
