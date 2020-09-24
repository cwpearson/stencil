#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mat2d.hpp"

namespace qap {

namespace detail {

inline double cost_product(double we, double de) {
  if (0 == we || 0 == de) {
    return 0;
  } else {
    return we * de;
  }
}

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
      p = cost_product(w.at(a, b), d.at(f[a], f[b]));
      // if (0 == w.at(a,b) || 0 == d.at(f[a],f[b])) {
      //   p = 0;
      // } else {
      //   p = w.at(a,b) * d.at(f[a],f[b]);
      // }
      ret += p;
    }
  }

  return ret;
}
} // namespace detail

inline std::vector<size_t> solve(const Mat2D<double> &w, Mat2D<double> &d, double *costp = nullptr) {

  typedef std::chrono::system_clock Clock;
  typedef std::chrono::duration<double> Duration;

  auto stop = Clock::now() + Duration(10);

  assert(w.shape() == d.shape());
  assert(w.shape().x == d.shape().y);

  std::vector<size_t> f(w.shape().x);
  for (size_t i = 0; i < w.shape().x; ++i) {
    f[i] = i;
  }

  std::vector<size_t> bestF = f;
  double bestCost = detail::cost(w, d, f);
  do {
    if (Clock::now() > stop) {
      LOG_WARN("qap::solve timed out");
      break;
    }
    const double cost = detail::cost(w, d, f);
    if (bestCost > cost) {
      bestF = f;
      bestCost = cost;
    }
  } while (std::next_permutation(f.begin(), f.end()));

  if (costp) {
    *costp = bestCost;
  }

  return bestF;
}

inline std::vector<size_t> solve_catch(const Mat2D<double> &w, Mat2D<double> &d, double *costp = nullptr) {

  assert(w.shape() == d.shape());
  assert(w.shape().x == w.shape().y);

  // initial guess
  std::vector<size_t> bestF(w.shape().x);
  for (size_t i = 0; i < w.shape().x; ++i) {
    bestF[i] = i;
  }
  double bestCost = detail::cost(w, d, bestF);

  bool improved;
  do {
    improved = false;

    std::vector<size_t> imprF = bestF;
    double imprCost = bestCost;

    // find the best improvement for swapping a single location
    for (size_t i = 0; i < w.shape().x; ++i) {
      // std::cerr << i << "\n";
      for (size_t j = i + 1; j < w.shape().x; ++j) {

        // adjust cost for swap
        // we will be adjusting placements i and j
        // remove contribution from entry i,j on cost
        std::vector<size_t> f = bestF;
        double cost = bestCost;
        // for (size_t m = 0; m < w.shape().x; ++m) {
        //   for (size_t n = 0; n < w.shape().x; ++n) {
        //     if (m == i || m == j || n == i || n == j) {
        //       cost -= detail::cost_product(w.at(m,n), d.at(f[m], f[n]));
        //     }
        //   }
        // }
        for (size_t k = 0; k < w.shape().x; ++k) {
          cost -= detail::cost_product(w.at(i, k), d.at(f[i], f[k]));
          cost -= detail::cost_product(w.at(j, k), d.at(f[j], f[k]));
          // don't double-count overlaps
          if (k != i && k != j) {
            cost -= detail::cost_product(w.at(k, i), d.at(f[k], f[i]));
            cost -= detail::cost_product(w.at(k, j), d.at(f[k], f[j]));
          }
        }
        auto tmp = f[i];
        f[i] = f[j];
        f[j] = tmp;
        // for (size_t m = 0; m < w.shape().x; ++m) {
        //   for (size_t n = 0; n < w.shape().x; ++n) {
        //     if (m == i || m == j || n == i || n == j) {
        //       cost += detail::cost_product(w.at(m,n), d.at(f[m], f[n]));
        //     }
        //   }
        // }
        for (size_t k = 0; k < w.shape().x; ++k) {
          cost += detail::cost_product(w.at(i, k), d.at(f[i], f[k]));
          cost += detail::cost_product(w.at(j, k), d.at(f[j], f[k]));
          // don't double-count overlaps
          if (k != i && k != j) {
            cost += detail::cost_product(w.at(k, i), d.at(f[k], f[i]));
            cost += detail::cost_product(w.at(k, j), d.at(f[k], f[j]));
          }
        }

        // if(cost != detail::cost(w,d,f)) {
        //   fprintf(stderr, "%.20f %.20f\n", cost, detail::cost(w,d,f));
        // }
        // const double cost = detail::cost(w,d,f);
        if (cost < imprCost) {
          // std::cerr << cost << " " << imprCost << " " << i << " " << j << "\n";
          imprF = f;
          imprCost = cost;
          improved = true;

          // goto nextiter;
        }
      }
    }

    // nextiter:
    if (improved) {
      bestF = imprF;
      bestCost = imprCost;
      // std::cerr << bestCost << "\n";
    }

  } while (improved);

  if (costp) {
    *costp = bestCost;
  }
  return bestF;
}

} // namespace qap
