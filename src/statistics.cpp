#include <algorithm>
#include <numeric>
#include <cmath>

#include "statistics.hpp"

void Statistics::clear() { x.clear(); }

void Statistics::insert(double d) { x.push_back(d); }

double Statistics::avg() const {
  return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}
double Statistics::min() const {
  return *std::min_element(x.begin(), x.end());
}
double Statistics::max() const {
  return *std::max_element(x.begin(), x.end());
}
size_t Statistics::count() const noexcept { return x.size(); }
double Statistics::trimean() {
  std::sort(x.begin(), x.end());
  size_t q1 = x.size() / 4 * 1;
  size_t q2 = x.size() / 4 * 2;
  size_t q3 = x.size() / 4 * 3;
  return (x[q1] + 2 * x[q2] + x[q3]) / 4;
}

double Statistics::med() {
  std::sort(x.begin(), x.end());
  if (x.size() % 2) {
    return x[x.size() / 2];
  } else {
    return x[x.size() / 2] + x[x.size() / 2 + 1];
  }
}

double Statistics::stddev() const {
  double xbar = avg();
  double acc = 0;
  for (double xi : x) {
    acc += std::pow(xi - xbar, 2);
  }
  return std::sqrt(acc / (x.size() - 1));
}