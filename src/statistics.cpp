#include <algorithm>
#include <limits>

#include "statistics.hpp"

Statistics::Statistics() { reset(); }

void Statistics::reset() {
  n = 0;
  sum_ = 0;
  min_ = std::numeric_limits<double>::infinity();
  max_ = -1 * std::numeric_limits<double>::infinity();
}

void Statistics::insert(double d) {
  ++n;
  sum_ += d;
  min_ = std::min(min_, d);
  max_ = std::max(max_, d);
}

double Statistics::avg() const noexcept { return sum_ / n; }
double Statistics::min() const noexcept { return min_; }
double Statistics::max() const noexcept { return max_; }
double Statistics::count() const noexcept { return n; }
