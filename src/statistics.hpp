#pragma once

#include <cstdlib>

struct Statistics {
  Statistics();
  size_t n;
  double sum_;
  double min_;
  double max_;

  void reset();

  void insert(double d);

  double avg() const noexcept;
  double min() const noexcept;
  double max() const noexcept;
  double count() const noexcept;
};