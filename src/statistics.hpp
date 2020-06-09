#pragma once

#include <cstdlib>
#include <vector>

class Statistics {
private:
  std::vector<double> x;

public:
  void clear();
  void insert(double d);
  double avg() const;
  double min() const;
  double max() const;
  size_t count() const noexcept;
  double trimean();
  double med();
  double stddev() const;
};