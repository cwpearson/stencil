#include "stencil/numeric.hpp"

#include <algorithm>
#include <cmath>

template<typename T>
std::vector<T> prime_factors(T n) {
  std::vector<T> result;
  if (0 == n) {
    return result;
  }
  while (n % 2 == 0) {
    result.push_back(2);
    n = n / 2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    while (n % i == 0) {
      result.push_back(i);
      n = n / i;
    }
  }
  if (n > 2)
    result.push_back(n);
  std::sort(result.begin(), result.end(), [](T a, T b) { return b < a; });
  return result;
}

template std::vector<int> prime_factors(int n);
template std::vector<int64_t> prime_factors(int64_t n);
