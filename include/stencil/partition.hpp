#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "dim3.hpp"

class Partition {
public:
  // get the MPI rank for a particular index in the rank space
  virtual int get_rank(const Dim3 &rankIdx) const = 0;

  // get the gpu for a particular index in the GPU space
  virtual int get_gpu(const Dim3 &idx) const = 0;

  // the index of a GPU in the GPU space
  virtual Dim3 gpu_idx(int gpu) const = 0;
  // the index of the rank in the rank space
  virtual Dim3 rank_idx(int rank) const = 0;

  // the extent of the gpu space
  virtual Dim3 gpu_dim() const = 0;
  // the extent of the rank space
  virtual Dim3 rank_dim() const = 0;

  // get the target size of the local domain in this partition
  virtual Dim3 local_domain_size(const Dim3 &rankIdx,
                                 const Dim3 &gpuIdx) const = 0;
};

/*! Prime-factor partitioner
 */
class PFP : public Partition {
private:
  int gpus_;
  int ranks_;
  Dim3 size_;
  Dim3 gpuDim_;
  Dim3 rankDim_;
  Dim3 domSize_;

public:
  int get_rank(const Dim3 &idx) const override {
    return idx.x + idx.y * rankDim_.x + idx.z * rankDim_.y * rankDim_.x;
  }

  int get_gpu(const Dim3 &idx) const override {
    return idx.x + idx.y * gpuDim_.x + idx.z * gpuDim_.y * gpuDim_.x;
  }

  Dim3 gpu_idx(int gpu) const override {
    assert(gpu < gpus_);
    Dim3 ret;
    ret.x = gpu % gpuDim_.x;
    gpu /= gpuDim_.x;
    ret.y = gpu % gpuDim_.y;
    gpu /= gpuDim_.y;
    ret.z = gpu;
    return ret;
  }
  Dim3 rank_idx(int rank) const override {
    assert(rank < ranks_);
    Dim3 ret;
    ret.x = rank % rankDim_.x;
    rank /= rankDim_.x;
    ret.y = rank % rankDim_.y;
    rank /= rankDim_.y;
    ret.z = rank;
    return ret;
  }

  Dim3 gpu_dim() const override { return gpuDim_; }
  Dim3 rank_dim() const override { return rankDim_; }

  Dim3 local_domain_size(const Dim3 &rankIdx,
                         const Dim3 &gpuIdx) const override {
    return domSize_;
  }

  PFP(const Dim3 &domSize, const int ranks, const int gpus)
      : size_(domSize), gpuDim_(1, 1, 1), rankDim_(1, 1, 1), ranks_(ranks),
        gpus_(gpus) {

    domSize_ = size_;
    auto rankFactors = prime_factors(ranks_);

    // split repeatedly by prime factors of the number of MPI ranks to establish
    // the 3D partition among ranks
    for (size_t amt : rankFactors) {
      if (amt < 2) {
        continue;
      }
      double curCubeness = cubeness(domSize_.x, domSize_.y, domSize_.z);
      double xSplitCubeness =
          cubeness(div_ceil(domSize_.x, amt), domSize_.y, domSize_.z);
      double ySplitCubeness =
          cubeness(domSize_.x, div_ceil(domSize_.y, amt), domSize_.z);
      double zSplitCubeness =
          cubeness(domSize_.x, domSize_.y, div_ceil(domSize_.z, amt));

      if (xSplitCubeness >=
          std::max(ySplitCubeness, zSplitCubeness)) { // split in x
        domSize_.x = div_ceil(domSize_.x, amt);
        rankDim_.x *= amt;
      } else if (ySplitCubeness >=
                 std::max(xSplitCubeness, ySplitCubeness)) { // split in y
        domSize_.y = div_ceil(domSize_.y, amt);
        rankDim_.y *= amt;
      } else { // split in z
        domSize_.z = div_ceil(domSize_.z, amt);
        rankDim_.z *= amt;
      }
    }

    // split again along GPUs
    auto gpuFactors = prime_factors(gpus_);

    for (size_t amt : gpuFactors) {
      std::cerr << amt << "\n";
      if (amt < 2) {
        continue;
      }
      double curCubeness = cubeness(domSize_.x, domSize_.y, domSize_.z);
      double xSplitCubeness =
          cubeness(div_ceil(domSize_.x, amt), domSize_.y, domSize_.z);
      double ySplitCubeness =
          cubeness(domSize_.x, div_ceil(domSize_.y, amt), domSize_.z);
      double zSplitCubeness =
          cubeness(domSize_.x, domSize_.y, div_ceil(domSize_.z, amt));

      if (xSplitCubeness >=
          std::max(ySplitCubeness, zSplitCubeness)) { // split in x
        domSize_.x = div_ceil(domSize_.x, amt);
        gpuDim_.x *= amt;
      } else if (ySplitCubeness >=
                 std::max(xSplitCubeness, ySplitCubeness)) { // split in y
        domSize_.y = div_ceil(domSize_.y, amt);
        gpuDim_.y *= amt;
      } else { // split in z
        domSize_.z = div_ceil(domSize_.z, amt);
        gpuDim_.z *= amt;
      }
    }
  }

  // https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
  static std::vector<size_t> prime_factors(size_t n) {
    std::vector<size_t> result;

    while (n % 2 == 0) {
      result.push_back(2);
      n = n / 2;
    }
    for (int i = 3; i <= sqrt(n); i = i + 2) {
      // While i divides n, print i and divide n
      while (n % i == 0) {
        result.push_back(i);
        n = n / i;
      }
    }
    if (n > 2)
      result.push_back(n);

    std::sort(result.begin(), result.end(),
              [](size_t a, size_t b) { return b < a; });

    return result;
  }

  static double cubeness(double x, double y, double z) {
    double smallest = std::min(x, std::min(y, z));
    double largest = std::max(x, std::max(y, z));
    return smallest / largest;
  }

  static size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }
};