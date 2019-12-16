#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "dim3.cuh"

class Partition {
public:
  // get the MPI rank for a particular index in the rank space
  virtual int get_rank(const Dim3 &idx) const = 0;

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

  int get_rank(const Dim3 &idx) const override {
    return idx.x + idx.y * rankDim_.x + idx.z * rankDim_.y * rankDim_.x;
  }

  int get_gpu(const Dim3 &idx) const override {
    return idx.x + idx.y * gpuDim_.x + idx.z * gpuDim_.y * gpuDim_.x;
  }

  Dim3 gpu_idx(int gpu) const override {
    assert(gpu < gpus_);
    Dim3 gpuIdx;
    gpuIdx.x = gpu % gpuDim_.x;
    gpu /= gpuDim_.x;
    gpuIdx.y = gpu % gpuDim_.y;
    gpu /= gpuDim_.y;
    gpuIdx.z = gpu;
  }
  Dim3 rank_idx(int rank) const override {
    assert(rank < ranks_);
    Dim3 rankIdx;
    rankIdx.x = rank % rankDim_.x;
    rank /= rankDim_.x;
    rankIdx.y = rank % rankDim_.y;
    rank /= rankDim_.y;
    rankIdx.z = rank;
    return rankIdx;
  }

  Dim3 gpu_dim() const override { return rankDim_; }
  Dim3 rank_dim() const override { return gpuDim_; }

  PFP(const Dim3 &domSize, const int ranks, const int gpus)
      : size_(domSize), gpuDim_(1, 1, 1), rankDim_(1, 1, 1), ranks_(ranks),
        gpus_(gpus) {
    auto rankFactors = prime_factors(ranks);

    auto splitSize = size_;

    for (size_t amt : rankFactors) {
      double curCubeness = cubeness(size_.x, size_.y, size_.z);
      double xSplitCubeness =
          cubeness(div_ceil(splitSize.x, amt), splitSize.y, splitSize.z);
      double ySplitCubeness =
          cubeness(splitSize.x, div_ceil(splitSize.y, amt), splitSize.z);
      double zSplitCubeness =
          cubeness(splitSize.x, splitSize.y, div_ceil(splitSize.z, amt));

      if (xSplitCubeness > max(ySplitCubeness, zSplitCubeness)) { // split in x
        splitSize.x = div_ceil(splitSize.x, amt);
        rankDim_.x *= amt;
      } else if (ySplitCubeness >
                 max(xSplitCubeness, ySplitCubeness)) { // split in y
        splitSize.y = div_ceil(splitSize.y, amt);
        rankDim_.y *= amt;
      } else { // split in z
        splitSize.z = div_ceil(splitSize.z, amt);
        rankDim_.z *= amt;
      }
    }

    // split biggest dimension across GPUs
    if (splitSize.x > max(splitSize.y, splitSize.z)) {
      gpuDim_.x = gpus;
      splitSize.x /= gpus;
    } else if (splitSize.y > max(splitSize.x, splitSize.x)) {
      gpuDim_.y = gpus;
      splitSize.y /= gpus;
    } else {
      gpuDim_.z = gpus;
      splitSize.z /= gpus;
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
    double smallest = min(x, min(y, z));
    double largest = max(x, max(y, z));
    return smallest / largest;
  }

  static size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }
};