#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

#include <mpi.h>

#include "cuda_runtime.hpp"

class Dim3 {
public:
  size_t x;
  size_t y;
  size_t z;

public:
  Dim3(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
};

class Domain;

class Radius {
  friend class Domain;

private:
  size_t id_;

public:
  Radius(size_t i) : id_(i) {}
};

template <typename T> class DataHandle {
  friend class Domain;
  size_t id_;

public:
  DataHandle(size_t i) : id_(i) {}
};

// https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
std::vector<size_t> prime_factors(size_t n) {
  std::vector<size_t> result;

  while (n % 2 == 0) {
    result.push_back(2);
    n = n / 2;
  }

  // n must be odd at this point. So we can skip
  // one element (Note i = i + 2)
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    // While i divides n, print i and divide n
    while (n % i == 0) {
      result.push_back(i);
      n = n / i;
    }
  }

  // This condition is to handle the case when n
  // is a prime number greater than 2
  if (n > 2)
    result.push_back(n);

  return result;
}

double cubeness(double x, double y, double z) {
  double smallest = min(x, min(y, z));
  double largest = max(x, max(y, z));
  return smallest / largest;
}

size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }

// information about a neighboring domain
class RemoteDomain {
  // MPI rank and GPU for the remote domain
  int rank_;
  int device_;
};

class LocalDomain {
private:

  // my location in the distributed domain
  size_t xIdx_;
  size_t yIdx_;
  size_t zIdx_;

  // my local data size
  size_t xSz_;
  size_t ySz_;
  size_t zSz_;

  // my origin's global location
  size_t xOff_;
  size_t yOff_;
  size_t zOff_;

  std::vector<void *> dataPtrs_;
  std::vector<size_t> dataElemSize_;

public:

  // send in positive x direction
  void send_px() {

    // get my rank
    int rank;

    // get the neighbor's rank and GPU
    int nbrRank;

    if (rank == nbrRank) { // on-rank

    } else { // off-rank

    }
  }

  // recv from positive x direction
  void recv_px() {

  }

  void wait_all() {

  }

};



class Domain {
private:


  typedef size_t (*RankFn)(size_t x, size_t y, size_t z);
  typedef size_t (*GPUFn)(size_t x, size_t y, size_t z);

  // get the rank of a particular index
  RankFn rankFn_;
  // get the gpu of a particular index
  GPUFn gpuFn_;

  size_t x_;
  size_t y_;
  size_t z_;

  int rank_;
  int size_;
  int deviceCount_;

  std::vector<Dim3> radii_;

  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;

public:
  Domain(size_t x, size_t y, size_t z) : x_(x), y_(y), z_(z) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount_));
  }

  ~Domain() {}

  Radius add_radius(size_t x, size_t y, size_t z) {
    size_t idx = radii_.size();
    radii_.push_back(Dim3(x, y, z));
    return Radius(idx);
  }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
  }

  template <typename T> T *get_data(const DataHandle<T> handle) {
    assert(dataElemSize_.size() > handle.id_);
    assert(dataPtrs_.size() > handle.id_);
    void *ptr = dataPtrs_[handle.id_];
    assert(sizeof(T) == dataElemSize_[handle.id_]);
    return static_cast<T *>(dataPtrs_[handle.id_]);
  }

  void realize() {

    // find the max radius
    assert(!radii_.empty() && "should have added a stencil radius");
    Dim3 maxRadius = radii_[0];
    for (auto &r : radii_) {
      maxRadius.x = std::max(r.x, maxRadius.x);
      maxRadius.y = std::max(r.y, maxRadius.y);
      maxRadius.z = std::max(r.z, maxRadius.z);
    }

    // recursively split region among MPI ranks to make it ~cubical
    Dim3 splitSize(x_, y_, z_);
    Dim3 rankDim(1, 1, 1); // how MPI ranks are split among dimensions
    Dim3 gpuDim(1, 1, 1); // how GPUs are split among dimensions
    auto factors = prime_factors(size_);
    std::sort(factors.begin(), factors.end(),
              [](size_t a, size_t b) { return b < a; });
    for (size_t amt : factors) {

      if (rank_ == 0) {
        printf("split by %lu\n", amt);
      }

      double curCubeness = cubeness(x_, y_, z_);
      double xSplitCubeness =
          cubeness(div_ceil(splitSize.x, amt), splitSize.y, splitSize.z);
      double ySplitCubeness =
          cubeness(splitSize.x, div_ceil(splitSize.y, amt), splitSize.z);
      double zSplitCubeness =
          cubeness(splitSize.x, splitSize.y, div_ceil(splitSize.z, amt));

      if (rank_ == 0) {
        printf("%lu %lu %lu %f\n", x_, y_, z_, curCubeness);
      }

      if (xSplitCubeness > max(ySplitCubeness, zSplitCubeness)) { // split in x
        if (rank_ == 0) {
          printf("x split: %f\n", xSplitCubeness);
        }
        splitSize.x = div_ceil(splitSize.x, amt);
        rankDim.x *= amt;
      } else if (ySplitCubeness >
                 max(xSplitCubeness, ySplitCubeness)) { // split in y
        if (rank_ == 0) {
          printf("y split: %f\n", ySplitCubeness);
        }
        splitSize.y = div_ceil(splitSize.y, amt);
        rankDim.y *= amt;
      } else { // split in z
        if (rank_ == 0) {
          printf("z split: %f\n", zSplitCubeness);
        }
        splitSize.z = div_ceil(splitSize.z, amt);
        rankDim.z *= amt;
      }
    }



    // split biggest dimension across GPUs
    if (splitSize.x > max(splitSize.y, splitSize.z)) {
      gpuDim.x = deviceCount_;
    } else if (splitSize.y > max(splitSize.x, splitSize.x)) {
      gpuDim.y = deviceCount_;
    } else {
      gpuDim.z = deviceCount_;
    }

    if (rank_ == 0) {
      printf("%lux%lux%lu of %lux%lux%lux (gpus %lux%lux%lu)\n", splitSize.x, splitSize.y,
             splitSize.z, rankDim.x, rankDim.y, rankDim.z, gpuDim.x, gpuDim.y, gpuDim.z);
    }

    // create local domains

    // allocate each data region
    for (size_t i = 0; i < dataElemSize_.size(); ++i) {
      size_t elemSz = dataElemSize_[i];

      size_t elemBytes = ((x_ + 2 * maxRadius.x) * (y_ + 2 * maxRadius.y) *
                          (z_ + 2 * maxRadius.z)) *
                         elemSz;
      std::cerr << "Allocate " << elemBytes << "\n";
      void *p = new char[elemBytes];
      assert(uintptr_t(p) % elemSz == 0 && "allocation should be aligned");
      dataPtrs_.push_back(p);
    }

    // create a communication plan for the maximum radius

    //
  }

  /*!
  start a halo exchange and return.
  Call sync() to block until exchange is done.
  */
  void exchange_async(const Radius &r) {
    assert(r.id_ < radii_.size() && "invalid radius handle");

    for (auto &d : domains_) {
      // send to / recv from +x
      d.send_px();
      d.recv_px();

    }


  }

  /*!
  wait for async exchange
  */
  void sync() {

    // wait for all domain communication
    for (auto &d : domains_) {
      d.wait_all();
    }

    // wait for everyone else's exchanges to be done
    MPI_Barrier(MPI_COMM_WORLD);

  }

  /*!
  do a halo exchange and return
  */
  void exchange(const Radius &r) {
    exchange_async(r);
    sync();
  }
};