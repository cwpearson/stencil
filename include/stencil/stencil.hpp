#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <set>
#include <vector>

#include <mpi.h>

#include <nvml.h>

#include "cuda_runtime.hpp"

#include "stencil/dim3.hpp"
#include "stencil/direction_map.hpp"
#include "stencil/gpu_topo.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/nvml.hpp"
#include "stencil/tx.hpp"

// https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
std::vector<size_t> prime_factors(size_t n) {
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

  return result;
}

/*!
  min element / max element
*/
double cubeness(double x, double y, double z) {
  double smallest = min(x, min(y, z));
  double largest = max(x, max(y, z));
  return smallest / largest;
}

/*! \brief ceil(n/d)
 */
size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }

class DistributedDomain {
private:
  Dim3 size_;

  int rank_;
  int worldSize_;

  // the GPUs this MPI rank will use
  std::vector<int> gpus_;

  // the stencil radius
  size_t radius_;

  // the dimension of the domain in MPI ranks and GPUs
  Dim3 rankDim_;
  Dim3 gpuDim_;

  // typically one per GPU
  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;
  // the index of the domain in the distributed domain
  std::vector<Dim3> domainIdx_;

  // senders/recvers for each direction in each domain
  // domainDirSenders[domainIdx].at(dir)
  // a sender/recver pair is associated with the send direction
  std::vector<DirectionMap<HaloSender *>> domainDirSender_;
  std::vector<DirectionMap<HaloRecver *>> domainDirRecver_;

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

  // MPI ranks co-located with me
  std::set<int64_t> colocated_;

  std::vector<std::vector<bool>> peerAccess_; //<! which GPUs have peer access

public:
  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), rankDim_(1, 1, 1), gpuDim_(1, 1, 1) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));

    // create a communicator for ranks on the same node
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    int shmrank, shmsize;
    MPI_Comm_rank(shmcomm, &shmrank);
    MPI_Comm_size(shmcomm, &shmsize);
    printf("DistributedDomain::ctor(): shmcomm rank %d/%d\n", shmrank, shmsize);

    // if fewer ranks than GPUs, round-robin GPUs to ranks
    if (shmsize <= deviceCount) {
      for (int gpu = 0; gpu < deviceCount; ++gpu) {
        if (gpu % shmsize == shmrank) {
          gpus_.push_back(gpu);
        }
      }
    } else { // if more ranks, share gpus among ranks
      gpus_.push_back(shmrank % deviceCount);
    }

    for (const auto gpu : gpus_) {
      printf("rank %d/%d local=%d using gpu %d\n", rank_, worldSize_, shmrank,
             gpu);
    }

    // Give every rank a list of co-located ranks
    std::vector<int> colocated(shmsize);
    MPI_Allgather(&rank_, 1, MPI_INT, colocated.data(), 1, MPI_INT, shmcomm);
    for (auto &r : colocated) {
      colocated_.insert(r);
    }
    assert(colocated_.count(rank_) == 1 && "should be colocated with self");
    printf("DistributedDomain::ctor(): rank %d colocated with %lu ranks\n",
           rank_, colocated_.size() - 1);

    // Try to enable peer access between all GPUs
    peerAccess_ = std::vector<std::vector<bool>>(
        gpus_.size(), std::vector<bool>(gpus_.size(), false));

    for (auto &src : gpus_) {
      for (auto &dst : gpus_) {
        CUDA_RUNTIME(cudaSetDevice(src))
        cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0 /*flags*/);
        if (cudaSuccess == err || cudaErrorPeerAccessAlreadyEnabled == err) {
          peerAccess_[src][dst] = true;
          std::cout << src << " -> " << dst << "peer access\n";
        } else if (cudaErrorInvalidDevice) {
          peerAccess_[src][dst] = false;
        } else {
          assert(0);
        }
      }
    }

    std::cerr << "gpu distance matrix: \n";
    Mat2D dist = get_gpu_distance_matrix();
    if (0 == rank_) {
      for (auto &r : dist) {
        for (auto &c : r) {
          std::cerr << c << " ";
        }
        std::cerr << "\n";
      }
    }
  }

  ~DistributedDomain(){
#warning dtor is a no-op
  }

  std::vector<LocalDomain> &domains() {
    return domains_;
  }

  void set_radius(size_t r) { radius_ = r; }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
  }

  size_t get_rank(const Dim3 idx) {
    assert(idx.x >= 0);
    assert(idx.y >= 0);
    assert(idx.z >= 0);
    assert(idx.x < rankDim_.x * gpuDim_.x);
    return (idx.x / gpuDim_.x) + (idx.y / gpuDim_.y) * rankDim_.x +
           (idx.z / gpuDim_.z) * rankDim_.y * rankDim_.x;
  }

  size_t get_logical_gpu(const Dim3 idx) {
    assert(idx.x >= 0);
    assert(idx.y >= 0);
    assert(idx.z >= 0);
    return (idx.x % gpuDim_.x) + (idx.y % gpuDim_.y) * gpuDim_.x +
           (idx.z % gpuDim_.z) * gpuDim_.y * gpuDim_.x;
  }

  Dim3 get_idx(size_t rank, size_t gpu) {
    assert(0);
#warning get_idx unimplemented
    return Dim3();
  }

  void realize(bool useUnified = false) {

    // recursively split region among MPI ranks to make it ~cubical
    Dim3 splitSize = size_;
    auto factors = prime_factors(worldSize_);
    std::sort(factors.begin(), factors.end(),
              [](size_t a, size_t b) { return b < a; });
    for (size_t amt : factors) {

      if (rank_ == 0) {
        printf("split by %lu\n", amt);
      }

      double curCubeness = cubeness(size_.x, size_.y, size_.z);
      double xSplitCubeness =
          cubeness(div_ceil(splitSize.x, amt), splitSize.y, splitSize.z);
      double ySplitCubeness =
          cubeness(splitSize.x, div_ceil(splitSize.y, amt), splitSize.z);
      double zSplitCubeness =
          cubeness(splitSize.x, splitSize.y, div_ceil(splitSize.z, amt));

      if (rank_ == 0) {
        printf("%lu %lu %lu %f\n", size_.x, size_.y, size_.z, curCubeness);
      }

      if (xSplitCubeness > max(ySplitCubeness, zSplitCubeness)) { // split in x
        if (rank_ == 0) {
          printf("x split: %f\n", xSplitCubeness);
        }
        splitSize.x = div_ceil(splitSize.x, amt);
        rankDim_.x *= amt;
      } else if (ySplitCubeness >
                 max(xSplitCubeness, ySplitCubeness)) { // split in y
        if (rank_ == 0) {
          printf("y split: %f\n", ySplitCubeness);
        }
        splitSize.y = div_ceil(splitSize.y, amt);
        rankDim_.y *= amt;
      } else { // split in z
        if (rank_ == 0) {
          printf("z split: %f\n", zSplitCubeness);
        }
        splitSize.z = div_ceil(splitSize.z, amt);
        rankDim_.z *= amt;
      }
    }

    // split biggest dimension across GPUs
    if (splitSize.x > max(splitSize.y, splitSize.z)) {
      gpuDim_.x = gpus_.size();
      splitSize.x /= gpus_.size();
    } else if (splitSize.y > max(splitSize.x, splitSize.x)) {
      gpuDim_.y = gpus_.size();
      splitSize.y /= gpus_.size();
    } else {
      gpuDim_.z = gpus_.size();
      splitSize.z /= gpus_.size();
    }

    if (rank_ == 0) {
      printf("%lux%lux%lu of %lux%lux%lux (gpus %lux%lux%lu)\n", splitSize.x,
             splitSize.y, splitSize.z, rankDim_.x, rankDim_.y, rankDim_.z,
             gpuDim_.x, gpuDim_.y, gpuDim_.z);
    }

    // create local domains
    for (int i = 0; i < gpus_.size(); i++) {

      auto gpu = gpus_[i];

      Dim3 rankIdx;
      Dim3 logicalGpuIdx;

      auto rank = rank_;
      rankIdx.x = rank % rankDim_.x;
      rank /= rankDim_.x;
      rankIdx.y = rank % rankDim_.y;
      rank /= rankDim_.y;
      rankIdx.z = rank;

      logicalGpuIdx.x = i % gpuDim_.x;
      i /= gpuDim_.x;
      logicalGpuIdx.y = i % gpuDim_.y;
      i /= gpuDim_.y;
      logicalGpuIdx.z = i;

      LocalDomain ld(splitSize, gpu);
      ld.radius_ = radius_;
      for (size_t dataIdx = 0; dataIdx < dataElemSize_.size(); ++dataIdx) {
        ld.add_data(dataElemSize_[dataIdx]);
      }

      domains_.push_back(ld);
      Dim3 idx = rankIdx * gpuDim_ + logicalGpuIdx;
      printf("rank,gpu=%d,%d(gpu actual idx=%d) => idx %ld %ld %ld\n", rank_, i,
             gpu, idx.x, idx.y, idx.z);
      domainIdx_.push_back(idx);
    }

    // realize local domains
    for (auto &d : domains_) {
      if (useUnified)
        d.realize_unified();
      else
        d.realize();
      printf("DistributedDomain.realize(): finished creating LocalDomain\n");
    }

    // initialize null senders and recvers for each domain
    domainDirSender_.resize(gpus_.size());
    domainDirRecver_.resize(gpus_.size());
    for (size_t domainIdx = 0; domainIdx < domains_.size(); ++domainIdx) {
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            domainDirSender_[domainIdx].at(x, y, z) = nullptr;
            domainDirRecver_[domainIdx].at(x, y, z) = nullptr;
          }
        }
      }
    }

    // build senders and recvers for each domain
    for (size_t di = 0; di < domains_.size(); ++di) {
      assert(domains_.size() == domainIdx_.size());

      auto &d = domains_[di];
      const Dim3 myIdx = domainIdx_[di];
      const int myRank = rank_;
      const int myGPU = d.gpu();
      const int logicalMyGPU = get_logical_gpu(myIdx);
      assert(myRank == get_rank(myIdx));
      assert(myGPU == gpus_[logicalMyGPU]);

      auto &dirSender = domainDirSender_[di];
      auto &dirRecver = domainDirRecver_[di];

      // send/recv pairs for faces
      for (const auto xDir : {-1}) {
        for (const auto yDir : {1}) {
          for (const auto zDir : {1}) {
            Dim3 dirVec(xDir, yDir, zDir);
            if (dirVec == Dim3(0, 0, 0)) {
              continue; // don't send in no direction
            }

            // who i am sending to for this dirVec
            Dim3 dstIdx = (myIdx + dirVec).wrap(rankDim_ * gpuDim_);

            // who is sending to me for this dirVec
            Dim3 srcIdx = (myIdx - dirVec).wrap(rankDim_ * gpuDim_);

            int myRank = rank_;
            int myGPU = d.gpu();

            int logicalMyGPU = get_logical_gpu(myIdx);
            int logicalSrcGPU = get_logical_gpu(srcIdx);
            int logicalDstGPU = get_logical_gpu(dstIdx);

            assert(myRank == get_rank(myIdx));
            assert(myGPU == gpus_[logicalMyGPU]);
            int srcRank = get_rank(srcIdx);
            int srcGPU = gpus_[logicalSrcGPU];
            int dstRank = get_rank(dstIdx);
            int dstGPU = gpus_[logicalDstGPU];

            std::cerr << myIdx << " -> " << dstIdx << " dirVec=" << dirVec
                      << " r" << myRank << ",g" << myGPU << " -> r" << dstRank
                      << ",g" << dstGPU << "\n";

            // determine how to send face in that direction
            HaloSender *sender = nullptr;

            if (myRank == dstRank) { // both domains ownned by this rank
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " send same rank\n";
              sender = new RegionSender<AnySender>(d, myRank, myGPU, dstRank,
                                                   dstGPU, dirVec);
            } else if (colocated_.count(dstRank)) { // both domains on this node
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " send colocated\n";
              sender = new RegionSender<AnySender>(d, myRank, myGPU, dstRank,
                                                   dstGPU, dirVec);
            } else { // domains on different nodes
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " send diff nodes\n";
              sender = new RegionSender<AnySender>(d, myRank, myGPU, dstRank,
                                                   dstGPU, dirVec);
            }

            std::cerr << myIdx << " <- " << srcIdx << " dirVec=" << dirVec
                      << " r" << myRank << ",g" << myGPU << " <- r" << srcRank
                      << ",g" << srcGPU << "\n";

            // determine how to receive a face from that direction
            HaloRecver *recver = nullptr;
            if (myRank == srcRank) { // both domains onwned by this rank
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " recv same rank\n";
              recver = new RegionRecver<AnyRecver>(d, srcRank, srcGPU, myRank,
                                                   myGPU, dirVec);
            } else if (colocated_.count(srcRank)) { // both domains on this node
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " recv colocated\n";
              recver = new RegionRecver<AnyRecver>(d, srcRank, srcGPU, myRank,
                                                   myGPU, dirVec);
            } else { // domains on different nodes
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " recv diff nodes\n";
              recver = new RegionRecver<AnyRecver>(d, srcRank, srcGPU, myRank,
                                                   myGPU, dirVec);
            }

            assert(sender != nullptr);
            assert(recver != nullptr);

            sender->allocate();
            recver->allocate();
            dirSender.at_dir(dirVec.x, dirVec.y, dirVec.z) = sender;
            dirRecver.at_dir(dirVec.x, dirVec.y, dirVec.z) = recver;
          }
        }
      }
    }
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {
    // issue all sends
    for (auto &dirSenders : domainDirSender_) {
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            if (auto sender = dirSenders.at(x, y, z)) {
              sender->send();
            }
          }
        }
      }
    }

    // issue all recvs
    for (auto &dirRecvers : domainDirRecver_) {
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            if (auto recver = dirRecvers.at(x, y, z)) {
              recver->recv();
            }
          }
        }
      }
    }

    // wait for all sends and recvs
    for (size_t domainIdx = 0; domainIdx < domains_.size(); ++domainIdx) {
      auto &dirSender = domainDirSender_[domainIdx];
      auto &dirRecver = domainDirRecver_[domainIdx];
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            if (auto recver = dirRecver.at(x, y, z)) {
              recver->wait();
            }
            if (auto sender = dirSender.at(x, y, z)) {
              sender->wait();
            }
          }
        }
      }
    }

    // wait for all ranks to be done
    MPI_Barrier(MPI_COMM_WORLD);
  }
};
