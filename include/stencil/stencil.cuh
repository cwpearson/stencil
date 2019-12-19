#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <set>
#include <vector>

#include <mpi.h>

#include <nvml.h>

#include "cuda_runtime.hpp"

#include "stencil/dim3.cuh"
#include "stencil/local_domain.cuh"
#include "stencil/nvml.hpp"
#include "stencil/tx.cuh"

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
  std::vector<Dim3> indices_;

  // Senders / receivers for each face
  std::vector<HaloSender *> pzSenders_; // how to send +z face
  std::vector<HaloRecver *> pzRecvers_; // how to recv +z face
  std::vector<HaloSender *> mzSenders_; // how to send -z face
  std::vector<HaloRecver *> mzRecvers_;
  std::vector<HaloSender *> pySenders_;
  std::vector<HaloRecver *> pyRecvers_;
  std::vector<HaloSender *> mySenders_;
  std::vector<HaloRecver *> myRecvers_;
  std::vector<HaloSender *> pxSenders_;
  std::vector<HaloRecver *> pxRecvers_;
  std::vector<HaloSender *> mxSenders_;
  std::vector<HaloRecver *> mxRecvers_;

  // All senders:
  // x * y * z
  // 0: not present
  // 1: positive
  // 2: negative
  // a face has not-present in the other two dimensions
  // an edge has not-present in one dimension
  // a corner has all dimensions present
  std::vector<HaloSender *> senders[3][3][3];
  std::vector<HaloRecver *> recvers[3][3][3];

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

  std::set<int64_t> colocated_; //<! colocated MPI ranks

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

    using Mat2D = std::vector<std::vector<double>>;

    int devCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&devCount));
    Mat2D dist(devCount, std::vector<double>(devCount));

    // build a distance matrix for GPUs
    for (int src = 0; src < devCount; ++src) {
      for (int dst = 0; dst < devCount; ++dst) {
        if (src == dst) {
          dist[src][dst] = 0;
        }
        std::cerr << src << " " << dst;
        nvmlDevice_t srcDev, dstDev;
        NVML(nvmlDeviceGetHandleByIndex(src, &srcDev));
        NVML(nvmlDeviceGetHandleByIndex(dst, &dstDev));
        nvmlGpuTopologyLevel_t pathInfo;
        NVML(nvmlDeviceGetTopologyCommonAncestor(srcDev, dstDev, &pathInfo));

        switch (pathInfo) {
        case NVML_TOPOLOGY_INTERNAL: {
          dist[src][dst] = 1;
          break;
        }
        case NVML_TOPOLOGY_SINGLE: {
          dist[src][dst] = 2;
          break;
        }
        case NVML_TOPOLOGY_MULTIPLE: {
          dist[src][dst] = 3;
          break;
        }
        case NVML_TOPOLOGY_HOSTBRIDGE: {
          dist[src][dst] = 4;
          break;
        }
        case NVML_TOPOLOGY_NODE: {
          dist[src][dst] = 5;
          break;
        }
        case NVML_TOPOLOGY_SYSTEM: {
          dist[src][dst] = 6;
          break;
        }
        default:
          assert(0);
        }
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
      indices_.push_back(idx);
    }

    // realize local domains
    for (auto &d : domains_) {
      if (useUnified)
        d.realize_unified();
      else
        d.realize();
      printf("DistributedDomain.realize(): finished creating LocalDomain\n");
    }

    for (size_t di = 0; di < domains_.size(); ++di) {
      assert(domains_.size() == indices_.size());

      auto &d = domains_[di];
      const Dim3 myIdx = indices_[di];
      const int myRank = rank_;
      const int myGPU = d.gpu();
      const int logicalMyGPU = get_logical_gpu(myIdx);
      assert(myRank == get_rank(myIdx));
      assert(myGPU == gpus_[logicalMyGPU]);

      // consider face neighbor in dimension and direction
      for (const auto dim : {0, 1, 2}) { // dimensions (x,y,z)
        for (const auto dir : {-1, 1}) { // direction (neg, pos)

          Dim3 nbrIdx = myIdx;
          nbrIdx[dim] += dir;
          nbrIdx = nbrIdx.wrap(rankDim_ * gpuDim_);

          int myRank = rank_;
          int myGPU = d.gpu();

          int logicalMyGPU = get_logical_gpu(myIdx);
          int logicalNbrGPU = get_logical_gpu(nbrIdx);

          assert(myRank == get_rank(myIdx));
          assert(myGPU == gpus_[logicalMyGPU]);
          int nbrRank = get_rank(nbrIdx);
          int nbrGPU = gpus_[logicalNbrGPU]; // FIXME: assuming everyone has the
                                             // same GPU layout

          std::cout << myIdx << " -> " << nbrIdx << " dim=" << dim
                    << " dir=" << dir << " r" << myRank << ",g" << myGPU
                    << " -> r" << nbrRank << ",g" << nbrGPU << "\n";

          // determine how to send face in that direction
          HaloSender *sender = nullptr;
          if (myRank == nbrRank) { // both domains onwned by this rank
            printf(
                "DistributedDomain.realize(): dim=%d dir=%d send same rank\n",
                dim, dir);
            sender = new FaceSender<AnySender>(d, myRank, myGPU, nbrRank,
                                               nbrGPU, dim, dir > 0 /*pos*/);
          } else if (colocated_.count(nbrRank)) { // both domains on this node
            printf(
                "DistributedDomain.realize(): dim=%d dir=%d  send colocated\n",
                dim, dir);
            sender = new FaceSender<AnySender>(d, myRank, myGPU, nbrRank,
                                               nbrGPU, dim, dir > 0 /*pos*/);
          } else { // domains on different nodes
            printf("DistributedDomain.realize(): send different nodes\n");
            sender = new FaceSender<AnySender>(d, myRank, myGPU, nbrRank,
                                               nbrGPU, dim, dir > 0 /*pos*/);
          }

          std::cout << myIdx << " <- " << nbrIdx << " dim=" << dim
                    << " dir=" << dir << " r" << myRank << ",g" << myGPU
                    << " <- r" << nbrRank << ",g" << nbrGPU << "\n";

          // determine how to receive a face from that direction
          HaloRecver *recver = nullptr;
          if (myRank == nbrRank) { // both domains onwned by this rank
            printf(
                "DistributedDomain.realize(): dim=%d dir=%d recv same rank\n",
                dim, dir);
            recver = new FaceRecver<AnyRecver>(d, nbrRank, nbrGPU, myRank,
                                               myGPU, dim, dir > 0 /*pos*/);
          } else if (colocated_.count(nbrRank)) { // both domains on this node
            printf(
                "DistributedDomain.realize(): dim=%d dir=%d recv colocated\n",
                dim, dir);
            recver = new FaceRecver<AnyRecver>(d, nbrRank, nbrGPU, myRank,
                                               myGPU, dim, dir > 0 /*pos*/);
          } else { // domains on different nodes
            printf("DistributedDomain.realize(): recv different nodes\n");
            recver = new FaceRecver<AnyRecver>(d, nbrRank, nbrGPU, myRank,
                                               myGPU, dim, dir > 0 /*pos*/);
          }

          assert(sender != nullptr);
          assert(recver != nullptr);
          if (1 == dir) {
            if (0 == dim) {
              pxSenders_.push_back(sender);
              pxSenders_.back()->allocate();
              pxRecvers_.push_back(recver);
              pxRecvers_.back()->allocate();
            } else if (1 == dim) {
              pySenders_.push_back(sender);
              pySenders_.back()->allocate();
              pyRecvers_.push_back(recver);
              pyRecvers_.back()->allocate();
            } else if (2 == dim) {
              pzSenders_.push_back(sender);
              pzSenders_.back()->allocate();
              pzRecvers_.push_back(recver);
              pzRecvers_.back()->allocate();
            } else {
              assert(0 && "only 3D supported");
            }
          } else if (-1 == dir) {
            if (0 == dim) {
              mxSenders_.push_back(sender);
              mxSenders_.back()->allocate();
              mxRecvers_.push_back(recver);
              mxRecvers_.back()->allocate();
            } else if (1 == dim) {
              mySenders_.push_back(sender);
              mySenders_.back()->allocate();
              myRecvers_.push_back(recver);
              myRecvers_.back()->allocate();
            } else if (2 == dim) {
              mzSenders_.push_back(sender);
              mzSenders_.back()->allocate();
              mzRecvers_.push_back(recver);
              mzRecvers_.back()->allocate();
            } else {
              assert(0 && "only 3D supported");
            }
          } else {
            assert(0 && "unexpected direction");
          }
        }
      }

      // xy faces
      for (auto xDir : {-1, 1}) {
        for (auto yDir : {-1, 1}) {
          Dim3 nbrIdx = myIdx + Dim3(xDir, yDir, 0);
          nbrIdx = nbrIdx.wrap(rankDim_ * gpuDim_);
          const int logicalNbrGPU = get_logical_gpu(nbrIdx);
          const int nbrRank = get_rank(nbrIdx);
          const int nbrGPU = gpus_[logicalNbrGPU];

          HaloSender *sender = nullptr;
          HaloRecver *recver = nullptr;
#if 0
          sender = new EdgeSender<AnySender>(d, myRank, myGPU, nbrRank, nbrGPU,
                                         0 /*x*/, 1 /*y*/, xDim > 0 /*xpos*/,
                                         yDir > 0 /*ypos*/);
#endif
          recver = new EdgeRecver<AnyRecver>(d, nbrRank, nbrGPU, myRank, myGPU,
                                             0 /*x*/, 1 /*y*/, xDir > 0 /*pos*/,
                                             yDir > 0 /*pos*/);
        }
      }
      // xz faces
      // yz faces
    }
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {
    assert(pzSenders_.size() == domains_.size());
    assert(pySenders_.size() == domains_.size());
    assert(pxSenders_.size() == domains_.size());

    assert(pzRecvers_.size() == domains_.size());
    assert(pyRecvers_.size() == domains_.size());
    assert(pxRecvers_.size() == domains_.size());

    assert(mzSenders_.size() == domains_.size());
    assert(mySenders_.size() == domains_.size());
    assert(mxSenders_.size() == domains_.size());

    assert(mzRecvers_.size() == domains_.size());
    assert(myRecvers_.size() == domains_.size());
    assert(mxRecvers_.size() == domains_.size());

    // send +z

    for (size_t di = 0; di < domains_.size(); ++di) {
      std::cout << "DistributedDomain::exchange(): r" << rank_ << ": recv mz["
                << di << "]\n";
      mzRecvers_[di]->recv();
      std::cout << "DistributedDomain::exchange(): r" << rank_ << ": send pz["
                << di << "]\n";
      pzSenders_[di]->send();
    }
    for (size_t di = 0; di < domains_.size(); ++di) {
      mzRecvers_[di]->wait();
      pzSenders_[di]->wait();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // send +y
    for (size_t di = 0; di < domains_.size(); ++di) {
      myRecvers_[di]->recv();
      pySenders_[di]->send();
    }
    for (size_t di = 0; di < domains_.size(); ++di) {
      myRecvers_[di]->wait();
      pySenders_[di]->wait();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // send +x
    for (size_t di = 0; di < domains_.size(); ++di) {
      mxRecvers_[di]->recv();
      pxSenders_[di]->send();
    }
    for (size_t di = 0; di < domains_.size(); ++di) {
      mxRecvers_[di]->wait();
      pxSenders_[di]->wait();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
};