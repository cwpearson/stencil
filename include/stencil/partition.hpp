#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

#include "dim3.hpp"
#include "mat2d.hpp"

#include "gpu_topology.hpp"
#include "local_domain.cuh"
#include "mpi_topology.hpp"

class Partition {
public:
  // get the MPI rank for a domain
  virtual int get_rank(const Dim3 &idx) const = 0;

  // get the gpu for a domain
  virtual int get_gpu(const Dim3 &idx) const = 0;

  // the index of a GPU in the GPU space
  virtual Dim3 gpu_idx(int gpu) const = 0;

  // the index of the rank in the rank space
  virtual Dim3 rank_idx(int rank) const = 0;

  // the domain for rank and gpu
  // opposite of get_rank and get_gpu
  Dim3 dom_idx(int rank, int gpu) const {
    return rank_idx(rank) * gpu_dim() + gpu_idx(gpu);
  }

  // the extent of the gpu space
  virtual Dim3 gpu_dim() const = 0;

  // the extent of the rank space
  virtual Dim3 rank_dim() const = 0;

  // get the size of a domain
  virtual Dim3 local_domain_size(const Dim3 &domIdx) const = 0;
};

/*! Prime-factor Placer
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
    Dim3 rankIdx = idx / gpuDim_;
    return rankIdx.x + rankIdx.y * rankDim_.x +
           rankIdx.z * rankDim_.y * rankDim_.x;
  }

  int get_gpu(const Dim3 &idx) const override {
    Dim3 gpuIdx = idx % gpuDim_;
    return gpuIdx.x + gpuIdx.y * gpuDim_.x + gpuIdx.z * gpuDim_.y * gpuDim_.x;
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

  Dim3 local_domain_size(const Dim3 &domIdx) const override {

    Dim3 ret = domSize_;
    Dim3 rem = size_ % (rankDim_ * gpuDim_);

    if (rem.x != 0 && domIdx.x >= rem.x) {
      ret.x -= 1;
    }
    if (rem.y != 0 && domIdx.y >= rem.y) {
      ret.y -= 1;
    }
    if (rem.z != 0 && domIdx.z >= rem.z) {
      ret.z -= 1;
    }

    return ret;
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

    // split again for GPUs
    auto gpuFactors = prime_factors(gpus_);
    for (size_t amt : gpuFactors) {
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

  /*! \brief ceil(n/d)
   */
  static size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }
};

/*! average
 */
double avg(const std::vector<double> &x) {
  assert(x.size() >= 1);

  double acc = 0;
  for (auto &e : x) {
    acc += e;
  }
  return acc / x.size();
}

/*! corrected sample standard deviation
 */
double cssd(const std::vector<double> &x) {
  assert(x.size() >= 1);

  const double xBar = avg(x);

  double acc = 0;
  for (auto &xi : x) {
    acc += std::pow(xi - xBar, 2);
  }
  return std::sqrt(acc / x.size());
}

/*! sample correlation coefficient
 */
double scc(const std::vector<double> &x, const std::vector<double> &y) {
  assert(x.size() == y.size());

  const size_t n = x.size();
  const double xBar = avg(x);
  const double yBar = avg(y);

  double num = 0;
  for (size_t i = 0; i < n; ++i) {
    num += (x[i] - xBar) * (y[i] - yBar);
  }

  double den = (n - 1) * cssd(x) * cssd(y);

  if (0 == num && 0 == den) {
    return 1;
  } else {
    assert(0 != den);
    return num / den;
  }
}

double match(const Mat2D<double> &x, const Mat2D<double> &y) {

  assert(x.size() == y.size());
  assert(x.size() > 0);
  assert(x[0].size() == y[0].size());

  std::vector<double> rowCorrs(x.size()); // row correlations

  for (size_t ri = 0; ri < x.size(); ++ri) {
    double tmp = scc(x[ri], y[ri]);
    rowCorrs[ri] = tmp;
  }
#if 0
  std::cerr << "row corrs\n";
  for (auto &e : rowCorrs) {
    std::cerr << e << "\n";
  }
#endif

  // avg row correlations
  double acc = 0;
  for (auto &c : rowCorrs) {
    acc += c;
  }
  return acc / rowCorrs.size();
}

/*! return a copy of `m` with ret[map[i]][map[j]]=m[i][j]
 */
Mat2D<double> permute(const Mat2D<double> &m, std::vector<size_t> map) {
  assert(m.size() == map.size());
  for (auto &r : m) {
    assert(r.size() == map.size());
  }

  Mat2D<double> result(m.size());
  for (auto &v : result) {
    v.resize(m[0].size());
  }

  for (size_t r = 0; r < m.size(); ++r) {
    for (size_t c = 0; c < m.size(); ++c) {
      assert(r < map.size());
      assert(c < map.size());
      size_t nr = map[r];
      size_t nc = map[c];
      assert(nr < result.size());
      assert(nc < result[nr].size());
      result[r][c] = m[nr][nc];
    }
  }

  return result;
}

class NodeAwarePlacement {
private:
  int gpus_;
  int ranks_;
  Dim3 size_;
  Dim3 gpuDim_;
  Dim3 rankDim_;
  Dim3 domSize_;

  static size_t linearize(Dim3 idx, Dim3 dim) {
    assert(idx.x >= 0);
    assert(idx.y >= 0);
    assert(idx.z >= 0);
    assert(dim.x >= 0);
    assert(dim.y >= 0);
    assert(dim.z >= 0);
    assert(idx.x < dim.x);
    assert(idx.y < dim.y);
    assert(idx.z < dim.z);
    return idx.x + idx.y * dim.x + idx.z * dim.y * dim.x;
  }

  static Dim3 dimensionize(int64_t i, Dim3 dim) {
    assert(i < dim.flatten());
    assert(i >= 0);
    Dim3 ret;
    ret.x = i % dim.x;
    i /= dim.x;
    ret.y = i % dim.y;
    i /= dim.y;
    ret.z = i;
    return ret;
  }

  /* Return a number proportional to the bytes in a halo exchange, along
   * direction `dir` for a domain of size `sz` with radius `radius`
   */
  double comm_cost(Dim3 dir, const Dim3 sz, const size_t radius) {
    assert(dir.all_lt(2));
    assert(dir.all_gt(-2));
    double count = LocalDomain::halo_extent(dir, sz, radius).flatten();
    return count;
  }

  // map of domIdx to cuda device for all ranks
  std::map<Dim3, int> cudaId_;
  // convert domIdx to domId for all ranks
  std::map<Dim3, int> domId_;
  // convert rank and domain ID to domainIdx
  // since each domain is already attached to a GPU, and the domIdx controls the communication requirements,
  // there is no direct conversion between domainId and domIdx.
  // domIdx_[rank][domId] = domIdx
  std::vector<std::vector<Dim3>> domIdx_;

public:
  Dim3 gpu_dim() const { return gpuDim_; }
  Dim3 rank_dim() const { return rankDim_; }

  /*! return the rankIdx for a rank
   */
  Dim3 rank_idx(const int rank) const {
    assert(rank < ranks_);
    return dimensionize(rank, rankDim_);
  }

  /*! return the compute domain index associated with a particular rank and
      domain ID `domId`.
  */
  Dim3 dom_idx(int rank, int domId) {
    assert(rank < ranks_);
    assert(rank < domIdx_.size());
    assert(domId < domIdx_[rank].size());
    const Dim3 ret = domIdx_[rank][domId];
    assert(ret.all_lt(gpu_dim() * rank_dim()));
    return ret;
  }

  /* return the rank for a domain
   */
  int get_rank(Dim3 idx) const {
    idx /= gpuDim_;
    return linearize(idx, rankDim_);
  }

  /*! return the domain id for a domain, consistent with indices passed into
      the constructor
  */
  int get_gpu(const Dim3 &domIdx) {
    return domId_[domIdx];
  }

  int get_cuda(const Dim3 &domIdx) {
    return cudaId_[domIdx];
  }

  Dim3 domain_size(const Dim3 &domIdx) {

    Dim3 ret = domSize_;
    Dim3 rem = size_ % (rankDim_ * gpuDim_);

    if (rem.x != 0 && domIdx.x >= rem.x) {
      ret.x -= 1;
    }
    if (rem.y != 0 && domIdx.y >= rem.y) {
      ret.y -= 1;
    }
    if (rem.z != 0 && domIdx.z >= rem.z) {
      ret.z -= 1;
    }

    return ret;
  }

  NodeAwarePlacement(const Dim3 &domSize,
                     const int ranks, // how many ranks there are
                     MpiTopology &mpiTopo, GpuTopology &gpuTopo, size_t radius,
                     const std::vector<int>
                         &rankGpus // which GPUs this rank wants to contribute
                     )
      : size_(domSize), ranks_(ranks), gpuDim_(1, 1, 1), rankDim_(1, 1, 1) {

    // TODO: make sure everyone is contributing the same number of GPUs
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (auto &dev : rankGpus) {
      std::cerr << "rank " << rank << " contributing cuda " << dev << "\n";
    }

    domIdx_.resize(ranks_);
    for (auto &v : domIdx_) {
      v.resize(rankGpus.size());
    }

    domSize_ = size_;
    auto rankFactors = prime_factors(ranks_);

    // split repeatedly by prime factors of the number of MPI ranks to
    // establish the 3D partition among ranks
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

    // partition domain among GPUs in the rank
    auto gpuFactors = prime_factors(rankGpus.size());
    for (size_t amt : gpuFactors) {
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

    std::cerr << "NodeAwarePlacement: " << rankDim_ << "x" << gpuDim_ << "\n";

    // each rank on the node reports its global rank
    std::vector<int> nodeRanks(mpiTopo.colocated_size());
    MPI_Allgather(&rank, 1, MPI_INT, nodeRanks.data(), 1, MPI_INT,
                  mpiTopo.colocated_comm());
    std::cerr << "NAP: colo global ranks: ";
    for (auto &e : nodeRanks) {
      std::cerr << e << " ";
    }
    std::cerr << "\n";

    // each rank reports which GPUs it is contributing
    std::vector<int> nodeGpus(mpiTopo.colocated_size() * rankGpus.size());
    MPI_Allgather(rankGpus.data(), rankGpus.size(), MPI_INT, nodeGpus.data(),
                  rankGpus.size(), MPI_INT, mpiTopo.colocated_comm());
    std::cerr << "NAP: colo GPUs: ";
    for (auto &e : nodeGpus) {
      std::cerr << e << " ";
    }
    std::cerr << "\n";

    // record communication costs between all domains on this node
    Mat2D<double> commCost;
    const size_t numDomains = gpuDim_.flatten() * nodeRanks.size();
    commCost.resize(numDomains);
    for (auto &r : commCost) {
      r = std::vector<double>(numDomains, 0);
    }
    const Dim3 domDim = gpuDim_ * rankDim_;
    for (size_t i = 0; i < nodeRanks.size(); ++i) {
      for (size_t j = 0; j < rankGpus.size(); ++j) {
        Dim3 srcRankIdx = dimensionize(nodeRanks[i], rankDim_);
        Dim3 srcGpuIdx = dimensionize(j, gpuDim_);
        Dim3 srcDomIdx = srcRankIdx * gpuDim_ + srcGpuIdx;

        for (size_t k = 0; k < nodeRanks.size(); ++k) {
          for (size_t m = 0; m < rankGpus.size(); ++m) {
            Dim3 dstRankIdx = dimensionize(nodeRanks[k], rankDim_);
            Dim3 dstGpuIdx = dimensionize(m, gpuDim_);
            Dim3 dstDomIdx = dstRankIdx * gpuDim_ + dstGpuIdx;

            Dim3 dir = dstDomIdx - srcDomIdx;
            // periodic boundary
            if (dir.x == domDim.x - 1) dir.x = -1;
	    if (dir.y == domDim.y - 1) dir.y = -1;
            if (dir.z == domDim.z - 1) dir.z = -1;
            if (dir.x == 1 - domDim.x) dir.x = 1;
	    if (dir.y == 1 - domDim.y) dir.y = 1;
            if (dir.z == 1 - domDim.z) dir.z = 1;
            std::cerr << dir << "=" << srcDomIdx << "->" << dstDomIdx << "\n";
            if (Dim3(0, 0, 0) == dir || dir.any_gt(1) || dir.any_lt(-1)) {
              continue;
            } else {
              const Dim3 sz = domain_size(srcDomIdx);
              double cost = comm_cost(dir, sz, radius);
              commCost[i * gpuDim_.flatten() + j][k * gpuDim_.flatten() + m] =
                  cost;
            }
          }
        }
      }
    }

    {
      std::stringstream ss;
      for (auto &r : commCost) {
        for (auto &c : r) {
          ss << c << " ";
        }
        ss << "\n";
      }
      std::cerr << "domain exchange cost matrix\n";
      std::cerr << ss.str();
    }

    // build a bandwidth matrix between all participating GPUs on the node
    Mat2D<double> gpuBandwidth;
    gpuBandwidth.resize(numDomains);
    for (auto &r : gpuBandwidth) {
      r = std::vector<double>(numDomains, 0);
    }

    for (size_t i = 0; i < gpuBandwidth.size(); ++i) {
      for (size_t j = 0; j < gpuBandwidth[i].size(); ++j) {
        int srcId = nodeGpus[i];
        int dstId = nodeGpus[j];
        gpuBandwidth[i][j] = gpuTopo.bandwidth(srcId, dstId);
      }
    }

    {
      std::stringstream ss;
      for (auto &r : gpuBandwidth) {
        for (auto &c : r) {
          ss << c << " ";
        }
        ss << "\n";
      }
      std::cerr << "bw matrix\n";
      std::cerr << ss.str();
    }

    assert(gpuBandwidth.size() == commCost.size());

    // which domain on the node to map to which GPU on the node
    std::vector<size_t> mapping;
    for (size_t i = 0; i < numDomains; ++i) {
      mapping.push_back(i);
    }

    std::vector<size_t> bestMap = mapping;
    double bestFit = -1;
    do {

      auto placedCommCost = permute(commCost, mapping);

      {
#if 0
        std::cerr << "checking permutation";
        for (auto &e : mapping) {
          std::cerr << " " << e;
        }
        std::stringstream ss;
        for (auto &r : placedCommCost) {
          for (auto &c : r) {
            ss << c << " ";
          }
          ss << "\n";
        }

        std::cerr << "\n";
        std::cerr << ss.str();
#endif
      }

      const double score = match(placedCommCost, gpuBandwidth);
#if 0
      std::cerr << "score=" << score << "\n";
#endif
      if (score > bestFit) {
        bestFit = score;
        bestMap = mapping;
#if 0
        {
          std::stringstream ss;
          for (auto &r : placedCommCost) {
            for (auto &c : r) {
              ss << c << " ";
            }
            ss << "\n";
          }
          std::cerr << "new best placement:\n";
          for (auto &e : mapping) {
            std::cerr << e << " ";
          }
          std::cerr << "\n";
          std::cerr << ss.str();
        }
#endif
      }
    } while (std::next_permutation(mapping.begin(), mapping.end()));


    // gather up per-rank and global information about what cuda device is used for each domain
    std::vector<Dim3> localDomIdx;
    std::vector<Dim3> allDomIdx(rankGpus.size() * mpiTopo.size());
    std::vector<int> localDomId;
    std::vector<int> allDomId(rankGpus.size() * mpiTopo.size());
    std::vector<int> allCudaId(rankGpus.size() * mpiTopo.size());

    std::cerr << "found best placement\n";
    std::cerr << "bestMap was ";
    for (auto &e : bestMap) std::cerr << e << " ";
    std::cerr << "\n";
    for (size_t i = 0; i < bestMap.size(); ++i) {
      const int id = bestMap[i]; // id of gpu in the node
      const int coloRank = id / rankGpus.size();
      const int rank = nodeRanks[coloRank];
      const int domId = id % rankGpus.size();
      const Dim3 gpuIdx = dimensionize(domId, gpuDim_);
      const Dim3 rankIdx = dimensionize(rank, rankDim_); 
      const Dim3 domIdx = rankIdx * gpuDim_ + gpuIdx;
      std::cerr << "rank=" << rank << " colo-rank=" << coloRank << " id=" << id << "(cuda=" << nodeGpus[id]
                << ") rankIdx=" << rankIdx << " gpuIdx=" << gpuIdx << " domIdx=" << domIdx
                << " domId=" << domId << "\n";
      // keep track of my own domain id / domain index corredponence
      if (coloRank == mpiTopo.colocated_rank()) {
        localDomIdx.push_back(domIdx);
        localDomId.push_back(domId);
      }
    }
    std::cerr << "\n";

    // should be one domain in the rank per GPU
    assert(localDomIdx.size() == rankGpus.size());

    // all ranks provide their domain indices
    {
    size_t numBytes = localDomIdx.size() * sizeof(localDomIdx[0]);
    MPI_Allgather(localDomIdx.data(), numBytes, MPI_BYTE, allDomIdx.data(), numBytes, MPI_BYTE, mpiTopo.comm());
    }

    // all ranks provide the corresponding cuda device ID
    MPI_Allgather(rankGpus.data(), rankGpus.size(), MPI_INT, allCudaId.data(), rankGpus.size(), MPI_INT, mpiTopo.comm());

    // all ranks provide the corresponding domain ID
    MPI_Allgather(localDomId.data(), localDomId.size(), MPI_INT, allDomId.data(), localDomId.size(), MPI_INT, mpiTopo.comm());


    if (mpiTopo.rank() == 0) {
      std::cerr << "allDomIdx:";
      for (auto &e : allDomIdx) std::cerr << e << " ";
      std::cerr << "\n";      
      std::cerr << "allCudaId:";
      for (auto &e : allCudaId) std::cerr << e << " ";
      std::cerr << "\n";      
      std::cerr << "allDomId:";
      for (auto &e : allDomId) std::cerr << e << " ";
      std::cerr << "\n";      
   }

    // record info from all ranks
    for (size_t i = 0; i < allCudaId.size(); ++i) {
      const int rank = i / rankGpus.size(); // TODO: assuming all ranks provide the same number of GPUs
      const Dim3 domIdx = allDomIdx[i];
      const int cuda = allCudaId[i];
      const int domId = allDomId[i];

      assert(cudaId_.count(domIdx) == 0);
      cudaId_[domIdx] = cuda;
      domId_[domIdx] = domId;

      assert(rank < domIdx_.size());
      assert(domId < domIdx_[rank].size());
      domIdx_[rank][domId] = domIdx;
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

  /*! \brief ceil(n/d)
   */
  static size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }
};
