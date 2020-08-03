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
#include "gpu_topology.hpp"
#include "local_domain.cuh"
#include "mat2d.hpp"
#include "mpi_topology.hpp"
#include "stencil/logging.hpp"
#include "stencil/qap.hpp"
#include "stencil/radius.hpp"

namespace collective {}

class RankPartition {

private:
  Dim3 dim_;  // the number of subdomains
  Dim3 size_; // the size of each subdomain
  Dim3 rem_;  // input size % dim_

  /*! return the prime factors of n
   */
  static std::vector<int64_t> prime_factors(int64_t n) {
    std::vector<int64_t> result;
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
    std::sort(result.begin(), result.end(), [](int64_t a, int64_t b) { return b < a; });
    return result;
  }

  static int64_t div_ceil(int64_t n, int64_t d) { return (n + d - 1) / d; }

public:
  RankPartition(const Dim3 &size, const int64_t n) : size_(size), dim_(1, 1, 1) {

    // split repeatedly by the prime factors of n
    std::vector<int64_t> factors = prime_factors(n);
    for (size_t amt : factors) {
      if (amt < 2) {
        continue;
      }

      if (size_.x >= size_.y && size_.x >= size_.z) { // split in x
        size_.x = div_ceil(size_.x, amt);
        dim_.x *= amt;
      } else if (size_.y >= size_.z) { // split in y
        size_.y = div_ceil(size_.y, amt);
        dim_.y *= amt;
      } else { // split in z
        size_.z = div_ceil(size_.z, amt);
        dim_.z *= amt;
      }
    }

    rem_ = size % dim_;
  }
  RankPartition() : RankPartition(Dim3(0, 0, 0), 0) {}

  virtual Dim3 dim() const { return dim_; }

  virtual Dim3 subdomain_size(const Dim3 &idx) const {

    Dim3 ret = size_;

    if (rem_.x != 0 && idx.x >= rem_.x) {
      ret.x -= 1;
    }
    if (rem_.y != 0 && idx.y >= rem_.y) {
      ret.y -= 1;
    }
    if (rem_.z != 0 && idx.z >= rem_.z) {
      ret.z -= 1;
    }

    return ret;
  }

  Dim3 subdomain_origin(const Dim3 &idx) const noexcept {
    Dim3 ret = size_ * idx;

    if (rem_.x != 0 && idx.x >= rem_.x) {
      ret.x -= (idx.x - rem_.x);
    }
    if (rem_.y != 0 && idx.y >= rem_.y) {
      ret.y -= (idx.y - rem_.y);
    }
    if (rem_.z != 0 && idx.z >= rem_.z) {
      ret.z -= (idx.z - rem_.z);
    }

    return ret;
  }

  /* get a unique 1D integer for an index */
  size_t linearize(Dim3 idx) const {
    Dim3 dim = this->dim();
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

  /* opposite of linearize */
  Dim3 dimensionize(int64_t i) {
    Dim3 dim = this->dim();
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
};

/* A 2-level partition of a 3D space amongst nodes in the system, and then GPUs in the node
 */
class NodePartition {

private:
  Dim3 sysDim_;  // dimension of the system
  Dim3 nodeDim_; // dimension of a node

  Dim3 size_; // approximate subdomain size
  Dim3 rem_;  // input size % sysDim_ * nodeDim_

  /*! return the prime factors of n, sorted smallest to largest
   */
  static std::vector<int64_t> prime_factors(int64_t n) {
    std::vector<int64_t> result;
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
    std::sort(result.begin(), result.end(), [](int64_t a, int64_t b) { return b < a; });
    return result;
  }

  static int64_t div_ceil(int64_t n, int64_t d) { return (n + d - 1) / d; }

  /* linearize an index `idx` in a space `dim` */
  static int64_t linearize(const Dim3 idx, const Dim3 dim) {
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

  /* opposite of linearize */
  static Dim3 dimensionize(int64_t i, const Dim3 &dim) {
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

public:
  NodePartition(const Dim3 &size, const Radius radius, const int64_t nodes, const int64_t gpus)
      : size_(size), sysDim_(1, 1, 1), nodeDim_(1, 1, 1) {

    // split among nodes
    std::vector<int64_t> factors = prime_factors(nodes);
    for (size_t amt : factors) {
      if (amt < 2) {
        continue;
      }

      /* Recursively split a 3D region along whatever plane results in the smallest interface.
         The interface size is scaled by the kernel radius in that dimension, positive + negative
      */

      const int64_t xIface = size_.y * size_.z * (radius.dir(1, 0, 0) + radius.dir(-1, 0, 0));
      const int64_t yIface = size_.x * size_.z * (radius.dir(0, 1, 0) + radius.dir(0, -1, 0));
      const int64_t zIface = size_.x * size_.y * (radius.dir(0, 0, 1) + radius.dir(0, 0, -1));

      if (xIface <= yIface && xIface <= zIface) { // split in x
        size_.x = div_ceil(size_.x, amt);
        sysDim_.x *= amt;
      } else if (yIface <= zIface) { // split in y
        size_.y = div_ceil(size_.y, amt);
        sysDim_.y *= amt;
      } else { // split in z
        size_.z = div_ceil(size_.z, amt);
        sysDim_.z *= amt;
      }
    }

    // split among gpus
    factors = prime_factors(gpus);
    for (size_t amt : factors) {
      if (amt < 2) {
        continue;
      }

      const int64_t xIface = size_.y * size_.z * (radius.dir(1, 0, 0) + radius.dir(-1, 0, 0));
      const int64_t yIface = size_.x * size_.z * (radius.dir(0, 1, 0) + radius.dir(0, -1, 0));
      const int64_t zIface = size_.x * size_.y * (radius.dir(0, 0, 1) + radius.dir(0, 0, -1));

      if (xIface <= yIface && xIface <= zIface) { // split in x
        size_.x = div_ceil(size_.x, amt);
        nodeDim_.x *= amt;
      } else if (yIface <= zIface) { // split in y
        size_.y = div_ceil(size_.y, amt);
        nodeDim_.y *= amt;
      } else { // split in z
        size_.z = div_ceil(size_.z, amt);
        nodeDim_.z *= amt;
      }
    }

    rem_ = size % (sysDim_ * nodeDim_);
  }

  NodePartition() : NodePartition(Dim3(0, 0, 0), Radius::constant(0), 0, 0) {}

  Dim3 sys_dim() const noexcept { return sysDim_; }

  Dim3 node_dim() const noexcept { return nodeDim_; }

  Dim3 dim() const noexcept { return sys_dim() * node_dim(); }

  Dim3 subdomain_size(const Dim3 &idx) const {

    Dim3 ret = size_;

    if (rem_.x != 0 && idx.x >= rem_.x) {
      ret.x -= 1;
    }
    if (rem_.y != 0 && idx.y >= rem_.y) {
      ret.y -= 1;
    }
    if (rem_.z != 0 && idx.z >= rem_.z) {
      ret.z -= 1;
    }

    return ret;
  }

  Dim3 subdomain_origin(const Dim3 &idx) const noexcept {
    Dim3 ret = size_ * idx;

    if (rem_.x != 0 && idx.x >= rem_.x) {
      ret.x -= (idx.x - rem_.x);
    }
    if (rem_.y != 0 && idx.y >= rem_.y) {
      ret.y -= (idx.y - rem_.y);
    }
    if (rem_.z != 0 && idx.z >= rem_.z) {
      ret.z -= (idx.z - rem_.z);
    }

    return ret;
  }

  Dim3 sys_idx(int64_t i) const noexcept { return dimensionize(i, sys_dim()); }
  Dim3 node_idx(int64_t i) const noexcept { return dimensionize(i, node_dim()); }
  Dim3 idx(int64_t i) const noexcept { return dimensionize(i, dim()); }
};

enum class PlacementStrategy { NodeAware, Trivial };

class Placement {

public:
  // get the index of subdomain i for rank
  virtual Dim3 get_idx(const int rank, const int i) = 0;

  // get the MPI rank for a subdomain
  virtual int get_rank(const Dim3 &idx) = 0;

  // get the subdomain's id within the rank it is placed in
  virtual int get_subdomain_id(const Dim3 &idx) = 0;

  // get the cuda id for a subdomain
  virtual int get_cuda(const Dim3 &idx) = 0;

  // get the size of a subdomain
  virtual Dim3 subdomain_size(const Dim3 &idx) = 0;

  // get the origin of a subdomain
  virtual Dim3 subdomain_origin(const Dim3 &idx) = 0;

  // upper bound for idx
  virtual Dim3 dim() = 0;
};

class Trivial : public Placement {
private:
  RankPartition partition_;

  /* idx_[rank][id] = idx */
  std::vector<std::vector<Dim3>> idx_;

  // get the rank that owns a subdomain
  std::map<Dim3, int> rank_;

  // get the subdomain id (within a rank) for a subdomain
  std::map<Dim3, int> subdomain_id_;

  // get the CUDA devixe id for a subdomain
  std::map<Dim3, int> cuda_;

public:
  /*! return the compute domain index associated with a particular rank and
      domain ID `domId`.
  */
  Dim3 get_idx(int rank, int domId) override {
    assert(rank < idx_.size());
    assert(domId < idx_[rank].size());
    return idx_[rank][domId];
  }

  /* return the rank for a domain
   */
  int get_rank(const Dim3 &idx) override { return rank_[idx]; }

  /*! return the domain id for a domain, consistent with indices passed into
      the constructor
  */
  int get_subdomain_id(const Dim3 &idx) override { return subdomain_id_[idx]; }

  /* which cuda device a subdomain runs on */
  int get_cuda(const Dim3 &idx) override { return cuda_[idx]; }

  /* size of a subdomain */
  Dim3 subdomain_size(const Dim3 &idx) override { return partition_.subdomain_size(idx); }

  /* origin of a subdomain */
  Dim3 subdomain_origin(const Dim3 &idx) override { return partition_.subdomain_origin(idx); }

  Dim3 dim() override { return partition_.dim(); }

  Trivial(const Dim3 &size, // total domain size
          MpiTopology &mpiTopo,
          const std::vector<int> &rankCudaIds // which CUDA devices the calling
                                              // rank wants to contribute
  ) {
    MPI_Barrier(MPI_COMM_WORLD);
    // have everyone request one work item per GPU
    const int workItems = rankCudaIds.size();
    std::vector<int> workItemCounts(mpiTopo.size());
    MPI_Allgather(&workItems, 1, MPI_INT, workItemCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (mpiTopo.rank() == 0) {
      std::cerr << "Trivial: workItemCounts:";
      for (auto &e : workItemCounts) {
        std::cerr << " " << e;
      }
      std::cerr << "\n";
    }

    const int numSubdomains = std::accumulate(workItemCounts.begin(), workItemCounts.end(), 0);

    if (mpiTopo.rank() == 0) {
      std::cerr << "Trivial: numSubdomains=" << numSubdomains << "\n";
    }

    partition_ = RankPartition(size, numSubdomains);

    // determine which rank each subdomain will be assigned to
    // determine what the subdomain id within each rank each subdomain is
    std::vector<int> rankAssignments;
    std::vector<int> rankIds;
    for (size_t rank = 0; rank < workItemCounts.size(); ++rank) {
      for (int i = 0; i < workItemCounts[rank]; ++i) {
        rankAssignments.push_back(rank);
        rankIds.push_back(int(i));
      }
    }
    assert(rankAssignments.size() == numSubdomains);
    assert(rankIds.size() == numSubdomains);
    if (mpiTopo.rank() == 0) {
      std::cerr << "Trivial: rankAssignments:";
      for (auto &e : rankAssignments) {
        std::cerr << " " << e;
      }
      std::cerr << "\n";

      std::cerr << "Trivial: rankIds:";
      for (auto &e : rankIds) {
        std::cerr << " " << e;
      }
      std::cerr << "\n";
    }

    // figure out the cuda IDs contributed from each rank
    std::vector<int> offs;
    int val = 0;
    for (const auto &e : workItemCounts) {
      offs.push_back(val);
      val += e;
    }

    // determine which CUDA id each subdomain will be assigned to
    std::vector<int> cudaAssignments(numSubdomains);
    MPI_Allgatherv(rankCudaIds.data(), rankCudaIds.size(), MPI_INT, cudaAssignments.data(), workItemCounts.data(),
                   offs.data(), MPI_INT, MPI_COMM_WORLD);

    if (0 == mpiTopo.rank()) {
      std::cerr << "Trivial: cudaAssignments:";
      for (auto &e : cudaAssignments) {
        std::cerr << " " << e;
      }
      std::cerr << "\n";
    }

    // fill data
    assert(cudaAssignments.size() == numSubdomains);
    assert(rankIds.size() == numSubdomains);
    assert(rankAssignments.size() == numSubdomains);
    for (int i = 0; i < numSubdomains; ++i) {
      const int rank = rankAssignments[i];
      const int id = rankIds[i];
      const int cuda = cudaAssignments[i];
      const Dim3 idx = partition_.dimensionize(i);
      const Dim3 sdSize = partition_.subdomain_size(idx);

      if (0 == mpiTopo.rank()) {
        std::cerr << idx << "is sd" << id << " on r" << rank << " cuda" << cuda << ")" << sdSize << "\n";
      }

      assert(rank >= 0);
      assert(id >= 0);
      assert(rank_.count(idx) == 0);

      rank_[idx] = rank;
      assert(subdomain_id_.count(idx) == 0);
      subdomain_id_[idx] = id;
      if (idx_.size() <= size_t(rank)) {
        idx_.resize(rank + 1);
      }
      if (idx_[rank].size() <= size_t(id)) {
        idx_[rank].resize(id + 1);
      }
      idx_[rank][id] = idx;
      cuda_[idx] = cuda;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

/*! average
 */
inline double avg(const std::vector<double> &x) {
  assert(x.size() >= 1);

  double acc = 0;
  for (auto &e : x) {
    acc += e;
  }
  return acc / double(x.size());
}

/*! corrected sample standard deviation
 */
inline double cssd(const std::vector<double> &x) {
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
inline double scc(const std::vector<double> &x, const std::vector<double> &y) {
  assert(x.size() == y.size());

  const size_t n = x.size();
  const double xBar = avg(x);
  const double yBar = avg(y);

  double num = 0;
  for (size_t i = 0; i < n; ++i) {
    num += (x[i] - xBar) * (y[i] - yBar);
  }

  double den = double(n - 1) * cssd(x) * cssd(y);

  if (0 == num && 0 == den) {
    return 1;
  } else {
    assert(0 != den);
    return num / den;
  }
}

/*
double match(const Mat2D<double> &x, const Mat2D<double> &y) {

  assert(x.shape() == y.shape());
  assert(x.size().x > 0);

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
*/

class NodeAware : public Placement {
private:
  NodePartition partition_;

  constexpr static double interNodeBandwidth = 15;
  constexpr static double gpuMemoryBandwidth = 900;

  /* Return a number proportional to the bytes in a halo exchange, along
   * direction `dir` for a domain of size `sz` with radius `radius`
   */
  double comm_cost(Dim3 dir, const Dim3 sz, const Radius radius) {
    assert(dir.all_lt(2));
    assert(dir.all_gt(-2));
    double count = double(LocalDomain::halo_extent(dir, sz, radius).flatten());
    return count;
  }

  // convert idx to rank
  std::map<Dim3, int> rank_;

  // convert idx to subdomain id
  std::map<Dim3, int> subdomainId_;

  // get cuda device for idx
  std::map<Dim3, int> cuda_;

  // convert rank and subdomain to idx
  std::vector<std::vector<Dim3>> idx_;

public:
  /*! return the compute domain index associated with a particular rank and
      domain ID `domId`.
  */
  Dim3 get_idx(int rank, int domId) override {
    assert(rank < idx_.size());
    assert(domId < idx_[rank].size());
    const Dim3 ret = idx_[rank][domId];
    return ret;
  }

  /* return the rank for a domain
   */
  int get_rank(const Dim3 &idx) override { return rank_[idx]; }

  /*! return the domain id for a domain, consistent with indices passed into
      the constructor
  */
  int get_subdomain_id(const Dim3 &idx) override { return subdomainId_[idx]; }

  int get_cuda(const Dim3 &idx) override { return cuda_[idx]; }

  Dim3 subdomain_size(const Dim3 &idx) override { return partition_.subdomain_size(idx); }

  /* origin of a subdomain */
  Dim3 subdomain_origin(const Dim3 &idx) override { return partition_.subdomain_origin(idx); }

  Dim3 dim() override { return partition_.dim(); }

  NodeAware(const Dim3 &size, // total domain size
            MpiTopology &mpiTopo, Radius radius,
            const std::vector<int> &rankCudaIds // which CUDA devices the calling
                                                // rank wants to contribute
  ) {
    LOG_DEBUG("NodeAware: entered ctor");
    MPI_Barrier(MPI_COMM_WORLD);
    LOG_DEBUG("NodeAware: after barrier");

    // TODO: actually check that everyone has the same number of GPUs
    const int gpusPerRank = rankCudaIds.size();
    const int ranksPerNode = mpiTopo.colocated_size();
    const int gpusPerNode = gpusPerRank * ranksPerNode;
    const int numNodes = mpiTopo.size() / mpiTopo.colocated_size();
    const int numSubdomains = numNodes * gpusPerNode;

    partition_ = NodePartition(size, radius, numNodes, gpusPerNode);

    if (0 == mpiTopo.rank()) {
      std::cerr << "NodeAware: " << partition_.sys_dim() << "x" << partition_.node_dim() << "\n";
    }

    // get the name of each node
    char name[MPI_MAX_PROCESSOR_NAME] = {0};
    int nameLen;
    MPI_Get_processor_name(name, &nameLen);
    if (0 == mpiTopo.rank()) {
      LOG_DEBUG("NodeAware: got name " << name);
    }

    // gather the names to root
    std::vector<char> allNames;
    if (0 == mpiTopo.rank()) {
      allNames.resize(MPI_MAX_PROCESSOR_NAME * mpiTopo.size());
    }
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allNames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0,
               MPI_COMM_WORLD);
    if (0 == mpiTopo.rank()) {
      LOG_DEBUG("NodeAware: gathered names to root");
    }

    // get the name for each rank
    std::vector<std::string> rankNames;
    if (0 == mpiTopo.rank()) {
      for (int rank = 0; rank < mpiTopo.size(); ++rank) {
        std::string _name(allNames.data() + rank * MPI_MAX_PROCESSOR_NAME);
        rankNames.push_back(_name);
      }
    }
    if (0 == mpiTopo.rank()) {
      std::cerr << "DEBUG: NodeAware: built names for each rank\n";
    }

    // number each name
    std::map<std::string, size_t> nodeNumbers;
    size_t nn = 0;
    if (0 == mpiTopo.rank()) {
      for (int rank = 0; rank < mpiTopo.size(); ++rank) {
        auto p = nodeNumbers.emplace(rankNames[rank], nn);
        if (p.second) {
          ++nn;
        }
      }
    }
    if (0 == mpiTopo.rank()) {
      std::cerr << "DEBUG: NodeAware: numbered each name\n";
    }

    // store the node number for each rank
    std::vector<int> rankNode(mpiTopo.size());
    // a vec of ranks that are in each node
    std::vector<std::vector<int>> nodeRanks(nodeNumbers.size());
    if (0 == mpiTopo.rank()) {
      for (int rank = 0; rank < mpiTopo.size(); ++rank) {
        int node = nodeNumbers[rankNames[rank]];
        rankNode[rank] = node;
        nodeRanks[node].push_back(rank);
      }
    }
    if (0 == mpiTopo.rank()) {
      std::cerr << "DEBUG: NodeAware: build ranks in each node\n";
    }

    // gather up all CUDA ids that all ranks are contributing
    std::vector<int> globalCudaIds(numSubdomains);
    MPI_Allgather(rankCudaIds.data(), rankCudaIds.size(), MPI_INT, globalCudaIds.data(), rankCudaIds.size(), MPI_INT,
                  mpiTopo.comm());

    if (0 == mpiTopo.rank()) {
      std::cerr << "globalCudaIds:";
      for (auto &e : globalCudaIds)
        std::cerr << " " << e;
      std::cerr << "\n";
    }

    // OUTPUTS
    // the CUDA device for each subdomain
    std::vector<int> cudaAssignment(numSubdomains);
    // the rank for each subdomain
    std::vector<int> rankAssignment(numSubdomains);
    // subdomain ID for each subdomain
    std::vector<int> idForDomain(numSubdomains);

    if (0 == mpiTopo.rank()) {

      const Dim3 nodeDim = partition_.node_dim();
      const Dim3 globalDim = nodeDim * partition_.sys_dim();

      // do placement separately for each node
      for (int node = 0; node < numNodes; ++node) {

        const Dim3 sysIdx = partition_.sys_idx(node);
        auto &ranks = nodeRanks[node]; // ranks in this node
        std::cerr << "placement on node " << node << " " << sysIdx << "\n";
        std::cerr << "ranks:";
        for (auto &e : ranks)
          std::cerr << " " << e;
        std::cerr << "\n";
        assert(ranks.size() == ranksPerNode);

        // make a bandwidth matrix for the components in this node
        Mat2D<double> bandwidth(gpusPerNode, gpusPerNode, 0.0);
        for (int64_t ri = 0; ri < ranksPerNode; ++ri) {  // rank i in this node
          for (int64_t gi = 0; gi < gpusPerRank; ++gi) { // gpu i in this rank
            const int64_t ci = ri * gpusPerRank + gi;
            for (int64_t rj = 0; rj < ranksPerNode; ++rj) {  // rank j in this node
              for (int64_t gj = 0; gj < gpusPerRank; ++gj) { // gpu j in this rank
                const int64_t cj = rj * gpusPerRank + gj;

                // recover the cuda device ID for this component
                const int di = globalCudaIds[ranks[ri] * gpusPerRank + gi];
                const int dj = globalCudaIds[ranks[rj] * gpusPerRank + gj];
                bandwidth[ci][cj] = gpu_topo::bandwidth(di, dj);
              }
            }
          }
        }

        // build a stencil communication matrix for the domains in this node
        Mat2D<double> comm(gpusPerNode, gpusPerNode, 0.0);
        for (int64_t i = 0; i < gpusPerNode; ++i) {
          const Dim3 srcIdx = sysIdx * nodeDim + partition_.node_idx(i);
          for (int64_t j = 0; j < gpusPerNode; ++j) {
            const Dim3 dstIdx = sysIdx * nodeDim + partition_.node_idx(j);

            Dim3 dir = dstIdx - srcIdx;
            // periodic boundary
            if (dir.x != 0 && dir.x == globalDim.x - 1)
              dir.x = -1;
            if (dir.y != 0 && dir.y == globalDim.y - 1)
              dir.y = -1;
            if (dir.z != 0 && dir.z == globalDim.z - 1)
              dir.z = -1;
            if (dir.x != 0 && dir.x == 1 - globalDim.x)
              dir.x = 1;
            if (dir.y != 0 && dir.y == 1 - globalDim.y)
              dir.y = 1;
            if (dir.z != 0 && dir.z == 1 - globalDim.z)
              dir.z = 1;
            std::cerr << dir << "=" << srcIdx << "->" << dstIdx << "\n";
            if (Dim3(0, 0, 0) == dir || dir.any_gt(1) || dir.any_lt(-1)) {
              continue;
            } else {
              const Dim3 sz = partition_.subdomain_size(srcIdx);
              double cost = comm_cost(dir, sz, radius);
              comm[i][j] = cost;
            }
          }
        }

        // which component each subdomain should be on
        Mat2D<double> distance = make_reciprocal(bandwidth);
        std::vector<size_t> components = qap::solve(comm, distance);

        std::cerr << "components:";
        for (auto &e : components)
          std::cerr << " " << e;
        std::cerr << "\n";

        for (int64_t id = 0; id < gpusPerNode; ++id) {
          const Dim3 nodeIdx = partition_.node_idx(id);
          const Dim3 sdSize = partition_.subdomain_size(sysIdx * nodeDim + nodeIdx);

          // each component is owned by a rank and has a local ID
          size_t component = components[id];
          const int ri = component / gpusPerRank;
          const int rank = ranks[ri];
          const int gpuId = component % gpusPerRank;
          const int cuda = globalCudaIds[rank * gpusPerRank + gpuId];

          std::cerr << "nodeIdx=" << nodeIdx << " size=" << sdSize << " rank=" << rank << " gpuId=" << gpuId
                    << " cuda=" << cuda << "\n";

          rankAssignment[node * gpusPerNode + id] = rank;
          idForDomain[node * gpusPerNode + id] = gpuId;
          cudaAssignment[node * gpusPerNode + id] = cuda;
        }
      }

    } // 0 == rank

    // broadcast the data to all ranks
    MPI_Bcast(rankAssignment.data(), rankAssignment.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(idForDomain.data(), idForDomain.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cudaAssignment.data(), cudaAssignment.size(), MPI_INT, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < rankAssignment.size(); ++i) {
      // convert i into a domain idx
      const Dim3 idx = partition_.idx(i);
      const int subdomain = idForDomain[i];
      const int rank = rankAssignment[i];
      const int cuda = cudaAssignment[i];

      // convert index to rank, gpuId, and actual device
      rank_[idx] = rank;
      subdomainId_[idx] = subdomain;
      cuda_[idx] = cuda;

      if (0 == mpiTopo.rank()) {
        std::cerr << "idx=" << idx << " size=" << partition_.subdomain_size(idx) << " rank=" << rank
                  << " subdomain=" << subdomain << " cuda=" << cuda << "\n";
      }

      // convert rank and subdomain to idx
      assert(rank >= 0);
      if (idx_.size() <= size_t(rank))
        idx_.resize(rank + 1);
      assert(subdomain >= 0);
      if (idx_[rank].size() <= size_t(subdomain))
        idx_[rank].resize(subdomain + 1);
      idx_[rank][subdomain] = idx;
    }
  }
};
