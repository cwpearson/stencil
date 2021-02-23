#pragma once

#include "stencil/partition.hpp"

#include <random>

/* Group subgrids by node, but within every node, randomly place subdomains onto GPUs.

*/
class IntraNodeRandom : public Placement {
private:
  NodePartition partition_;
  std::mt19937 generator_;

  // convert idx to rank
  std::map<Dim3, int> rank_;

  // convert idx to subdomain id
  std::map<Dim3, int> subdomainId_;

  // get cuda device for idx
  std::map<Dim3, int> cuda_;

  // convert rank and subdomain to idx
  std::vector<std::vector<Dim3>> idx_;

public:
  IntraNodeRandom(const Dim3 &size, // total domain size
                  MpiTopology &mpiTopo, Radius radius,
                  const std::vector<int> &rankCudaIds // which CUDA devices the calling
                                                      // rank wants to contribute
  );

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

};
