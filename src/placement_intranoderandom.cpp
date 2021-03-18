#include "stencil/placement_intranoderandom.hpp"

IntraNodeRandom::IntraNodeRandom(const Dim3 &size, // total domain size
                                 MpiTopology &mpiTopo, Radius radius,
                                 const std::vector<int> &rankCudaIds // which CUDA devices the calling
                                                                     // rank wants to contribute
                                 )
    : generator_(0) {
  LOG_DEBUG("IntraNodeRandom: entered ctor");
  MPI_Barrier(MPI_COMM_WORLD);
  LOG_DEBUG("IntraNodeRandom: after barrier");

  // TODO: actually check that everyone has the same number of GPUs
  const int gpusPerRank = rankCudaIds.size();
  const int ranksPerNode = mpiTopo.colocated_size();
  const int gpusPerNode = gpusPerRank * ranksPerNode;
  const int numNodes = mpiTopo.size() / mpiTopo.colocated_size();
  const int numSubdomains = numNodes * gpusPerNode;

  partition_ = NodePartition(size, radius, numNodes, gpusPerNode);

  if (0 == mpi::world_rank()) {
    LOG_INFO("IntraNodeRandom: " << partition_.sys_dim() << "x" << partition_.node_dim());
  }

  // get the name of each node
  char name[MPI_MAX_PROCESSOR_NAME] = {0};
  int nameLen;
  MPI_Get_processor_name(name, &nameLen);
  if (0 == mpiTopo.rank()) {
    LOG_DEBUG("IntraNodeRandom: got name " << name);
  }

  // gather the names to root
  std::vector<char> allNames;
  if (0 == mpiTopo.rank()) {
    allNames.resize(MPI_MAX_PROCESSOR_NAME * mpiTopo.size());
  }
  MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allNames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0,
             MPI_COMM_WORLD);
  if (0 == mpiTopo.rank()) {
    LOG_DEBUG("IntraNodeRandom: gathered names to root");
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
    std::cerr << "DEBUG: IntraNodeRandom: built names for each rank\n";
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
    std::cerr << "DEBUG: IntraNodeRandom: numbered each name\n";
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
    std::cerr << "DEBUG: IntraNodeRandom: build ranks in each node\n";
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

    // randomize placement on each node
    for (int node = 0; node < numNodes; ++node) {

      const Dim3 sysIdx = partition_.sys_idx(node);
      auto &ranks = nodeRanks[node]; // ranks in this node
      std::cerr << "placement on node " << node << " sys_idx=" << sysIdx << "\n";
      std::cerr << "ranks:";
      for (auto &e : ranks)
        std::cerr << " " << e;
      std::cerr << "\n";
      assert(ranks.size() == ranksPerNode);

      // which component each subdomain should be on. assign randomly
      std::vector<size_t> components(gpusPerNode);
      std::iota(components.begin(), components.end(), 0);
      std::shuffle(components.begin(), components.end(), generator_);

      std::cerr << "components:";
      for (auto &e : components)
        std::cerr << " " << e;
      std::cerr << "\n";

      for (int64_t id = 0; id < gpusPerNode; ++id) {

        // each component is owned by a rank and has a local ID
        size_t component = components[id];
        const int ri = component / gpusPerRank;
        const int rank = ranks[ri];
        const int gpuId = component % gpusPerRank;
        const int cuda = globalCudaIds[rank * gpusPerRank + gpuId];

        // implicitly, the global ID is grouped by node, and subdomain id within that node
        const size_t gi = node * gpusPerNode + id;
        {
          const Dim3 nodeIdx = partition_.node_idx(id);
#if 0
          LOG_DEBUG("global id=" << gi << " nodeIdx=" << nodeIdx
                                 << " size=" << partition_.subdomain_size(sysIdx * nodeDim + nodeIdx)
                                 << " rank=" << rank << " gpuId=" << gpuId << " (cuda=" << cuda << ")");
#endif
          (void)nodeIdx; // in case debug is not defined
        }

        rankAssignment[gi] = rank;
        idForDomain[gi] = gpuId;
        cudaAssignment[gi] = cuda;
      }
    }

  } // 0 == rank

  // broadcast the data to all ranks
  MPI_Bcast(rankAssignment.data(), rankAssignment.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(idForDomain.data(), idForDomain.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(cudaAssignment.data(), cudaAssignment.size(), MPI_INT, 0, MPI_COMM_WORLD);

  // implicitly first M are first node, next M are second node, so not okay to use partition_.idx()
  for (size_t gi = 0; gi < rankAssignment.size(); ++gi) {
    const int subdomain = idForDomain[gi];
    const int rank = rankAssignment[gi];
    const int cuda = cudaAssignment[gi];

    // recover node
    const size_t node = gi / gpusPerNode;
    const size_t inNodeId = gi % gpusPerNode;

    // convert into a full global index
    Dim3 sysIdx = partition_.sys_idx(node);
    Dim3 nodeIdx = partition_.node_idx(inNodeId);
    Dim3 idx = sysIdx * partition_.node_dim() + nodeIdx;

    // convert index to rank, gpuId, and actual device
    rank_[idx] = rank;
    subdomainId_[idx] = subdomain;
    cuda_[idx] = cuda;

    if (0 == mpiTopo.rank()) {
      std::cerr << "idx=" << idx << " size=" << partition_.subdomain_size(idx) << " rank=" << rank << " node=" << node
                << " inNodeId=" << inNodeId << " sysIdx=" << sysIdx << " nodeIdx=" << nodeIdx
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