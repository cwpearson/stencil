#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <set>
#include <vector>

#include <mpi.h>

#include <nvToolsExt.h>
#include <nvml.h>

#include "cuda_runtime.hpp"

#include "stencil/dim3.hpp"
#include "stencil/direction_map.hpp"
#include "stencil/gpu_topology.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/logging.hpp"
#include "stencil/machine.hpp"
#include "stencil/mpi_topology.hpp"
#include "stencil/nvml.hpp"
#include "stencil/partition.hpp"
#include "stencil/radius.hpp"
#include "stencil/tx.hpp"
#include "stencil/tx_cuda.cuh"

enum class MethodFlags {
  None = 0,
  CudaMpi = 1,
  CudaAwareMpi = 2,
  CudaMpiColocated = 4,
  CudaMemcpyPeer = 8,
  CudaKernel = 16,
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  All = 1 + 2 + 4 + 8 + 16
#else
  All = 1 + 4 + 8 + 16
#endif
};
static_assert(sizeof(MethodFlags) == sizeof(int), "int");

inline MethodFlags operator|(MethodFlags a, MethodFlags b) {
  return static_cast<MethodFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline MethodFlags &operator|=(MethodFlags &a, MethodFlags b) {
  a = a | b;
  return a;
}

inline MethodFlags operator&(MethodFlags a, MethodFlags b) {
  return static_cast<MethodFlags>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool operator&&(MethodFlags a, MethodFlags b) { return (a & b) != MethodFlags::None; }

inline bool any(MethodFlags a) noexcept { return a != MethodFlags::None; }

class DistributedDomain {
private:
  Dim3 size_;

  int rank_;
  int worldSize_;

  // the GPUs this distributed domain will use
  std::vector<int> gpus_;

  // MPI-related topology information
  MpiTopology mpiTopology_;
  Machine machine_;

  Placement *placement_;

  // the stencil radius in each direction
  Radius radius_;

  // typically one per GPU
  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;
  // the index of the domain in the distributed domain
  std::vector<Dim3> domainIdx_;

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

  // the names of each quantity
  std::vector<std::string> dataName_;

  MethodFlags flags_;
  PlacementStrategy strategy_;

  // PeerCopySenders for same-rank exchanges
  std::vector<std::map<size_t, PeerCopySender>> peerCopySenders_;

  std::vector<std::map<Dim3, StatefulSender *>> remoteSenders_; // remoteSender_[domain][dstIdx] = sender
  std::vector<std::map<Dim3, StatefulRecver *>> remoteRecvers_; // remoteRecver_[domain][srcIdx] = recver

  // kernel sender for same-domain sends
  PeerAccessSender peerAccessSender_;

  std::vector<std::map<Dim3, ColocatedHaloSender>> coloSenders_; // vec[domain][dstIdx] = sender
  std::vector<std::map<Dim3, ColocatedHaloRecver>> coloRecvers_;

#ifdef STENCIL_SETUP_STATS
  // count of how many bytes are sent through various methods in each exchange
  uint64_t numBytesCudaMpi_;
  uint64_t numBytesCudaMpiColocated_;
  uint64_t numBytesCudaMemcpyPeer_;
  uint64_t numBytesCudaKernel_;
#endif

public:
#ifdef STENCIL_EXCHANGE_STATS
  /* record total time spent on operations. Valid at MPI rank 0
   */
  double timeExchange_;
  double timeSwap_;
#endif

#ifdef STENCIL_SETUP_STATS
  /* total time spent on setup ops*/
  double timeMpiTopo_;
  double timePeerEn_;
  double timePlacement_;
  double timePlan_;
  double timeRealize_;
  double timeCreate_;
#endif

  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), placement_(nullptr), flags_(MethodFlags::All), strategy_(PlacementStrategy::NodeAware) {

#ifdef STENCIL_SETUP_STATS
    timeMpiTopo_ = 0;
    timePeerEn_ = 0;
    timePlacement_ = 0;
    timePlan_ = 0;
    timeRealize_ = 0;
    timeCreate_ = 0;
#endif

#ifdef STENCIL_EXCHANGE_STATS
    timeExchange_ = 0;
    timeSwap_ = 0;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

#ifdef STENCIL_SETUP_STATS
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif
    mpiTopology_ = std::move(MpiTopology(MPI_COMM_WORLD));
    machine_ = Machine::build(MPI_COMM_WORLD);
#ifdef STENCIL_SETUP_STATS
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = -1;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timeMpiTopo_ += maxElapsed;
    }
#endif

    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));
    std::cerr << "[" << rank_ << "] cudaGetDeviceCount= " << deviceCount << "\n";

    /*
  cudaComputeModeDefault = 0
      Default compute mode (Multiple threads can use cudaSetDevice() with this
  device) cudaComputeModeExclusive = 1 Compute-exclusive-thread mode (Only one
  thread in one process will be able to use cudaSetDevice() with this device)
  cudaComputeModeProhibited = 2
      Compute-prohibited mode (No threads can use cudaSetDevice() with this
  device) cudaComputeModeExclusiveProcess = 3 Compute-exclusive-process mode
  (Many threads in one process will be able to use cudaSetDevice() with this
  device)
    */
    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; ++i) {
      CUDA_RUNTIME(cudaGetDeviceProperties(&prop, i));
      LOG_INFO("cudaDeviceProp.computeMode=" << prop.computeMode);
      if (prop.computeMode != 0) {
        // TODO: once we can limit to one rank per exlcusive GPU, we can lift this
        LOG_FATAL("need cudaComputeModeDefault");
      }
    }

    // get the ranks on this node and the GPUs that are visible to ranks on this node
    const std::vector<int> coloRanks = machine_.colocated_ranks(rank_);
    LOG_INFO("colocated with " << coloRanks.size() << " ranks");
    const int node = machine_.node_of_rank(rank_);
    const std::vector<GPU::index_t> coloGPUs = machine_.gpus_of_node(node);
    LOG_DEBUG(coloGPUs.size() << " GPUs on node");

    const int gpusPerRank = coloGPUs.size() / coloRanks.size();

    // if there are more ranks than GPUs, distribute ranks among GPUs
    std::map<int, std::vector<GPU::index_t>> rankGPUs;
    if (coloRanks.size() > coloGPUs.size()) {
      // max number of ranks using each GPU
      // TODO: if exclusive compute mode, then set use limit to 1
      // TODO: track use-limit per GPU? I guess it's possible some are exclusive and some are not
      const int useLimit = coloRanks.size() / coloGPUs.size();
      LOG_DEBUG("max " << useLimit << " ranks using each GPU");

      // how many ranks are using each GPU
      std::map<GPU::index_t, int> users;
      for (GPU::index_t gi : coloGPUs) {
        users[gi] = 0;
      }

      for (auto rank : coloRanks) {
        bool found = false;
        // grab visible GPUs that are still available
        for (auto gi : machine_.gpus_of_rank(rank)) {
          if (users[gi] < useLimit) {
            found = true;
            users[gi] += 1;
            rankGPUs[rank].push_back(gi);
            LOG_INFO("assign GPU" << gi << " to rank " << rank);
            break;
          }
        }
        if (!found) {
          LOG_FATAL("unable to find a GPU");
        }
      }
    }
    // if there are >= GPUs as ranks, go through ranks until all GPUs have been claimed
    else {
      LOG_DEBUG("distribute ");
      std::set<GPU::index_t> available(coloGPUs.begin(), coloGPUs.end());
      for (int i = 0; !available.empty(); ++i) {
        int rank = coloRanks[i % coloRanks.size()];
        bool found = false;
        // grab a visible GPU that is still available
        for (auto gi : machine_.gpus_of_rank(rank)) {
          if (available.count(gi)) {
            found = true;
            available.erase(gi);
            rankGPUs[rank].push_back(gi);
            LOG_INFO("assign GPU " << gi << " to rank " << rank);
            break;
          }
        }
        if (!found) {
          LOG_FATAL("there were GPUs left, but a rank counldn't claim one");
        }
      }
    }

    gpus_.insert(gpus_.end(), rankGPUs[rank_].begin(), rankGPUs[rank_].end());
    for (GPU::index_t gi : gpus_) {
      LOG_INFO("using GPU " << gi);
    }

    if (gpus_.empty()) {
      LOG_FATAL("selected no GPUs");
    }

#ifdef STENCIL_SETUP_STATS
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    // Try to enable peer access between visible GPUs
    nvtxRangePush("peer_en");
    for (int src = 0; src < deviceCount; ++src) {
      for (int dst = 0; dst < deviceCount; ++dst) {
        gpu_topo::enable_peer(src, dst);
      }
    }
    nvtxRangePop();
#ifdef STENCIL_SETUP_STATS
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timePeerEn_ += maxElapsed;
    }
#endif
    CUDA_RUNTIME(cudaGetLastError());
  }

  ~DistributedDomain() {
    for (auto &m : remoteSenders_) {
      for (auto &kv : m) {
        delete kv.second;
      }
    }
    for (auto &m : remoteRecvers_) {
      for (auto &kv : m) {
        delete kv.second;
      }
    }
  }

  const Dim3 &size() const noexcept { return size_; }
  std::vector<LocalDomain> &domains() noexcept { return domains_; }
  const std::vector<LocalDomain> &domains() const noexcept { return domains_; }

  /* set the radius in all directions to r
   */
  void set_radius(size_t r) noexcept { radius_ = Radius::constant(r); }

  void set_radius(const Radius &r) noexcept { radius_ = r; }

  template <typename T> DataHandle<T> add_data(const std::string &name = "") {
    dataElemSize_.push_back(sizeof(T));
    dataName_.push_back(name);
    return DataHandle<T>(dataElemSize_.size() - 1, name);
  }

  /* Choose comm methods from MethodFlags. Call before realize()

    d.set_methods(MethodFlags::Any);
    d.set_methods(MethodFlags::CudaAwareMpi | MethodFlags::Kernel);
  */
  void set_methods(MethodFlags flags) noexcept { flags_ = flags; }

  /* set the placement method.

  Call before realize()
  Should be called by all ranks with the same parameters.
  */
  void set_placement(PlacementStrategy strategy) noexcept { strategy_ = strategy; }

  /*! return true if any provided methods are enabled
   */
  bool any_methods(MethodFlags methods) const noexcept { return (methods & flags_) != MethodFlags::None; }

  /* Choose GPUs for this rank. Call before realize()
   */
  void set_gpus(const std::vector<int> &cudaIds) { gpus_ = cudaIds; }

  /* return the coordinate in the domain that subdomain i's interior starts at
   */
  const Dim3 &get_origin(int64_t i) const { return domains_[i].origin(); }

  /* return the compute region of the entire distributed domain
   */
  const Rect3 get_compute_region() const noexcept;

  /* return the total number of bytes moved during the halo exchange for `method`
  ( after realize() )
  */
  uint64_t exchange_bytes_for_method(const MethodFlags &method) const;

  /* Initialize resources for a previously-configured domain.
  (before exchange())
  */
  void realize();

  /* Swap current and next pointers
   */
  void swap();

  /* Return the coordinates of the stencil region that can be safely operated on
   * during exchange
   * One vector per LocalDomain
   */
  std::vector<Rect3> get_interior() const;

  /* Return the coordinates of the stencil regions that CANNOT be safely operated on during
   * exchange.
   * The GPU kernel can modify this data when the exchange is no occuring
   * One vector per LocalDomain
   */
  std::vector<std::vector<Rect3>> get_exterior() const;

  /*!
  Do a halo exchange of the "current" quantities and return
  */
  void exchange();

  /* Dump distributed domain to a series of paraview files

     The files are named prefixN.txt, where N is a unique number for each
     subdomain `zero_nans` causes nans to be replaced with 0.0
  */
  void write_paraview(const std::string &prefix, bool zeroNaNs = false);
};
