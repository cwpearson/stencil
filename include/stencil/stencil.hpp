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

  // total number of bytes moved during the halo exchange, set in realize()
  uint64_t sendBytes_;

public:
#if STENCIL_MEASURE_TIME == 1
  /* record total time spent on operations. Valid at MPI rank 0
   */
  double timeMpiTopo_;
  double timeNodeGpus_;
  double timePeerEn_;
  double timePlacement_;
  double timePlan_;
  double timeRealize_;
  double timeCreate_;
  double timeExchange_;
  double timeSwap_;
#endif

  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), placement_(nullptr), flags_(MethodFlags::All), strategy_(PlacementStrategy::NodeAware), sendBytes_(0) {

#if STENCIL_MEASURE_TIME == 1
    timeMpiTopo_ = 0;
    timeNodeGpus_ = 0;
    timePeerEn_ = 0;
    timePlacement_ = 0;
    timePlan_ = 0;
    timeRealize_ = 0;
    timeCreate_ = 0;
    timeExchange_ = 0;
    timeSwap_ = 0;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

#if STENCIL_MEASURE_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif
    mpiTopology_ = std::move(MpiTopology(MPI_COMM_WORLD));
#if STENCIL_MEASURE_TIME == 1
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = -1;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timeMpiTopo_ += maxElapsed;
    }
#endif

    std::cerr << "[" << rank_ << "] colocated with " << mpiTopology_.colocated_size() << " ranks\n";

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
      std::cerr << "[" << rank_ << "] cudaDeviceProp.computeMode=" << prop.computeMode << "\n";
    }

    // Determine GPUs this DistributedDomain is reposible for
    if (gpus_.empty()) {
      // if fewer colocated ranks than GPUs, round-robin GPUs to ranks
      if (mpiTopology_.colocated_size() <= deviceCount) {
        for (int id = 0; id < deviceCount; ++id) {
          if (id % mpiTopology_.colocated_size() == mpiTopology_.colocated_rank()) {
            gpus_.push_back(id);
          }
        }
      } else { // if more ranks, share gpus among ranks
        gpus_.push_back(mpiTopology_.colocated_rank() % deviceCount);
      }
    }
    assert(!gpus_.empty());

// create a list of cuda device IDs in use by the ranks on this node
// TODO: assumes all ranks use the same number of GPUs
#if STENCIL_MEASURE_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    std::vector<int> nodeCudaIds(gpus_.size() * mpiTopology_.colocated_size());
    MPI_Allgather(gpus_.data(), int(gpus_.size()), MPI_INT, nodeCudaIds.data(), int(gpus_.size()), MPI_INT,
                  mpiTopology_.colocated_comm());
#if STENCIL_MEASURE_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timeNodeGpus_ += maxElapsed;
    }
#endif
    {
      {
#if STENCIL_OUTPUT_LEVEL <= 2
        std::stringstream ss;
        ss << "[" << rank_ << "] colocated with ranks using gpus";
        for (auto &e : nodeCudaIds) {
          ss << " " << e;
        }
        LOG_INFO(ss.str());
#endif
      }
    }

#if STENCIL_MEASURE_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    // Try to enable peer access between all GPUs
    nvtxRangePush("peer_en");
    for (const auto &srcGpu : gpus_) {
      for (const auto &dstGpu : nodeCudaIds) {
        gpu_topo::enable_peer(srcGpu, dstGpu);
      }
    }
    nvtxRangePop();
#if STENCIL_MEASURE_TIME == 1
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

  /* return the total number of bytes moved during the halo exchange
  ( after realize() )
  */
  size_t halo_exchange_bytes() const { return sendBytes_; }

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

  /* total number of bytes moved during exchange
     valid after realize()
  */
  uint64_t exchanged_bytes() const noexcept { return sendBytes_; }

  /* Dump distributed domain to a series of paraview files

     The files are named prefixN.txt, where N is a unique number for each
     subdomain `zero_nans` causes nans to be replaced with 0.0
  */
  void write_paraview(const std::string &prefix, bool zeroNaNs = false);
};
