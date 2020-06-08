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
#include "stencil/mpi_topology.hpp"
#include "stencil/nvml.hpp"
#include "stencil/partition.hpp"
#include "stencil/radius.hpp"
#include "stencil/tx.hpp"

#ifndef STENCIL_OUTPUT_LEVEL
#define STENCIL_OUTPUT_LEVEL 0
#endif

#if STENCIL_OUTPUT_LEVEL <= 0
#define LOG_SPEW(x)                                                            \
  std::cerr << "SPEW[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_SPEW(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 1
#define LOG_DEBUG(x)                                                           \
  std::cerr << "DEBUG[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_DEBUG(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 2
#define LOG_INFO(x)                                                            \
  std::cerr << "INFO[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_INFO(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 3
#define LOG_WARN(x)                                                            \
  std::cerr << "WARN[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_WARN(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 4
#define LOG_ERROR(x)                                                           \
  std::cerr << "ERROR[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_ERROR(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 5
#define LOG_FATAL(x)                                                           \
  std::cerr << "FATAL[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";   \
  exit(1);
#else
#define LOG_FATAL(x) exit(1);
#endif

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

inline bool operator&&(MethodFlags a, MethodFlags b) {
  return (a & b) != MethodFlags::None;
}

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

  std::vector<std::map<Dim3, StatefulSender *>>
      remoteSenders_; // remoteSender_[domain][dstIdx] = sender
  std::vector<std::map<Dim3, StatefulRecver *>>
      remoteRecvers_; // remoteRecver_[domain][srcIdx] = recver

  // kernel sender for same-domain sends
  PeerAccessSender peerAccessSender_;

  std::vector<std::map<Dim3, ColocatedHaloSender>>
      coloSenders_; // vec[domain][dstIdx] = sender
  std::vector<std::map<Dim3, ColocatedHaloRecver>> coloRecvers_;

public:
#if STENCIL_TIME == 1
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
#endif

  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), placement_(nullptr), flags_(MethodFlags::All),
        strategy_(PlacementStrategy::NodeAware) {

#if STENCIL_TIME == 1
    timeMpiTopo_ = 0;
    timeNodeGpus_ = 0;
    timePeerEn_ = 0;
    timePlacement_ = 0;
    timePlan_ = 0;
    timeRealize_ = 0;
    timeCreate_ = 0;
    timeExchange_ = 0;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif
    mpiTopology_ = std::move(MpiTopology(MPI_COMM_WORLD));
#if STENCIL_TIME == 1
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = -1;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timeMpiTopo_ += maxElapsed;
    }
#endif

    std::cerr << "[" << rank_ << "] colocated with "
              << mpiTopology_.colocated_size() << " ranks\n";

    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));
    std::cerr << "[" << rank_ << "] cudaGetDeviceCount= " << deviceCount
              << "\n";

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
      std::cerr << "[" << rank_
                << "] cudaDeviceProp.computeMode=" << prop.computeMode << "\n";
    }

    // Determine GPUs this DistributedDomain is reposible for
    if (gpus_.empty()) {
      // if fewer colocated ranks than GPUs, round-robin GPUs to ranks
      if (mpiTopology_.colocated_size() <= deviceCount) {
        for (int id = 0; id < deviceCount; ++id) {
          if (id % mpiTopology_.colocated_size() ==
              mpiTopology_.colocated_rank()) {
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
#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    std::vector<int> nodeCudaIds(gpus_.size() * mpiTopology_.colocated_size());
    MPI_Allgather(gpus_.data(), int(gpus_.size()), MPI_INT, nodeCudaIds.data(),
                  int(gpus_.size()), MPI_INT, mpiTopology_.colocated_comm());
#if STENCIL_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timeNodeGpus_ += maxElapsed;
    }
#endif
    {
      std::set<int> unique(nodeCudaIds.begin(), nodeCudaIds.end());
      // nodeCudaIds = std::vector<int>(unique.begin(), unique.end());
      std::cerr << "[" << rank_ << "] colocated with ranks using gpus";
      for (auto &e : nodeCudaIds) {
        std::cerr << " " << e;
      }
      std::cerr << "\n";
    }

#if STENCIL_TIME == 1
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
#if STENCIL_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
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
  std::vector<LocalDomain> &domains() { return domains_; }

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

  void set_placement(PlacementStrategy strategy) noexcept {
    strategy_ = strategy;
  }

  /*! return true if any provided methods are enabled
   */
  bool any_methods(MethodFlags methods) const noexcept {
    return (methods & flags_) != MethodFlags::None;
  }

  /* Choose GPUs for this rank. Call before realize()
   */
  void set_gpus(const std::vector<int> &cudaIds) { gpus_ = cudaIds; }

  /* return the coordinate in the domain that subdomain i's interior starts at
   */
  const Dim3 &get_origin(int64_t i) const { return domains_[i].origin(); }

  void realize() {
    CUDA_RUNTIME(cudaGetLastError());

    // TODO: make sure everyone has the same Placement Strategy

    // compute domain placement
#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif
    nvtxRangePush("placement");
    if (strategy_ == PlacementStrategy::NodeAware) {
      assert(!placement_);
      placement_ = new NodeAware(size_, mpiTopology_, radius_, gpus_);
    } else {
      assert(!placement_);
      placement_ = new Trivial(size_, mpiTopology_, gpus_);
    }
    assert(placement_);
    nvtxRangePop();
    CUDA_RUNTIME(cudaGetLastError());
#if STENCIL_TIME == 1
    double maxElapsed = -1;
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timePlacement_ += maxElapsed;
    }
#endif

#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    CUDA_RUNTIME(cudaGetLastError());
    for (int64_t domId = 0; domId < int64_t(gpus_.size()); domId++) {

      const Dim3 idx = placement_->get_idx(rank_, domId);
      const Dim3 sdSize = placement_->subdomain_size(idx);
      const Dim3 sdOrigin = placement_->subdomain_origin(idx);

      // placement algorithm should agree with me what my GPU is
      assert(placement_->get_cuda(idx) == gpus_[domId]);

      const int cudaId = placement_->get_cuda(idx);

      fprintf(stderr, "rank=%d gpu=%ld (cuda id=%d) => [%ld,%ld,%ld]\n", rank_,
              domId, cudaId, idx.x, idx.y, idx.z);

      LocalDomain sd(sdSize, sdOrigin, cudaId);
      sd.set_radius(radius_);
      for (size_t dataIdx = 0; dataIdx < dataElemSize_.size(); ++dataIdx) {
        sd.add_data(dataElemSize_[dataIdx]);
      }

      domains_.push_back(sd);
    }
    CUDA_RUNTIME(cudaGetLastError());
    // realize local domains
    for (auto &d : domains_) {
      d.realize();
    }
    CUDA_RUNTIME(cudaGetLastError());
#if STENCIL_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timeRealize_ += maxElapsed;
    }
#endif

#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif

    // outbox for same-GPU exchanges
    std::vector<Message> peerAccessOutbox;

    // outboxes for same-rank exchanges
    std::vector<std::vector<std::vector<Message>>> peerCopyOutboxes;
    // peerCopyOutboxes[di][dj] = peer copy from di to dj

    // outbox for co-located domains in different ranks
    // one outbox for each co-located domain
    std::vector<std::map<Dim3, std::vector<Message>>> coloOutboxes;
    std::vector<std::map<Dim3, std::vector<Message>>> coloInboxes;
    // coloOutboxed[di][dstRank] = messages

    // inbox for each remote domain my domains recv from
    std::vector<std::map<Dim3, std::vector<Message>>>
        remoteInboxes; // remoteOutboxes_[domain][srcIdx] = messages
    // outbox for each remote domain my domains send to
    std::vector<std::map<Dim3, std::vector<Message>>>
        remoteOutboxes; // remoteOutboxes[domain][dstIdx] = messages

    std::cerr << "comm plan\n";
    // plan messages
    /*  
    For each direction, look up where the destination device is and decide which communication
    method to use.
    We do not create a message where the message size would be zero
    */
    nvtxRangePush("DistributedDomain::realize() plan messages");
    peerCopyOutboxes.resize(gpus_.size());
    for (auto &v : peerCopyOutboxes) {
      v.resize(gpus_.size());
    }
    coloOutboxes.resize(gpus_.size());
    coloInboxes.resize(gpus_.size());
    remoteOutboxes.resize(gpus_.size());
    remoteInboxes.resize(gpus_.size());

    const Dim3 globalDim = placement_->dim();

    for (size_t di = 0; di < domains_.size(); ++di) {
      const Dim3 myIdx = placement_->get_idx(rank_, di);
      const int myDev = domains_[di].gpu();
      assert(myDev == placement_->get_cuda(myIdx));
      for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
          for (int x = -1; x <= 1; ++x) {
            const Dim3 dir(x, y, z);
            if (Dim3(0, 0, 0) == dir) {
              continue; // no message
            }

            // Only send to do sends when the stencil radius in the opposite
            // direction is non-zero for example, if +x radius is 2, our -x
            // neighbor needs a halo region from us, so we need to plan to send
            // in that direction
            if (0 == radius_.dir(dir * -1)) {
              continue; // no sends or recvs for this dir
            }

            // TODO: this assumes we have periodic boundaries
            // we can filter out some messages here if we do not
            const Dim3 dstIdx = (myIdx + dir).wrap(globalDim);
            const int dstRank = placement_->get_rank(dstIdx);
            const int dstGPU = placement_->get_subdomain_id(dstIdx);
            const int dstDev = placement_->get_cuda(dstIdx);
            Message sMsg(dir, di, dstGPU);

            if (any_methods(MethodFlags::CudaKernel)) {
              if (dstRank == rank_ && myDev == dstDev) {
                peerAccessOutbox.push_back(sMsg);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMemcpyPeer)) {
              if (dstRank == rank_ && gpu_topo::peer(myDev, dstDev)) {
                peerCopyOutboxes[di][dstGPU].push_back(sMsg);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpiColocated)) {
              if ((dstRank != rank_) && mpiTopology_.colocated(dstRank) &&
                  gpu_topo::peer(myDev, dstDev)) {
                assert(di < coloOutboxes.size());
                coloOutboxes[di].emplace(dstIdx, std::vector<Message>());
                coloOutboxes[di][dstIdx].push_back(sMsg);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpi | MethodFlags::CudaAwareMpi)) {
              assert(di < remoteOutboxes.size());
              remoteOutboxes[di][dstIdx].push_back(sMsg);
              LOG_SPEW("Plan send <remote> "
                       << myIdx << " (r" << rank_ << "d" << di << "g" << myDev
                       << ")"
                       << " -> " << dstIdx << " (r" << dstRank << "d" << dstGPU
                       << "g" << dstDev << ")"
                       << " (dir=" << dir << ", rad" << dir * -1 << "="
                       << radius_.dir(dir * -1) << ")");
              goto send_planned;
            }
            LOG_FATAL("No method available to send required message "
                      << sMsg.dir_ << "\n");
          send_planned: // successfully found a way to send

            // TODO: this assumes we have periodic boundaries
            // we can filter out some messages here if we do not
            const Dim3 srcIdx = (myIdx - dir).wrap(globalDim);
            const int srcRank = placement_->get_rank(srcIdx);
            const int srcGPU = placement_->get_subdomain_id(srcIdx);
            const int srcDev = placement_->get_cuda(srcIdx);
            Message rMsg(dir, srcGPU, di);

            if (any_methods(MethodFlags::CudaKernel)) {
              if (srcRank == rank_ && srcDev == myDev) {
                // no recver needed
                goto recv_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMemcpyPeer)) {
              if (srcRank == rank_ && gpu_topo::peer(srcDev, myDev)) {
                // no recver needed
                goto recv_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpiColocated)) {
              if ((srcRank != rank_) && mpiTopology_.colocated(srcRank) &&
                  gpu_topo::peer(srcDev, myDev)) {
                assert(di < coloInboxes.size());
                coloInboxes[di].emplace(srcIdx, std::vector<Message>());
                coloInboxes[di][srcIdx].push_back(sMsg);
                goto recv_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpi | MethodFlags::CudaAwareMpi)) {
              assert(di < remoteInboxes.size());
              remoteInboxes[di].emplace(srcIdx, std::vector<Message>());
              remoteInboxes[di][srcIdx].push_back(sMsg);
              LOG_SPEW("Plan recv <remote> "
                       << srcIdx << "->" << myIdx << " (dir=" << dir << "): r"
                       << dir * -1 << "=" << radius_.dir(dir * -1));
              goto recv_planned;
            }
            LOG_FATAL("No method available to recv required message");
          recv_planned: // found a way to recv
            (void)0;
          }
        }
      }
    }

    nvtxRangePop(); // plan
#if STENCIL_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timePlan_ += maxElapsed;
    }
#endif

    // summarize communication plan
    std::string planFileName = "plan_" + std::to_string(rank_) + ".txt";
    std::ofstream planFile(planFileName, std::ofstream::out);

    planFile << "rank=" << rank_ << "\n\n";

    planFile << "domains\n";
    for (size_t di = 0; di < domains_.size(); ++di) {
      planFile << di << ":cuda" << domains_[di].gpu() << ":"
               << placement_->get_idx(rank_, di)
               << " sz=" << domains_[di].size() << "\n";
    }
    planFile << "\n";

    planFile << "== peerAccess ==\n";
    for (auto &m : peerAccessOutbox) {
      size_t numBytes = 0;
      for (int qi = 0; qi < domains_[m.srcGPU_].num_data(); ++qi) {
        numBytes += domains_[m.srcGPU_].halo_bytes(m.dir_, qi);
      }
      planFile << m.srcGPU_ << "->" << m.dstGPU_ << " " << m.dir_ << " "
               << numBytes << "B\n";
    }
    planFile << "\n";

    planFile << "== peerCopy ==\n";
    for (size_t srcGPU = 0; srcGPU < peerCopyOutboxes.size(); ++srcGPU) {
      for (size_t dstGPU = 0; dstGPU < peerCopyOutboxes[srcGPU].size();
           ++dstGPU) {
        size_t numBytes = 0;
        for (const auto &msg : peerCopyOutboxes[srcGPU][dstGPU]) {
          for (int64_t i = 0; i < domains_[srcGPU].num_data(); ++i) {
            int64_t haloBytes = domains_[srcGPU].halo_bytes(msg.dir_, i);
            numBytes += haloBytes;
          }
          planFile << srcGPU << "->" << dstGPU << " " << msg.dir_ << " "
                   << numBytes << "B\n";
        }
      }
    }
    planFile << "\n";

    planFile << "== colo ==\n";
    for (auto &obxs : coloOutboxes) {
      for (auto &kv : obxs) {
        const Dim3 dstIdx = kv.first;
        auto &box = kv.second;
        planFile << "colo to dstIdx=" << dstIdx << "\n";
        for (auto &m : box) {
          planFile << "dir=" << m.dir_ << " (" << m.srcGPU_ << "->" << m.dstGPU_
                   << ")\n";
        }
      }
    }
    planFile << "\n";

    planFile << "== remote ==\n";
    for (auto &obxs : remoteOutboxes) {
      for (auto &kv : obxs) {
        const Dim3 dstIdx = kv.first;
        auto &box = kv.second;
        planFile << "remote to dstIdx=" << dstIdx << "\n";
        for (auto &m : box) {
          planFile << "dir=" << m.dir_ << " (" << m.srcGPU_ << "->" << m.dstGPU_
                   << ")\n";
        }
      }
    }
    planFile.close();

#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    // create remote sender/recvers
    std::cerr << "create remote\n";
    nvtxRangePush("DistributedDomain::realize: create remote");
    // per-domain senders and messages
    remoteSenders_.resize(gpus_.size());
    remoteRecvers_.resize(gpus_.size());

    // create all required remote senders/recvers
    for (size_t di = 0; di < domains_.size(); ++di) {
      for (auto &kv : remoteOutboxes[di]) {
        const Dim3 dstIdx = kv.first;
        const int dstRank = placement_->get_rank(dstIdx);
        const int dstGPU = placement_->get_subdomain_id(dstIdx);
        if (0 == remoteSenders_[di].count(dstIdx)) {
          StatefulSender *sender = nullptr;
          if (any_methods(MethodFlags::CudaAwareMpi)) {
            sender = new CudaAwareMpiSender(rank_, di, dstRank, dstGPU,
                                            domains_[di]);
          } else if (any_methods(MethodFlags::CudaMpi)) {
            sender = new RemoteSender(rank_, di, dstRank, dstGPU, domains_[di]);
          }
          assert(sender);
          remoteSenders_[di].emplace(dstIdx, sender);
        }
      }
      for (auto &kv : remoteInboxes[di]) {
        const Dim3 srcIdx = kv.first;
        const int srcRank = placement_->get_rank(srcIdx);
        const int srcGPU = placement_->get_subdomain_id(srcIdx);
        if (0 == remoteRecvers_[di].count(srcIdx)) {
          StatefulRecver *recver = nullptr;
          if (any_methods(MethodFlags::CudaAwareMpi)) {
            recver = new CudaAwareMpiRecver(srcRank, srcGPU, rank_, di,
                                            domains_[di]);
          } else if (any_methods(MethodFlags::CudaMpi)) {
            recver = new RemoteRecver(srcRank, srcGPU, rank_, di, domains_[di]);
          }
          assert(recver);
          remoteRecvers_[di].emplace(srcIdx, recver);
        }
      }
    }
    nvtxRangePop(); // create remote

    std::cerr << "create colocated\n";
    // create colocated sender/recvers
    nvtxRangePush("DistributedDomain::realize: create colocated");
    // per-domain senders and messages
    coloSenders_.resize(gpus_.size());
    coloRecvers_.resize(gpus_.size());

    // create all required colocated senders/recvers
    for (size_t di = 0; di < domains_.size(); ++di) {
      for (auto &kv : coloOutboxes[di]) {
        const Dim3 dstIdx = kv.first;
        const int dstRank = placement_->get_rank(dstIdx);
        const int dstGPU = placement_->get_subdomain_id(dstIdx);
        std::cerr << "rank " << rank_ << " create ColoSender to " << dstIdx
                  << " on " << dstRank << " (" << dstGPU << ")\n";
        coloSenders_[di].emplace(
            dstIdx,
            ColocatedHaloSender(rank_, di, dstRank, dstGPU, domains_[di]));
      }
      for (auto &kv : coloInboxes[di]) {
        const Dim3 srcIdx = kv.first;
        const int srcRank = placement_->get_rank(srcIdx);
        const int srcGPU = placement_->get_subdomain_id(srcIdx);
        std::cerr << "rank " << rank_ << " create ColoRecver from " << srcIdx
                  << " on " << srcRank << " (" << srcGPU << ")\n";
        coloRecvers_[di].emplace(
            srcIdx,
            ColocatedHaloRecver(srcRank, srcGPU, rank_, di, domains_[di]));
      }
    }
    nvtxRangePop(); // create colocated

    std::cerr << "create peer copy\n";
    // create colocated sender/recvers
    nvtxRangePush("DistributedDomain::realize: create PeerCopySender");
    // per-domain senders and messages
    peerCopySenders_.resize(gpus_.size());

    // create all required colocated senders/recvers
    for (size_t srcGPU = 0; srcGPU < peerCopyOutboxes.size(); ++srcGPU) {
      for (size_t dstGPU = 0; dstGPU < peerCopyOutboxes[srcGPU].size();
           ++dstGPU) {
        if (!peerCopyOutboxes[srcGPU][dstGPU].empty()) {
          peerCopySenders_[srcGPU].emplace(
              dstGPU, PeerCopySender(srcGPU, dstGPU, domains_[srcGPU],
                                     domains_[dstGPU]));
        }
      }
    }
    nvtxRangePop(); // create peer copy

    // prepare senders and receivers
    std::cerr << "DistributedDomain::realize: prepare PeerAccessSender\n";
    nvtxRangePush("DistributedDomain::realize: prep peerAccessSender");
    peerAccessSender_.prepare(peerAccessOutbox, domains_);
    nvtxRangePop();
    std::cerr << "DistributedDomain::realize: prepare PeerCopySender\n";
    nvtxRangePush("DistributedDomain::realize: prep peerCopySender");
    for (size_t srcGPU = 0; srcGPU < peerCopySenders_.size(); ++srcGPU) {
      for (auto &kv : peerCopySenders_[srcGPU]) {
        const int dstGPU = kv.first;
        auto &sender = kv.second;
        sender.prepare(peerCopyOutboxes[srcGPU][dstGPU]);
      }
    }
    nvtxRangePop();
    std::cerr << "DistributedDomain::realize: start_prepare "
                 "ColocatedHaloSender/ColocatedHaloRecver\n";
    nvtxRangePush("DistributedDomain::realize: prep colocated");
    assert(coloSenders_.size() == coloRecvers_.size());
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 srcIdx = placement_->get_idx(rank_, di);
        const Dim3 dstIdx = kv.first;
        const int dstRank = placement_->get_rank(dstIdx);
        auto &sender = kv.second;
        std::cerr << "rank=" << rank_ << " colo sender.start_prepare " << srcIdx
                  << "->" << dstIdx << "(rank " << dstRank << ")\n";
        sender.start_prepare(coloOutboxes[di][dstIdx]);
      }
      for (auto &kv : coloRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        const Dim3 dstIdx = placement_->get_idx(rank_, di);
        auto &recver = kv.second;
        std::cerr << "rank=" << rank_ << " colo recver.start_prepare " << srcIdx
                  << "->" << dstIdx << "\n";
        recver.start_prepare(coloInboxes[di][srcIdx]);
      }
    }
    std::cerr << "rank=" << rank_
              << " DistributedDomain::realize: finish_prepare "
                 "ColocatedHaloSender/ColocatedHaloRecver\n";
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        const int srcDev = domains_[di].gpu();
        const Dim3 srcIdx = placement_->get_idx(rank_, di);
        const int dstDev = placement_->get_cuda(dstIdx);
        std::cerr << "rank=" << rank_ << " colo sender.finish_prepare "
                  << srcIdx << " -> " << dstIdx << "\n";
        sender.finish_prepare();
      }
      for (auto &kv : coloRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        auto &recver = kv.second;
        std::cerr << "rank=" << rank_
                  << " colo recver.finish_prepare for colo from " << srcIdx
                  << "\n";
        recver.finish_prepare();
      }
    }
    nvtxRangePop(); // prep colocated
    std::cerr
        << "rank=" << rank_
        << "DistributedDomain::realize: prepare RemoteSender/RemoteRecver\n";
    nvtxRangePush("DistributedDomain::realize: prep remote");
    assert(remoteSenders_.size() == remoteRecvers_.size());
    for (size_t di = 0; di < remoteSenders_.size(); ++di) {
      for (auto &kv : remoteSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        sender->prepare(remoteOutboxes[di][dstIdx]);
      }
      for (auto &kv : remoteRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        auto &recver = kv.second;
        recver->prepare(remoteInboxes[di][srcIdx]);
      }
    }
    nvtxRangePop(); // prep remote

#if STENCIL_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timeCreate_ += maxElapsed;
    }
#endif
  }

  /* Swap current and next pointers
   */
  void swap() {
    for (auto &d : domains_) {
      d.swap();
    }
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {

#if STENCIL_TIME == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif

    /*! Try to start sends in order from longest to shortest
     * we expect remote to be longest, followed by peer copy, followed by colo
     * colo is shorter than peer copy due to the node-aware data placement:
     * if we try to place bigger exchanges nearby, they will be faster
     */

    // start remote send d2h
    LOG_DEBUG("[" << rank_ << "] remote send start");
    nvtxRangePush("DD::exchange: remote send d2h");
    for (auto &domSenders : remoteSenders_) {
      for (auto &kv : domSenders) {
        StatefulSender *sender = kv.second;
        sender->send();
      }
    }
    nvtxRangePop();

    // send same-rank messages
#if STENCIL_LOUD == 1
    fprintf(stderr, "rank=%d send peer copy\n", rank_);
#endif
    nvtxRangePush("DD::exchange: peer copy send");
    for (auto &src : peerCopySenders_) {
      for (auto &kv : src) {
        PeerCopySender &sender = kv.second;
        sender.send();
      }
    }
    nvtxRangePop();

    // start colocated Senders
#if STENCIL_LOUD == 1
    fprintf(stderr, "rank=%d start colo send\n", rank_);
#endif
    nvtxRangePush("DD::exchange: colo send");
    for (auto &domSenders : coloSenders_) {
      for (auto &kv : domSenders) {
        ColocatedHaloSender &sender = kv.second;
        sender.send();
      }
    }
    nvtxRangePop();

    // send self messages
#if STENCIL_LOUD == 1
    fprintf(stderr, "rank=%d send peer access\n", rank_);
#endif
    nvtxRangePush("DD::exchange: peer access send");
    peerAccessSender_.send();
    nvtxRangePop();

// start colocated recvers
#if STENCIL_LOUD == 1
    fprintf(stderr, "rank=%d start colo recv\n", rank_);
#endif
    nvtxRangePush("DD::exchange: colo recv");
    for (auto &domRecvers : coloRecvers_) {
      for (auto &kv : domRecvers) {
        ColocatedHaloRecver &recver = kv.second;
        recver.recv();
      }
    }
    nvtxRangePop();

    // start remote recv h2h
    LOG_DEBUG("[" << rank_ << "] remote recv start");
    nvtxRangePush("DD::exchange: remote recv h2h");
    for (auto &domRecvers : remoteRecvers_) {
      for (auto &kv : domRecvers) {
        StatefulRecver *recver = kv.second;
        recver->recv();
      }
    }
    nvtxRangePop();

    // poll senders and recvers to move onto next step until all are done
    LOG_DEBUG("[" << rank_ << "] start poll");
    nvtxRangePush("DD::exchange: poll");
    bool pending = true;
    while (pending) {
      pending = false;
    recvers:
      // move recvers from h2h to h2d
      for (auto &domRecvers : remoteRecvers_) {
        for (auto &kv : domRecvers) {
          StatefulRecver *recver = kv.second;
          if (recver->active()) {
            pending = true;
            if (recver->next_ready()) {
              // const Dim3 srcIdx = kv.first;
              // std::cerr << "[" << rank_ << "] src=" << srcIdx << "
              // recv_h2d\n";
              recver->next();
              goto senders; // try to overlap recv_h2d with send_h2h
            }
          }
        }
      }
    senders:
      // move senders from d2h to h2h
      for (auto &domSenders : remoteSenders_) {
        for (auto &kv : domSenders) {
          StatefulSender *sender = kv.second;
          if (sender->active()) {
            pending = true;
            if (sender->next_ready()) {
              // const Dim3 dstIdx = kv.first;
              // std::cerr << "[" << rank_ << "] dst=" << dstIdx << "
              // send_h2h\n";
              sender->next();
              goto recvers; // try to overlap recv_h2d with send_h2h
            }
          }
        }
      }
    }
    nvtxRangePop(); // DD::exchange: poll

    // wait for sends
    LOG_DEBUG("[" << rank_ << "] wait for peer access senders");
    nvtxRangePush("peerAccessSender.wait()");
    peerAccessSender_.wait();
    nvtxRangePop();

    nvtxRangePush("peerCopySender.wait()");
    for (auto &src : peerCopySenders_) {
      for (auto &kv : src) {
        PeerCopySender &sender = kv.second;
        sender.wait();
      }
    }
    nvtxRangePop(); // peerCopySender.wait()

    // wait for colocated
#if STENCIL_LOUD == 1
    std::cerr << "colocated senders wait\n";
#endif
    nvtxRangePush("colocated.wait()");
    for (auto &domSenders : coloSenders_) {
      for (auto &kv : domSenders) {
        ColocatedHaloSender &sender = kv.second;
        sender.wait();
      }
    }
#if STENCIL_LOUD == 1
    std::cerr << "colocated recvers wait\n";
#endif
    for (auto &domRecvers : coloRecvers_) {
      for (auto &kv : domRecvers) {
        ColocatedHaloRecver &recver = kv.second;
        recver.wait();
      }
    }
    nvtxRangePop(); // colocated wait

    nvtxRangePush("remote wait");
    // wait for remote senders and recvers
    // printf("rank=%d wait for RemoteRecver/RemoteSender\n", rank_);
    for (auto &domRecvers : remoteRecvers_) {
      for (auto &kv : domRecvers) {
        StatefulRecver *recver = kv.second;
        recver->wait();
      }
    }
    for (auto &domSenders : remoteSenders_) {
      for (auto &kv : domSenders) {
        StatefulSender *sender = kv.second;
        sender->wait();
      }
    }
    nvtxRangePop(); // remote wait

#if STENCIL_TIME == 1
    double maxElapsed = -1;
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      timeExchange_ += maxElapsed;
    }
#endif

    // wait for all ranks to be done
    nvtxRangePush("barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop(); // barrier
  }

  /* Dump distributed domain to a series of paraview files

     The files are named prefixN.txt, where N is a unique number for each
     subdomain `zero_nans` causes nans to be replaced with 0.0
  */
  void write_paraview(const std::string &prefix, bool zeroNaNs = false) {

    const char delim[] = ",";

    nvtxRangePush("write_paraview");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int64_t num = rank * domains_.size();

    for (size_t di = 0; di < domains_.size(); ++di) {
      int64_t id = rank * domains_.size() + di;
      const std::string path = prefix + std::to_string(id) + ".txt";

      LocalDomain &domain = domains_[di];

      std::cerr << "write_paraview(): copy interiors to host\n";
      std::vector<std::vector<unsigned char>> quantities;
      for (int64_t qi = 0; qi < domain.num_data(); ++qi) {
        quantities.push_back(domain.interior_to_host(qi));
      }

      std::cerr << "write_paraview(): open " << path << "\n";
      FILE *outf = fopen(path.c_str(), "w");

      // column headers
      fprintf(outf, "z%sy%sx", delim, delim);
      for (int64_t qi = 0; qi < domain.num_data(); ++qi) {
        std::string colName = domain.dataName_[qi];
        if (colName.empty()) {
          colName = "data" + std::to_string(qi);
        }
        fprintf(outf, "%s%s", delim, colName.c_str());
      }
      fprintf(outf, "\n");

      const Dim3 origin = domains_[di].origin();

      // print rows
      for (int64_t lz = 0; lz < domain.sz_.z; ++lz) {
        for (int64_t ly = 0; ly < domain.sz_.y; ++ly) {
          for (int64_t lx = 0; lx < domain.sz_.x; ++lx) {
            Dim3 pos = origin + Dim3(lx, ly, lz);

            fprintf(outf, "%ld%s%ld%s%ld", pos.z, delim, pos.y, delim, pos.x);

            for (int64_t qi = 0; qi < domain.num_data(); ++qi) {
              if (8 == domain.elem_size(qi)) {
                double val = reinterpret_cast<double *>(
                    quantities[qi].data())[lz * (domain.sz_.y * domain.sz_.x) +
                                           ly * domain.sz_.x + lx];
                if (zeroNaNs && std::isnan(val)) {
                  val = 0.0;
                }
                fprintf(outf, "%s%f", delim, val);
              } else if (4 == domain.elem_size(qi)) {
                float val = reinterpret_cast<float *>(
                    quantities[qi].data())[lz * (domain.sz_.y * domain.sz_.x) +
                                           ly * domain.sz_.x + lx];
                if (zeroNaNs && std::isnan(val)) {
                  val = 0.0f;
                }
                fprintf(outf, "%s%f", delim, val);
              }
            }

            fprintf(outf, "\n");
          }
        }
      }
    }

    nvtxRangePop();
  }
};

#undef LOG_SPEW
#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_WARN
#undef LOG_ERROR
#undef LOG_FATAL