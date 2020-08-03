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
#endif

  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), placement_(nullptr), flags_(MethodFlags::All), strategy_(PlacementStrategy::NodeAware) {

#if STENCIL_MEASURE_TIME == 1
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
      std::set<int> unique(nodeCudaIds.begin(), nodeCudaIds.end());
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

  void realize() {
    CUDA_RUNTIME(cudaGetLastError());

    // TODO: make sure everyone has the same Placement Strategy

    // compute domain placement
#if STENCIL_MEASURE_TIME == 1
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
#if STENCIL_MEASURE_TIME == 1
    double maxElapsed = -1;
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timePlacement_ += maxElapsed;
    }
#endif

#if STENCIL_MEASURE_TIME == 1
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

      fprintf(stderr, "rank=%d gpu=%ld (cuda id=%d) => [%ld,%ld,%ld]\n", rank_, domId, cudaId, idx.x, idx.y, idx.z);

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
#if STENCIL_MEASURE_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timeRealize_ += maxElapsed;
    }
#endif

#if STENCIL_MEASURE_TIME == 1
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
    std::vector<std::map<Dim3, std::vector<Message>>> remoteInboxes; // remoteOutboxes_[domain][srcIdx] = messages
    // outbox for each remote domain my domains send to
    std::vector<std::map<Dim3, std::vector<Message>>> remoteOutboxes; // remoteOutboxes[domain][dstIdx] = messages

    LOG_DEBUG("[" << rank_ << "]"
                  << "comm plan");
    // plan messages
    /*
    For each direction, look up where the destination device is and decide which
    communication method to use. We do not create a message where the message
    size would be zero
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
            // send direction
            const Dim3 dir(x, y, z);
            if (Dim3(0, 0, 0) == dir) {
              continue; // no message
            }

            // Only do sends when the stencil radius in the opposite
            // direction is non-zero for example, if +x radius is 2, our -x
            // neighbor needs a halo region from us, so we need to plan to send
            // in that direction
            if (0 == radius_.dir(dir * -1)) {
              continue; // no sends or recvs for this dir
            } else {
              LOG_DEBUG(dir << " radius = " << radius_.dir(dir * -1));
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
              if ((dstRank != rank_) && mpiTopology_.colocated(dstRank) && gpu_topo::peer(myDev, dstDev)) {
                assert(di < coloOutboxes.size());
                coloOutboxes[di].emplace(dstIdx, std::vector<Message>());
                coloOutboxes[di][dstIdx].push_back(sMsg);
                LOG_DEBUG("mpi-colocated for Mesage dir=" << sMsg.dir_);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpi | MethodFlags::CudaAwareMpi)) {
              assert(di < remoteOutboxes.size());
              remoteOutboxes[di][dstIdx].push_back(sMsg);
              LOG_DEBUG("Plan send <remote> "
                        << myIdx << " (r" << rank_ << "d" << di << "g" << myDev << ")"
                        << " -> " << dstIdx << " (r" << dstRank << "d" << dstGPU << "g" << dstDev << ")"
                        << " (dir=" << dir << ", rad" << dir * -1 << "=" << radius_.dir(dir * -1) << ")");
              goto send_planned;
            }
            LOG_FATAL("No method available to send required message " << sMsg.dir_ << "\n");
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
              if ((srcRank != rank_) && mpiTopology_.colocated(srcRank) && gpu_topo::peer(srcDev, myDev)) {
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
              LOG_SPEW("Plan recv <remote> " << srcIdx << "->" << myIdx << " (dir=" << dir << "): r" << dir * -1 << "="
                                             << radius_.dir(dir * -1));
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
#if STENCIL_MEASURE_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timePlan_ += maxElapsed;
    }
#endif

    /* -------------------------
    summarize communication plan
    ----------------------------

    Dump one file per rank describing who and how we communicate
    Also total up the number of bytes we are sending for an aggregate bandwidth
    estimation.

    ----------------------------*/
    {
      sendBytes_ = 0;
      std::string planFileName = "plan_" + std::to_string(rank_) + ".txt";
      std::ofstream planFile(planFileName, std::ofstream::out);

      planFile << "rank=" << rank_ << "\n\n";

      planFile << "== quantities == \n";

      planFile << "domains\n";
      for (size_t di = 0; di < domains_.size(); ++di) {
        planFile << di << ":cuda" << domains_[di].gpu() << ":" << placement_->get_idx(rank_, di)
                 << " sz=" << domains_[di].size() << "\n";
      }
      planFile << "\n";

      planFile << "== peerAccess ==\n";
      for (auto &msg : peerAccessOutbox) {
        size_t peerBytes = 0;
        for (int qi = 0; qi < domains_[msg.srcGPU_].num_data(); ++qi) {
          // send size matches size of halo that we're recving into
          const size_t bytes = domains_[msg.srcGPU_].halo_bytes(msg.dir_ * -1, qi);
          peerBytes += bytes;
          sendBytes_ += bytes;
        }
        planFile << msg.srcGPU_ << "->" << msg.dstGPU_ << " " << msg.dir_ << " " << peerBytes << "B\n";
      }
      planFile << "\n";

      planFile << "== peerCopy ==\n";
      for (size_t srcGPU = 0; srcGPU < peerCopyOutboxes.size(); ++srcGPU) {
        for (size_t dstGPU = 0; dstGPU < peerCopyOutboxes[srcGPU].size(); ++dstGPU) {
          size_t peerBytes = 0;
          for (const auto &msg : peerCopyOutboxes[srcGPU][dstGPU]) {
            for (int64_t i = 0; i < domains_[srcGPU].num_data(); ++i) {
              // send size matches size of halo that we're recving into
              const int64_t bytes = domains_[srcGPU].halo_bytes(msg.dir_ * -1, i);
              peerBytes += bytes;
              sendBytes_ += bytes;
            }
            planFile << srcGPU << "->" << dstGPU << " " << msg.dir_ << " " << peerBytes << "B\n";
          }
        }
      }
      planFile << "\n";

      // std::vector<std::map<Dim3, std::vector<Message>>> coloOutboxes;
      planFile << "== colo ==\n";
      for (size_t di = 0; di < coloOutboxes.size(); ++di) {
        std::map<Dim3, std::vector<Message>> &obxs = coloOutboxes[di];
        for (auto &kv : obxs) {
          const Dim3 dstIdx = kv.first;
          auto &box = kv.second;
          planFile << "colo to dstIdx=" << dstIdx << "\n";
          for (auto &msg : box) {
            planFile << "dir=" << msg.dir_ << " (" << msg.srcGPU_ << "->" << msg.dstGPU_ << ")\n";
            for (int64_t i = 0; i < domains_[di].num_data(); ++i) {
              // send size matches size of halo that we're recving into
              sendBytes_ += domains_[di].halo_bytes(msg.dir_ * -1, i);
            }
          }
        }
      }
      planFile << "\n";

      planFile << "== remote ==\n";
      for (size_t di = 0; di < remoteOutboxes.size(); ++di) {
        std::map<Dim3, std::vector<Message>> &obxs = remoteOutboxes[di];
        for (auto &kv : obxs) {
          const Dim3 dstIdx = kv.first;
          auto &box = kv.second;
          planFile << "remote to dstIdx=" << dstIdx << "\n";
          for (auto &msg : box) {
            planFile << "dir=" << msg.dir_ << " (" << msg.srcGPU_ << "->" << msg.dstGPU_ << ")\n";
            for (int64_t i = 0; i < domains_[di].num_data(); ++i) {
              // send size matches size of halo that we're recving into
              sendBytes_ += domains_[di].halo_bytes(msg.dir_ * -1, i);
            }
          }
        }
      }
      planFile.close();

      // give every rank the total send volume
      nvtxRangePush("distribute send bytes to all ranks");
      MPI_Allreduce(MPI_IN_PLACE, &sendBytes_, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

      if (rank_ == 0) {
        LOG_INFO(sendBytes_ << "B in halo exchange");
      }
      nvtxRangePop();
    }

#if STENCIL_MEASURE_TIME == 1
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
            sender = new CudaAwareMpiSender(rank_, di, dstRank, dstGPU, domains_[di]);
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
            recver = new CudaAwareMpiRecver(srcRank, srcGPU, rank_, di, domains_[di]);
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
        std::cerr << "rank " << rank_ << " create ColoSender to " << dstIdx << " on " << dstRank << " (" << dstGPU
                  << ")\n";
        coloSenders_[di].emplace(dstIdx, ColocatedHaloSender(rank_, di, dstRank, dstGPU, domains_[di]));
      }
      for (auto &kv : coloInboxes[di]) {
        const Dim3 srcIdx = kv.first;
        const int srcRank = placement_->get_rank(srcIdx);
        const int srcGPU = placement_->get_subdomain_id(srcIdx);
        std::cerr << "rank " << rank_ << " create ColoRecver from " << srcIdx << " on " << srcRank << " (" << srcGPU
                  << ")\n";
        coloRecvers_[di].emplace(srcIdx, ColocatedHaloRecver(srcRank, srcGPU, rank_, di, domains_[di]));
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
      for (size_t dstGPU = 0; dstGPU < peerCopyOutboxes[srcGPU].size(); ++dstGPU) {
        if (!peerCopyOutboxes[srcGPU][dstGPU].empty()) {
          peerCopySenders_[srcGPU].emplace(dstGPU, PeerCopySender(srcGPU, dstGPU, domains_[srcGPU], domains_[dstGPU]));
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
        const Dim3 dstIdx = kv.first;
        const int dstRank = placement_->get_rank(dstIdx);
        auto &sender = kv.second;
        LOG_DEBUG(" colo sender.start_prepare " << placement_->get_idx(rank_, di) << "->" << dstIdx << "(rank "
                                                << dstRank << ")");
        sender.start_prepare(coloOutboxes[di][dstIdx]);
      }
      for (auto &kv : coloRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        auto &recver = kv.second;
        LOG_DEBUG(" colo recver.start_prepare " << srcIdx << "->" << placement_->get_idx(rank_, di));
        recver.start_prepare(coloInboxes[di][srcIdx]);
      }
    }
    LOG_DEBUG("DistributedDomain::realize: finish_prepare ColocatedHaloSender/ColocatedHaloRecver");
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        const int srcDev = domains_[di].gpu();
        const int dstDev = placement_->get_cuda(dstIdx);
        LOG_DEBUG("colo sender.finish_prepare " << placement_->get_idx(rank_, di) << " -> " << dstIdx);
        sender.finish_prepare();
      }
      for (auto &kv : coloRecvers_[di]) {
        auto &recver = kv.second;
        LOG_DEBUG("colo recver.finish_prepare for colo from " << kv.first);
        recver.finish_prepare();
      }
    }
    nvtxRangePop(); // prep colocated
    LOG_DEBUG("DistributedDomain::realize: prepare RemoteSender/RemoteRecver");
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

#if STENCIL_MEASURE_TIME == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank_) {
      timeCreate_ += maxElapsed;
    }
#endif
  }

  /* Swap current and next pointers
   */
  void swap() {
    LOG_DEBUG("enter swap()");
    for (auto &d : domains_) {
      d.swap();
    }
    LOG_DEBUG("finish swap()");
  }

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
