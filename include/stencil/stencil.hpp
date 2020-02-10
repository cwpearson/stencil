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
#include "stencil/tx.hpp"

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

  // GPU-related topology information
  GpuTopology gpuTopology_;

  // the stencil radius
  size_t radius_;

  // typically one per GPU
  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;
  // the index of the domain in the distributed domain
  std::vector<Dim3> domainIdx_;

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

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
  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), flags_(MethodFlags::All) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

#if STENCIL_PRINT_TIMINGS == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif
    mpiTopology_ = std::move(MpiTopology(MPI_COMM_WORLD));
#if STENCIL_PRINT_TIMINGS == 1
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = -1;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.mpi_topo %f s\n", maxElapsed);
    }
#endif

    std::cerr << "[" << rank_ << "] colocated with "
              << mpiTopology_.colocated_size() << " ranks\n";

    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));
    std::cerr << "[" << rank_ << "] cudaGetDeviceCount= " << deviceCount << "\n";

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
#if STENCIL_PRINT_TIMINGS == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    std::vector<int> nodeCudaIds(gpus_.size() * mpiTopology_.colocated_size());
    MPI_Allgather(gpus_.data(), gpus_.size(), MPI_INT, nodeCudaIds.data(),
                  gpus_.size(), MPI_INT, mpiTopology_.colocated_comm());
#if STENCIL_PRINT_TIMINGS == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.node_gpus %f s\n", maxElapsed);
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

    // determine topology info for used GPUs
#if STENCIL_PRINT_TIMINGS == 1
    for (int dev : nodeCudaIds) {
      CUDA_RUNTIME(cudaSetDevice(dev));
      CUDA_RUNTIME(cudaFree(0));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    gpuTopology_ = GpuTopology(nodeCudaIds);
#if STENCIL_PRINT_TIMINGS == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.gpu_topo %f s\n", maxElapsed);
    }
#endif

#if STENCIL_PRINT_TIMINGS == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    // Try to enable peer access between all GPUs
    nvtxRangePush("peer_en");
    gpuTopology_.enable_peer();
    nvtxRangePop();
#if STENCIL_PRINT_TIMINGS == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.peer_en %f s\n", maxElapsed);
    }
#endif
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

  std::vector<LocalDomain> &domains() { return domains_; }

  void set_radius(size_t r) { radius_ = r; }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
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

  void realize(bool useUnified = false) {

    // compute domain placement
#if STENCIL_PRINT_TIMINGS == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif
    nvtxRangePush("placement");
    Placement *placement = nullptr;
    if (strategy_ == PlacementStrategy::NodeAware) {
      placement =
          new NodeAware(size_, mpiTopology_, gpuTopology_, radius_, gpus_);
    } else {
      placement =
          new Trivial(size_, mpiTopology_, gpuTopology_, radius_, gpus_);
    }
    assert(placement);
    nvtxRangePop();
#if STENCIL_PRINT_TIMINGS == 1
    double maxElapsed = -1;
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.placement %f s\n", maxElapsed);
    }
#endif

#if STENCIL_PRINT_TIMINGS == 1
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif
    for (int domId = 0; domId < gpus_.size(); domId++) {

      const Dim3 idx = placement->get_idx(rank_, domId);
      const Dim3 sdSize = placement->subdomain_size(idx);

      // placement algorithm should agree with me what my GPU is
      assert(placement->get_cuda(idx) == gpus_[domId]);

      const int cudaId = placement->get_cuda(idx);

      fprintf(stderr, "rank=%d gpu=%d (cuda id=%d) => [%ld,%ld,%ld]\n", rank_,
              domId, cudaId, idx.x, idx.y, idx.z);

      LocalDomain sd(sdSize, cudaId);
      sd.set_radius(radius_);
      for (size_t dataIdx = 0; dataIdx < dataElemSize_.size(); ++dataIdx) {
        sd.add_data(dataElemSize_[dataIdx]);
      }

      domains_.push_back(sd);
    }
    // realize local domains
    for (auto &d : domains_) {
      d.realize();
    }
#if STENCIL_PRINT_TIMINGS == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.realize %f s\n", maxElapsed);
    }
#endif

#if STENCIL_PRINT_TIMINGS == 1
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
    nvtxRangePush("DistributedDomain::realize() plan messages");
    peerCopyOutboxes.resize(gpus_.size());
    for (auto &v : peerCopyOutboxes) {
      v.resize(gpus_.size());
    }
    coloOutboxes.resize(gpus_.size());
    coloInboxes.resize(gpus_.size());
    remoteOutboxes.resize(gpus_.size());
    remoteInboxes.resize(gpus_.size());

    const Dim3 globalDim = placement->dim();

    for (size_t di = 0; di < domains_.size(); ++di) {
      const Dim3 myIdx = placement->get_idx(rank_, di);
      const int myDev = domains_[di].gpu();
      assert(myDev == placement->get_cuda(myIdx));
      for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
          for (int x = -1; x <= 1; ++x) {
            const Dim3 dir(x, y, z);
            if (Dim3(0, 0, 0) == dir) {
              continue; // no message
            }

            const Dim3 dstIdx = (myIdx + dir).wrap(globalDim);
            const int dstRank = placement->get_rank(dstIdx);
            const int dstGPU = placement->get_subdomain_id(dstIdx);
            const int dstDev = placement->get_cuda(dstIdx);
            Message sMsg(dir, di, dstGPU);

            if (any_methods(MethodFlags::CudaKernel)) {
              if (dstRank == rank_ && myDev == dstDev) {
                peerAccessOutbox.push_back(sMsg);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMemcpyPeer)) {
              if (dstRank == rank_ && gpuTopology_.peer(myDev, dstDev)) {
                peerCopyOutboxes[di][dstGPU].push_back(sMsg);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpiColocated)) {
              if (dstRank != rank_ && mpiTopology_.colocated(dstRank) &&
                  gpuTopology_.peer(myDev, dstDev)) {
                assert(di < coloOutboxes.size());
                coloOutboxes[di].emplace(dstIdx, std::vector<Message>());
                coloOutboxes[di][dstIdx].push_back(sMsg);
                goto send_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpi | MethodFlags::CudaAwareMpi)) {
              assert(di < remoteOutboxes.size());
              remoteOutboxes[di].emplace(dstIdx, std::vector<Message>());
              remoteOutboxes[di][dstIdx].push_back(sMsg);
              goto send_planned;
            }
            std::cerr << "No method available to send required message\n";
            exit(EXIT_FAILURE);
          send_planned: // successfully found a way to send

            const Dim3 srcIdx = (myIdx - dir).wrap(globalDim);
            const int srcRank = placement->get_rank(srcIdx);
            const int srcGPU = placement->get_subdomain_id(srcIdx);
            const int srcDev = placement->get_cuda(srcIdx);
            Message rMsg(dir, srcGPU, di);

            if (any_methods(MethodFlags::CudaKernel)) {
              if (srcRank == rank_ && srcDev == myDev) {
                // no recver needed
                goto recv_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMemcpyPeer)) {
              if (srcRank == rank_ && gpuTopology_.peer(srcDev, myDev)) {
                // no recver needed
                goto recv_planned;
              }
            }
            if (any_methods(MethodFlags::CudaMpiColocated)) {
              if (srcRank != rank_ && mpiTopology_.colocated(srcRank) &&
                  gpuTopology_.peer(srcDev, myDev)) {
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
              goto recv_planned;
            }
            std::cerr << "No method available to recv required message\n";
            exit(EXIT_FAILURE);
          recv_planned: // found a way to recv
            (void)0;
          }
        }
      }
    }
    nvtxRangePop(); // plan
#if STENCIL_PRINT_TIMINGS == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.plan %f s\n", maxElapsed);
    }
#endif

    // summarize communication plan
    std::string planFileName = "plan_" + std::to_string(rank_) + ".txt";
    std::ofstream planFile(planFileName, std::ofstream::out);

    planFile << "rank=" << rank_ << "\n\n";

    planFile << "domains\n";
    for (size_t di = 0; di < domains_.size(); ++di) {
      planFile << di << ":" << domains_[di].gpu() << ":"
               << placement->get_idx(rank_, di) << " sz=" << domains_[di].size()
               << "\n";
    }
    planFile << "\n";

    planFile << "== peerAccess ==\n";
    for (auto &m : peerAccessOutbox) {
      planFile << m.srcGPU_ << "->" << m.dstGPU_ << " " << m.dir_ << "\n";
    }
    planFile << "\n";

    planFile << "== peerCopy ==\n";
    for (size_t srcGPU = 0; srcGPU < peerCopyOutboxes.size(); ++srcGPU) {
      for (size_t dstGPU = 0; dstGPU < peerCopyOutboxes[srcGPU].size();
           ++dstGPU) {
        size_t numBytes = 0;
        for (const auto &msg : peerCopyOutboxes[srcGPU][dstGPU]) {
          for (size_t i = 0; i < domains_[srcGPU].num_data(); ++i) {
            size_t haloBytes = domains_[srcGPU].halo_bytes(msg.dir_, i);
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

#if STENCIL_PRINT_TIMINGS == 1
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
      const Dim3 myIdx = placement->get_idx(rank_, di);
      for (auto &kv : remoteOutboxes[di]) {
        const Dim3 dstIdx = kv.first;
        const int dstRank = placement->get_rank(dstIdx);
        const int dstGPU = placement->get_subdomain_id(dstIdx);
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
        const int srcRank = placement->get_rank(srcIdx);
        const int srcGPU = placement->get_subdomain_id(srcIdx);
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
      const Dim3 myIdx = placement->get_idx(rank_, di);
      for (auto &kv : coloOutboxes[di]) {
        const Dim3 dstIdx = kv.first;
        const int dstRank = placement->get_rank(dstIdx);
        const int dstGPU = placement->get_subdomain_id(dstIdx);
        coloSenders_[di].emplace(
            dstIdx,
            ColocatedHaloSender(rank_, di, dstRank, dstGPU, domains_[di]));
      }
      for (auto &kv : coloInboxes[di]) {
        const Dim3 srcIdx = kv.first;
        const int srcRank = placement->get_rank(srcIdx);
        const int srcGPU = placement->get_subdomain_id(srcIdx);
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
    std::cerr << "DistributedDomain::realize: prepare peerAccessSender\n";
    nvtxRangePush("DistributedDomain::realize: prep peerAccessSender");
    peerAccessSender_.prepare(peerAccessOutbox, domains_);
    nvtxRangePop();
    std::cerr << "DistributedDomain::realize: prepare peerCopySender\n";
    nvtxRangePush("DistributedDomain::realize: prep peerCopySender");
    for (size_t srcGPU = 0; srcGPU < peerCopySenders_.size(); ++srcGPU) {
      for (auto &kv : peerCopySenders_[srcGPU]) {
        const int dstGPU = kv.first;
        auto &sender = kv.second;
        sender.prepare(peerCopyOutboxes[srcGPU][dstGPU]);
      }
    }
    nvtxRangePop();
    std::cerr << "DistributedDomain::realize: prepare colocatedHaloSender\n";
    nvtxRangePush("DistributedDomain::realize: prep colocated");
    assert(coloSenders_.size() == coloRecvers_.size());
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 srcIdx = placement->get_idx(rank_, di);
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        std::cerr << "rank=" << rank_ << " colo sender.start_prepare " << srcIdx
                  << "->" << dstIdx << "\n";
        sender.start_prepare(coloOutboxes[di][dstIdx]);
      }
      for (auto &kv : coloRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        const Dim3 dstIdx = placement->get_idx(rank_, di);
        auto &recver = kv.second;
        std::cerr << "rank=" << rank_ << " colo recver.start_prepare " << srcIdx
                  << "->" << dstIdx << "\n";
        recver.start_prepare(coloInboxes[di][srcIdx]);
      }
    }
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        const int srcDev = domains_[di].gpu();
        const int dstDev = placement->get_cuda(dstIdx);
        std::cerr << "rank=" << rank_
                  << " colo sender.finish_prepare for colo to " << dstIdx
                  << "\n";
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
    std::cerr << "DistributedDomain::realize: prepare remoteSender\n";
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

#if STENCIL_PRINT_TIMINGS == 1
    elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.create %f s\n", maxElapsed);
    }
#endif
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {

#if STENCIL_PRINT_TIMINGS == 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#endif

    // start remote send d2h
    fprintf(stderr, "rank=%d send remote d2h\n", rank_);
    nvtxRangePush("DD::exchange: remote send d2h");
    for (auto &domSenders : remoteSenders_) {
      for (auto &kv : domSenders) {
        StatefulSender *sender = kv.second;
        sender->send();
      }
    }
    nvtxRangePop();

    // start colocated Senders
    fprintf(stderr, "rank=%d start colo send\n", rank_);
    nvtxRangePush("DD::exchange: colo send");
    for (auto &domSenders : coloSenders_) {
      for (auto &kv : domSenders) {
        ColocatedHaloSender &sender = kv.second;
        sender.send();
      }
    }
    nvtxRangePop();

    // send same-rank messages
    fprintf(stderr, "rank=%d send peer copy\n", rank_);
    nvtxRangePush("DD::exchange: peer copy send");
    for (auto &src : peerCopySenders_) {
      for (auto &kv : src) {
        PeerCopySender &sender = kv.second;
        sender.send();
      }
    }
    nvtxRangePop();

    // send local messages
    fprintf(stderr, "rank=%d send peer access\n", rank_);
    nvtxRangePush("DD::exchange: peer access send");
    peerAccessSender_.send();
    nvtxRangePop();

    // start colocated recvers
    fprintf(stderr, "rank=%d start colo recv\n", rank_);
    nvtxRangePush("DD::exchange: colo recv");
    for (auto &domRecvers : coloRecvers_) {
      for (auto &kv : domRecvers) {
        ColocatedHaloRecver &recver = kv.second;
        recver.recv();
      }
    }
    nvtxRangePop();

    // start remote recv h2h
    fprintf(stderr, "rank=%d recv remote h2h\n", rank_);
    nvtxRangePush("DD::exchange: remote recv h2h");
    for (auto &domRecvers : remoteRecvers_) {
      for (auto &kv : domRecvers) {
        StatefulRecver *recver = kv.second;
        recver->recv();
      }
    }
    nvtxRangePop();

    // poll senders and recvers to move onto next step until all are done
    fprintf(stderr, "rank=%d start poll\n", rank_);
    nvtxRangePush("DD::exchange: poll");
    bool pending = true;
    while (pending) {
      pending = false;
    recvers:
      // move recvers from h2h to h2d
      for (auto &domRecvers : remoteRecvers_) {
        for (auto &kv : domRecvers) {
          const Dim3 srcIdx = kv.first;
          StatefulRecver *recver = kv.second;
          if (recver->active()) {
            pending = true;
            if (recver->next_ready()) {
              std::cerr << "rank=" << rank_ << " src=" << srcIdx
                        << " recv_h2d\n";
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
          const Dim3 dstIdx = kv.first;
          StatefulSender *sender = kv.second;
          if (sender->active()) {
            pending = true;
            if (sender->next_ready()) {
              std::cerr << "rank=" << rank_ << " dst=" << dstIdx
                        << " send_h2h\n";
              sender->next();
              goto recvers; // try to overlap recv_h2d with send_h2h
            }
          }
        }
      }
    }
    nvtxRangePop(); // DD::exchange: poll

    // wait for sends
    fprintf(stderr, "rank=%d wait for sameRankSender\n", rank_);
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
    nvtxRangePush("colocated.wait()");
    std::cerr << "colocated senders wait\n";
    for (auto &domSenders : coloSenders_) {
      for (auto &kv : domSenders) {
        ColocatedHaloSender &sender = kv.second;
        sender.wait();
      }
    }
    std::cerr << "colocated recvers wait\n";
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

#if STENCIL_PRINT_TIMINGS == 1
    double maxElapsed = -1;
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (0 == rank_) {
      printf("time.exchange %f s\n", maxElapsed);
    }
#endif

    // wait for all ranks to be done
    nvtxRangePush("barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop(); // barrier
  }
};
