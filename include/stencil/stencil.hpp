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
  All = 1 + 2 + 4 + 8 + 16
};
static_assert(sizeof(MethodFlags) == sizeof(int));

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

  std::vector<std::map<Dim3, StatefulSender *>>
      remoteSenders_; // remoteSender_[domain][dstIdx] = sender
  std::vector<std::map<Dim3, StatefulRecver *>>
      remoteRecvers_; // remoteRecver_[domain][srcIdx] = recver

  // kernel sender for same-domain sends
  PeerAccessSender peerAccessSender_;

  // cudaMemcpyPeerAsync sender for local exchanges
  PeerCopySender peerCopySender_;

  std::vector<std::map<Dim3, ColocatedHaloSender>>
      coloSenders_; // vec[domain][dstIdx] = sender
  std::vector<std::map<Dim3, ColocatedHaloRecver>> coloRecvers_;

public:
  DistributedDomain(size_t x, size_t y, size_t z)
      : size_(x, y, z), flags_(MethodFlags::All) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

    // create a communicator for ranks on the same node
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    mpiTopology_ = std::move(MpiTopology(MPI_COMM_WORLD));
    double elapsed = MPI_Wtime() - start;
    printf("time.mpi_topo [%d] %fs\n", rank_, elapsed);

    std::cout << "[" << rank_ << "] colocated with "
              << mpiTopology_.colocated_size() << " ranks\n";

    // Determine GPUs this DistributedDomain is reposible for
    if (gpus_.empty()) {
      int deviceCount;
      CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));
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
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    std::vector<int> nodeCudaIds(gpus_.size() * mpiTopology_.colocated_size());
    MPI_Allgather(gpus_.data(), gpus_.size(), MPI_INT, nodeCudaIds.data(),
                  gpus_.size(), MPI_INT, mpiTopology_.colocated_comm());
    elapsed = MPI_Wtime() - start;
    printf("time.node_gpus [%d] %fs\n", rank_, elapsed);
    {
      std::set<int> unique(nodeCudaIds.begin(), nodeCudaIds.end());
      // nodeCudaIds = std::vector<int>(unique.begin(), unique.end());
      std::cout << "[" << rank_ << "] colocated with ranks using gpus";
      for (auto &e : nodeCudaIds) {
        std::cout << " " << e;
      }
      std::cout << "\n";
    }

    // determine topology info for used GPUs
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    gpuTopology_ = GpuTopology(nodeCudaIds);
    elapsed = MPI_Wtime() - start;
    printf("time.gpu_topo [%d] %fs\n", rank_, elapsed);

    start = MPI_Wtime();
    // Try to enable peer access between all GPUs
    nvtxRangePush("peer_en");
    gpuTopology_.enable_peer();
    nvtxRangePop();
    elapsed = MPI_Wtime() - start;
    printf("time.peer [%d] %fs\n", rank_, elapsed);

    MPI_Barrier(MPI_COMM_WORLD);
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
    nvtxRangePush("node-aware placement");
    std::cerr << "[" << rank_ << "] do NAP\n";
    NodeAwarePlacement *nap_ = new NodeAwarePlacement(
        size_, worldSize_, mpiTopology_, gpuTopology_, radius_, gpus_);
    nvtxRangePop();

    double start = MPI_Wtime();
    for (int domId = 0; domId < gpus_.size(); domId++) {

      Dim3 idx = nap_->dom_idx(rank_, domId);
      Dim3 ldSize = nap_->domain_size(idx);

      // placement algorithm should agree what my GPU is
      assert(nap_->get_cuda(idx) == gpus_[domId]);

      LocalDomain ld(ldSize, gpus_[domId]);
      ld.radius_ = radius_;
      for (size_t dataIdx = 0; dataIdx < dataElemSize_.size(); ++dataIdx) {
        ld.add_data(dataElemSize_[dataIdx]);
      }

      domains_.push_back(ld);

      printf("rank=%d gpu=%d (cuda id=%d) => [%ld,%ld,%ld]\n", rank_, domId,
             gpus_[domId], idx.x, idx.y, idx.z);
    }

    // realize local domains
    for (auto &d : domains_) {
      d.realize();
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.local_realize [%d] %fs\n", rank_, elapsed);

    start = MPI_Wtime();
    nvtxRangePush("comm plan");

    // outbox for same-GPU exchanges
    std::vector<Message> peerAccessOutbox;

    // outbox for same-rank exchanges
    std::vector<Message> peerCopyOutbox;

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

    const Dim3 globalDim = nap_->gpu_dim() * nap_->rank_dim();

    std::cerr << "plan\n";
    // plan messages
    nvtxRangePush("DistributedDomain::realize() plan messages");
    remoteOutboxes.resize(gpus_.size());
    remoteInboxes.resize(gpus_.size());
    coloOutboxes.resize(gpus_.size());
    coloInboxes.resize(gpus_.size());
    for (size_t di = 0; di < domains_.size(); ++di) {
      const Dim3 myIdx = nap_->dom_idx(rank_, di);
      const int myDev = domains_[di].gpu();
      assert(myDev == nap_->get_cuda(myIdx));
      for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
          for (int x = -1; x <= 1; ++x) {
            const Dim3 dir(x, y, z);
            if (Dim3(0, 0, 0) == dir) {
              continue; // no message
            }

            const Dim3 dstIdx = (myIdx + dir).wrap(globalDim);
            const int dstRank = nap_->get_rank(dstIdx);
            const int dstGPU = nap_->get_gpu(dstIdx);
            const int dstDev = nap_->get_cuda(dstIdx);
            Message sMsg(dir, di, dstGPU);
            if (rank_ == dstRank) {
              const int myDev = domains_[di].gpu();
              const int dstDev = domains_[dstGPU].gpu();
              if ((myDev == dstDev) && any_methods(MethodFlags::CudaKernel)) {
                peerAccessOutbox.push_back(sMsg);
              } else if (any_methods(MethodFlags::CudaMemcpyPeer)) {
                peerCopyOutbox.push_back(sMsg);
              } else if (any_methods(MethodFlags::CudaMpi |
                                     MethodFlags::CudaAwareMpi)) {
                assert(di < remoteOutboxes.size());
                remoteOutboxes[di].emplace(dstIdx, std::vector<Message>());
                remoteOutboxes[di][dstIdx].push_back(sMsg);
              } else {
                std::cerr << "No method available to send required message\n";
                exit(EXIT_FAILURE);
              }
            } else if (any_methods(MethodFlags::CudaMpiColocated) &&
                       mpiTopology_.colocated(dstRank) &&
                       gpuTopology_.peer(myDev, dstDev)) {
              assert(di < coloOutboxes.size());
              coloOutboxes[di].emplace(dstIdx, std::vector<Message>());
              coloOutboxes[di][dstIdx].push_back(sMsg);
            } else if (any_methods(MethodFlags::CudaMpi |
                                   MethodFlags::CudaAwareMpi)) {
              remoteOutboxes[di].emplace(dstIdx, std::vector<Message>());
              remoteOutboxes[di][dstIdx].push_back(sMsg);
            } else {
              std::cerr << "No method available to send required message\n";
              exit(EXIT_FAILURE);
            }

            const Dim3 srcIdx = (myIdx - dir).wrap(globalDim);
            const int srcRank = nap_->get_rank(srcIdx);
            const int srcGPU = nap_->get_gpu(srcIdx);
            const int srcDev = nap_->get_cuda(srcIdx);
            Message rMsg(dir, srcGPU, di);
            if (rank_ == srcRank) {
              const int myDev = domains_[di].gpu();
              const int srcDev = domains_[srcGPU].gpu();
              if ((myDev == srcDev) && any_methods(MethodFlags::CudaKernel)) {
                // no recver needed for same GPU
              } else if (any_methods(MethodFlags::CudaMemcpyPeer)) {
                // no recver needed for same rank
              } else if (any_methods(MethodFlags::CudaMpi |
                                     MethodFlags::CudaAwareMpi)) {
                remoteInboxes[di].emplace(srcIdx, std::vector<Message>());
                remoteInboxes[di][srcIdx].push_back(rMsg);
              } else {
                std::cerr << "No method available to recv required message\n";
                exit(EXIT_FAILURE);
              }
            } else if (any_methods(MethodFlags::CudaMpiColocated) &&
                       mpiTopology_.colocated(srcRank) &&
                       gpuTopology_.peer(srcDev, myDev)) {
              coloInboxes[di].emplace(srcIdx, std::vector<Message>());
              coloInboxes[di][srcIdx].push_back(rMsg);
            } else if (any_methods(MethodFlags::CudaMpi |
                                   MethodFlags::CudaAwareMpi)) {
              remoteInboxes[di].emplace(srcIdx, std::vector<Message>());
              remoteInboxes[di][srcIdx].push_back(rMsg);
            } else {
              std::cerr << "No method available to recv required message\n";
              exit(EXIT_FAILURE);
            }
          }
        }
      }
    }
    nvtxRangePop(); // plan

    // summarize communication plan
    std::string planFileName = "plan_" + std::to_string(rank_) + ".txt";
    std::ofstream planFile(planFileName, std::ofstream::out);
    
    planFile << "rank=" << rank_ <<"\n\n";

    planFile << "domains\n";
    for (size_t di = 0; di < domains_.size(); ++di) {
      planFile << di << ":" << domains_[di].gpu() << ":" << nap_->dom_idx(rank_, di) << "\n";
    }
    planFile << "\n";

    planFile << "peerAccess\n";
    for (auto &m : peerAccessOutbox) {
      planFile << m.srcGPU_ << "->" << m.dstGPU_ << " " << m.dir_ << "\n";
    }
    planFile << "\n";

    planFile << "peerCopy\n";
    for (auto &m : peerCopyOutbox) {
      planFile << m.dir_ << "\n";
    }
    planFile << "\n";
             
    for (auto &obxs : coloOutboxes) {
      for (auto &kv : obxs) {
        const Dim3 dstIdx = kv.first;
        auto &box = kv.second;
        planFile << "colo to dstIdx=" << dstIdx << "\n";
        for (auto &m : box) {
          planFile << "dir=" << m.dir_ << " (" << m.srcGPU_ << "->" << m.dstGPU_ << ")\n";
        }
      }
    }
    for (auto &obxs : remoteOutboxes) {
      for (auto &kv : obxs) {
        const Dim3 dstIdx = kv.first;
        auto &box = kv.second;
        planFile << "remote to dstIdx=" << dstIdx << "\n";
        for (auto &m : box) {
          planFile << "dir=" << m.dir_ << " (" << m.srcGPU_ << "->" << m.dstGPU_ << ")\n";
        }
      }
    }
    planFile.close();

    MPI_Barrier(MPI_COMM_WORLD);

    // create remote sender/recvers
    nvtxRangePush("DistributedDomain::realize: create remote");
    // per-domain senders and messages
    remoteSenders_.resize(gpus_.size());
    remoteRecvers_.resize(gpus_.size());

    std::cerr << "create remote\n";
    // create all required remote senders/recvers
    for (size_t di = 0; di < domains_.size(); ++di) {
      const Dim3 myIdx = nap_->dom_idx(rank_, di);
      for (auto &kv : remoteOutboxes[di]) {
        const Dim3 dstIdx = kv.first;
        const int dstRank = nap_->get_rank(dstIdx);
        const int dstGPU = nap_->get_gpu(dstIdx);
        if (0 == remoteSenders_[di].count(dstIdx)) {
          StatefulSender *sender;
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
        const int srcRank = nap_->get_rank(srcIdx);
        const int srcGPU = nap_->get_gpu(srcIdx);
        if (0 == remoteRecvers_[di].count(srcIdx)) {
          StatefulRecver *recver;
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
      const Dim3 myIdx = nap_->dom_idx(rank_, di);
      for (auto &kv : coloOutboxes[di]) {
        const Dim3 dstIdx = kv.first;
        const int dstRank = nap_->get_rank(dstIdx);
        const int dstGPU = nap_->get_gpu(dstIdx);
        coloSenders_[di].emplace(
            dstIdx,
            ColocatedHaloSender(rank_, di, dstRank, dstGPU, domains_[di]));
      }
      for (auto &kv : coloInboxes[di]) {
        const Dim3 srcIdx = kv.first;
        const int srcRank = nap_->get_rank(srcIdx);
        const int srcGPU = nap_->get_gpu(srcIdx);
        coloRecvers_[di].emplace(
            srcIdx,
            ColocatedHaloRecver(srcRank, srcGPU, rank_, di, domains_[di]));
      }
    }
    nvtxRangePop(); // create colocated

    // prepare senders and receivers
    std::cerr << "DistributedDomain::realize: prepare peerAccessSender\n";
    nvtxRangePush("DistributedDomain::realize: prep peerAccessSender");
    peerAccessSender_.prepare(peerAccessOutbox, domains_);
    nvtxRangePop();
    std::cerr << "DistributedDomain::realize: prepare peerCopySender\n";
    nvtxRangePush("DistributedDomain::realize: prep peerCopySender");
    peerCopySender_.prepare(peerCopyOutbox, domains_);
    nvtxRangePop();
    std::cerr << "DistributedDomain::realize: prepare colocatedHaloSender\n";
    assert(coloSenders_.size() == coloRecvers_.size());
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        std::cerr << "start_prepare for colo to " << dstIdx << "\n";
        sender.start_prepare(coloOutboxes[di][dstIdx]);
      }
      for (auto &kv : coloRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        auto &recver = kv.second;
        std::cerr << "start_prepare for colo from " << srcIdx << "\n";
        recver.start_prepare(coloInboxes[di][srcIdx]);
      }
    }
    for (size_t di = 0; di < coloSenders_.size(); ++di) {
      for (auto &kv : coloSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        const int srcDev = domains_[di].gpu();
        const int dstDev = nap_->get_cuda(dstIdx);
        std::cerr << srcDev << " " << dstDev << "\n";
        std::cerr << "rank=" << rank_ << " finish_prepare for colo to "
                  << dstIdx << "\n";
        sender.finish_prepare();
      }
      for (auto &kv : coloRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        auto &recver = kv.second;
        std::cerr << "rank=" << rank_ << " finish_prepare for colo from "
                  << srcIdx << "\n";
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

    nvtxRangePop(); // comm plan
    elapsed = MPI_Wtime() - start;
    printf("time.plan [%d] %fs\n", rank_, elapsed);
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {
    MPI_Barrier(MPI_COMM_WORLD); // stabilize time

    double start = MPI_Wtime();

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

    // send same-rank messages
    fprintf(stderr, "rank=%d send peer copy\n", rank_);
    nvtxRangePush("DD::exchange: peer copy send");
    peerCopySender_.send();
    nvtxRangePop();

    // send local messages
    fprintf(stderr, "rank=%d send peer access\n", rank_);
    nvtxRangePush("DD::exchange: peer access send");
    peerAccessSender_.send();
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
    nvtxRangePop();

    // wait for sends
    fprintf(stderr, "rank=%d wait for sameRankSender\n", rank_);
    nvtxRangePush("peerAccessSender.wait()");
    peerAccessSender_.wait();
    nvtxRangePop();

    nvtxRangePush("peerCopySender.wait()");
    peerCopySender_.wait();
    nvtxRangePop();

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

    double elapsed = MPI_Wtime() - start;
    printf("time.exchange [%d] %fs\n", rank_, elapsed);

    // wait for all ranks to be done
    nvtxRangePush("barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop(); // barrier
  }
};
