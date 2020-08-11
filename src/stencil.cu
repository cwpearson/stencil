#include "stencil/logging.hpp"
#include "stencil/stencil.hpp"

#include <vector>

uint64_t DistributedDomain::exchange_bytes_for_method(const MethodFlags &method) const {
  uint64_t ret = 0;
  if ((method && MethodFlags::CudaMpi) || (method && MethodFlags::CudaAwareMpi)) {
    ret += numBytesCudaMpi_;
  }
  if (method && MethodFlags::CudaMpiColocated) {
    ret += numBytesCudaMpiColocated_;
  }
  if (method && MethodFlags::CudaMemcpyPeer) {
    ret += numBytesCudaMemcpyPeer_;
  }
  if (method && MethodFlags::CudaKernel ) {
    ret += numBytesCudaKernel_;
  }
  return ret;
}

void DistributedDomain::realize() {
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
  nvtxRangePop(); // "placement"
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
  // realize local domains
  for (auto &d : domains_) {
    d.realize();
  }
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

  LOG_DEBUG("comm plan");
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
#ifdef STENCIL_TRACK_STATS
    numBytesCudaMpi_ = 0;
    numBytesCudaMpiColocated_ = 0;
    numBytesCudaMemcpyPeer_ = 0;
    numBytesCudaKernel_ = 0;
#endif
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
#ifdef STENCIL_TRACK_STATS
        numBytesCudaKernel_ += bytes;
#endif
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
#ifdef STENCIL_TRACK_STATS
            numBytesCudaMemcpyPeer_ += bytes;
#endif
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
#ifdef STENCIL_TRACK_STATS
          for (int64_t i = 0; i < domains_[di].num_data(); ++i) {
            // send size matches size of halo that we're recving into
            numBytesCudaMpiColocated_ += domains_[di].halo_bytes(msg.dir_ * -1, i);
          }
#endif
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
#ifdef STENCIL_TRACK_STATS
          for (int64_t i = 0; i < domains_[di].num_data(); ++i) {
            // send size matches size of halo that we're recving into
            numBytesCudaMpi_ += domains_[di].halo_bytes(msg.dir_ * -1, i);
          }
#endif
        }
      }
    }
    planFile.close();

// give every rank the total send volume
#ifdef STENCIL_TRACK_STATS
    nvtxRangePush("allreduce communication stats");
    MPI_Allreduce(MPI_IN_PLACE, &numBytesCudaMpi_, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numBytesCudaMpiColocated_, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numBytesCudaMemcpyPeer_, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numBytesCudaKernel_, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    nvtxRangePop();

    if (rank_ == 0) {
      LOG_INFO(numBytesCudaMpi_ << "B CudaMpi / exchange");
      LOG_INFO(numBytesCudaMpiColocated_ << "B CudaMpiColocated / exchange");
      LOG_INFO(numBytesCudaMemcpyPeer_ << "B CudaMemcpyPeer / exchange");
      LOG_INFO(numBytesCudaKernel_ << "B CudaKernel / exchange");
    }
#endif
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
      LOG_DEBUG(" colo sender.start_prepare " << placement_->get_idx(rank_, di) << "->" << dstIdx << "(rank " << dstRank
                                              << ")");
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

void DistributedDomain::swap() {
  LOG_DEBUG("swap()");

#if STENCIL_MEASURE_TIME == 1
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
#endif

  for (auto &d : domains_) {
    d.swap();
  }

#if STENCIL_MEASURE_TIME == 1
  double elapsed = MPI_Wtime() - start;
  double maxElapsed = -1;
  MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (0 == rank_) {
    timeSwap_ += maxElapsed;
  }
#endif
}

/* start with the whole compute region, and check each direction of the
stencil. move the correponding face/edge/corner inward enough to compensate
an access in that direction
*/
std::vector<Rect3> DistributedDomain::get_interior() const {

  // one sparse domain for each LocalDomain
  std::vector<Rect3> ret(domains_.size());

  // direction of our halo
  for (size_t di = 0; di < domains_.size(); ++di) {
    const LocalDomain &dom = domains_[di];

    const Rect3 comReg = dom.get_compute_region();
    Rect3 intReg = dom.get_compute_region();

    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const Dim3 dir(dx, dy, dz);
          if (Dim3(0, 0, 0) == dir) {
            continue;
          }

          // if the radius is non-zero in a negative direction,
          // move the lower corner of that direction inward
          if (dir.x < 0) {
            intReg.lo.x = std::max(comReg.lo.x + int64_t(radius_.dir(dir)), intReg.lo.x);
          } else if (dir.x > 0) {
            intReg.hi.x = std::min(comReg.hi.x - int64_t(radius_.dir(dir)), intReg.hi.x);
          }
          if (dir.y < 0) {
            intReg.lo.y = std::max(comReg.lo.y + int64_t(radius_.dir(dir)), intReg.lo.y);
          } else if (dir.y > 0) {
            intReg.hi.y = std::min(comReg.hi.y - int64_t(radius_.dir(dir)), intReg.hi.y);
          }
          if (dir.z < 0) {
            intReg.lo.z = std::max(comReg.lo.z + int64_t(radius_.dir(dir)), intReg.lo.z);
          } else if (dir.z > 0) {
            intReg.hi.z = std::min(comReg.hi.z - int64_t(radius_.dir(dir)), intReg.hi.z);
          }
        }
      }
    }
    ret[di] = intReg;
  }
  return ret;
}

/* the exterior is everything that is not in the interior.
   build non-overlapping regions by sliding faces of the compute region in
   until they reach the interior
*/
std::vector<std::vector<Rect3>> DistributedDomain::get_exterior() const {

  // one sparse domain for each LocalDomain
  std::vector<std::vector<Rect3>> ret(domains_.size());

  const std::vector<Rect3> intRegs = get_interior();

  for (size_t di = 0; di < domains_.size(); ++di) {
    const LocalDomain &dom = domains_[di];
    const Rect3 &intReg = intRegs[di];
    Rect3 comReg = dom.get_compute_region();

    // +x
    if (intReg.hi.x != comReg.hi.x) {
      Rect3 extReg(Dim3(intReg.hi.x, comReg.lo.y, comReg.lo.z), Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.x = intReg.hi.x; // slide face in
      ret[di].push_back(extReg);
    }
    // +y
    if (intReg.hi.y != comReg.hi.y) {
      Rect3 extReg(Dim3(comReg.lo.x, intReg.hi.y, comReg.lo.z), Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.y = intReg.hi.y; // slide face in
      ret[di].push_back(extReg);
    }
    // +z
    if (intReg.hi.z != comReg.hi.z) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, intReg.hi.z), Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.z = intReg.hi.z; // slide face in
      ret[di].push_back(extReg);
    }
    // -x
    if (intReg.lo.x != comReg.lo.x) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), Dim3(intReg.lo.x, comReg.hi.y, comReg.hi.z));
      comReg.lo.x = intReg.lo.x; // slide face in
      ret[di].push_back(extReg);
    }
    // -y
    if (intReg.lo.y != comReg.lo.y) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), Dim3(comReg.hi.x, intReg.lo.y, comReg.hi.z));
      comReg.lo.y = intReg.lo.y; // slide face in
      ret[di].push_back(extReg);
    }
    // -z
    if (intReg.lo.z != comReg.lo.z) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), Dim3(comReg.hi.x, comReg.hi.y, intReg.lo.z));
      comReg.lo.z = intReg.lo.z; // slide face in
      ret[di].push_back(extReg);
    }
  }
  return ret;
}

const Rect3 DistributedDomain::get_compute_region() const noexcept { return Rect3(Dim3(0, 0, 0), size_); }

void DistributedDomain::exchange() {

#if STENCIL_MEASURE_TIME == 1
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
#endif

  /*! Try to start sends in order from longest to shortest
   * we expect remote to be longest, followed by peer copy, followed by colo
   * colo is shorter than peer copy due to the node-aware data placement:
   * if we try to place bigger exchanges nearby, they will be faster
   */

  // start remote send d2h
  LOG_DEBUG("remote send start");
  nvtxRangePush("DD::exchange: remote send d2h");
  for (auto &domSenders : remoteSenders_) {
    for (auto &kv : domSenders) {
      StatefulSender *sender = kv.second;
      sender->send();
    }
  }
  nvtxRangePop();

  // send same-rank messages
  LOG_DEBUG("send peer copy");
  nvtxRangePush("DD::exchange: peer copy send");
  for (auto &src : peerCopySenders_) {
    for (auto &kv : src) {
      PeerCopySender &sender = kv.second;
      sender.send();
    }
  }
  nvtxRangePop();

  // start colocated Senders
  LOG_DEBUG("start colo send");
  nvtxRangePush("DD::exchange: colo send");
  for (auto &domSenders : coloSenders_) {
    for (auto &kv : domSenders) {
      ColocatedHaloSender &sender = kv.second;
      sender.send();
    }
  }
  nvtxRangePop();

  // send self messages
  LOG_DEBUG("send peer access");
  nvtxRangePush("DD::exchange: peer access send");
  peerAccessSender_.send();
  nvtxRangePop();

  // start colocated recvers
  LOG_DEBUG("start colo recv");
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

  // poll stateful senders and recvers to move onto next step until all are done
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
            goto senders; // try to overlap sends and recvs
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
            sender->next();
            goto colo; // try to overlap sends and recvs
          }
        }
      }
    }
  colo:
    for (auto &domRecvers : coloRecvers_) {
      for (auto &kv : domRecvers) {
        ColocatedHaloRecver &recver = kv.second;
        if (recver.active()) {
          pending = true;
          if (recver.next_ready()) {
            recver.next();
            goto recvers; // try to overlap sends and recvs
          }
        }
      }
    }
  }
  nvtxRangePop(); // DD::exchange: poll

  // wait for sends
  LOG_SPEW("[" << rank_ << "] wait for peer access senders");
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
  for (auto &domSenders : coloSenders_) {
    for (auto &kv : domSenders) {
      LOG_SPEW("domain=" << kv.first << " wait colocated sender");
      ColocatedHaloSender &sender = kv.second;
      sender.wait();
    }
  }
  for (auto &domRecvers : coloRecvers_) {
    for (auto &kv : domRecvers) {
      LOG_SPEW("domain=" << kv.first << " wait colocated recver");
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
      LOG_SPEW("domain=" << kv.first << " wait remote recver");
      StatefulRecver *recver = kv.second;
      assert(recver);
      recver->wait();
    }
  }
  for (auto &domSenders : remoteSenders_) {
    for (auto &kv : domSenders) {
      LOG_SPEW("domain=" << kv.first << " wait remote sender");
      StatefulSender *sender = kv.second;
      assert(sender);
      sender->wait();
    }
  }
  nvtxRangePop(); // remote wait

#if STENCIL_MEASURE_TIME == 1
  double maxElapsed = -1;
  double elapsed = MPI_Wtime() - start;
  MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (0 == rank_) {
    timeExchange_ += maxElapsed;
  }
#endif

  // No barrier necessary: the CPU thread has already blocked until all recvs are done, so it is safe to proceed.
}

void DistributedDomain::write_paraview(const std::string &prefix, bool zeroNaNs) {

  const char delim[] = ",";

  nvtxRangePush("write_paraview");

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int64_t num = rank * domains_.size();

  for (size_t di = 0; di < domains_.size(); ++di) {
    int64_t id = rank * domains_.size() + di;
    const std::string path = prefix + "_" + std::to_string(id) + ".txt";

    LocalDomain &domain = domains_[di];

    LOG_DEBUG("copy interiors to host");
    std::vector<std::vector<unsigned char>> quantities;
    for (int64_t qi = 0; qi < domain.num_data(); ++qi) {
      quantities.push_back(domain.interior_to_host(qi));
    }

    LOG_INFO("open " << path);
    FILE *outf = fopen(path.c_str(), "w");

    // column headers
    fprintf(outf, "Z%sY%sX", delim, delim);
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
                  quantities[qi].data())[lz * (domain.sz_.y * domain.sz_.x) + ly * domain.sz_.x + lx];
              if (zeroNaNs && std::isnan(val)) {
                val = 0.0;
              }
              fprintf(outf, "%s%f", delim, val);
            } else if (4 == domain.elem_size(qi)) {
              float val = reinterpret_cast<float *>(
                  quantities[qi].data())[lz * (domain.sz_.y * domain.sz_.x) + ly * domain.sz_.x + lx];
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