#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <set>
#include <vector>

#include <mpi.h>

#include <nvToolsExt.h>
#include <nvml.h>

#include "cuda_runtime.hpp"

#include "stencil/dim3.hpp"
#include "stencil/direction_map.hpp"
#include "stencil/gpu_topo.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/nvml.hpp"
#include "stencil/partition.hpp"
#include "stencil/tx.hpp"

class DistributedDomain {
private:
  Dim3 size_;

  int rank_;
  int worldSize_;

  // the GPUs this MPI rank will use
  std::vector<int> gpus_;

  // the stencil radius
  size_t radius_;

  // typically one per GPU
  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;
  // the index of the domain in the distributed domain
  std::vector<Dim3> domainIdx_;

  // information about mapping of computation domain to workers
  Partition *partition_;

  std::vector<std::map<Dim3, RemoteSender>>
      remoteSenders_; // remoteSender_[domain][dstIdx] = sender
  std::vector<std::map<Dim3, RemoteRecver>>
      remoteRecvers_; // remoteRecver_[domain][srcIdx] = sender

  // one sender for all local exchanges
  PeerAccessSender peerAccessSender_;

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

  // MPI ranks co-located with me
  std::set<int64_t> colocated_;

  std::vector<std::vector<bool>> peerAccess_; //<! which GPUs have peer access

public:
  DistributedDomain(size_t x, size_t y, size_t z) : size_(x, y, z) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));

    // create a communicator for ranks on the same node
    MPI_Barrier(MPI_COMM_WORLD); // to stabilize co-located timing
    double start = MPI_Wtime();
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    int shmrank, shmsize;
    MPI_Comm_rank(shmcomm, &shmrank);
    MPI_Comm_size(shmcomm, &shmsize);
    printf("DistributedDomain::ctor(): shmcomm rank %d/%d\n", shmrank, shmsize);

    // Give every rank a list of co-located ranks
    std::vector<int> colocated(shmsize);
    MPI_Allgather(&rank_, 1, MPI_INT, colocated.data(), 1, MPI_INT, shmcomm);
    for (auto &r : colocated) {
      colocated_.insert(r);
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.colocate [%d] %fs\n", rank_, elapsed);
    assert(colocated_.count(rank_) == 1 && "should be colocated with self");
    printf(
        "DistributedDomain::ctor(): rank %d colocated with %lu other ranks\n",
        rank_, colocated_.size() - 1);

    // if fewer ranks than GPUs, round-robin GPUs to ranks
    if (shmsize <= deviceCount) {
      for (int gpu = 0; gpu < deviceCount; ++gpu) {
        if (gpu % shmsize == shmrank) {
          gpus_.push_back(gpu);
        }
      }
    } else { // if more ranks, share gpus among ranks
      gpus_.push_back(shmrank % deviceCount);
    }

    for (const auto gpu : gpus_) {
      printf("rank %d/%d local=%d using gpu %d\n", rank_, worldSize_, shmrank,
             gpu);
    }

    start = MPI_Wtime();

    // Try to enable peer access between all GPUs
    nvtxRangePush("peer_en");

    // can't use gpus_.size() because we don't own all the GPUs
    peerAccess_ = std::vector<std::vector<bool>>(
        deviceCount, std::vector<bool>(deviceCount, false));

    for (int src = 0; src < deviceCount; ++src) {
      for (int dst = 0; dst < deviceCount; ++dst) {
        if (src == dst) {
          peerAccess_[src][dst] = true;
          std::cout << src << " -> " << dst << " peer access\n";
        } else {
          int canAccess;
	  CUDA_RUNTIME(cudaDeviceCanAccessPeer(&canAccess, src, dst));
	  if (canAccess) {
            CUDA_RUNTIME(cudaSetDevice(src))
            cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0 /*flags*/);
            if (cudaSuccess == err || cudaErrorPeerAccessAlreadyEnabled == err) {
              peerAccess_[src][dst] = true;
              std::cout << src << " -> " << dst << " peer access\n";
            } else if (cudaErrorInvalidDevice) {
              peerAccess_[src][dst] = false;
            } else {
              assert(0);
              peerAccess_[src][dst] = false;
            }
          } else {
            peerAccess_[src][dst] = false;
	  }
	}
      }
    }
    nvtxRangePop();
    elapsed = MPI_Wtime() - start;
    printf("time.peer [%d] %fs\n", rank_, elapsed);

    start = MPI_Wtime();
    nvtxRangePush("gpu_topo");
    Mat2D dist = get_gpu_distance_matrix();
    nvtxRangePop();
    if (0 == rank_) {
      std::cerr << "gpu distance matrix: \n";
      for (auto &r : dist) {
        for (auto &c : r) {
          std::cerr << c << " ";
        }
        std::cerr << "\n";
      }
    }
    elapsed = MPI_Wtime() - start;
    printf("time.topo [%d] %fs\n", rank_, elapsed);

    // determine decomposition information
    start = MPI_Wtime();
    nvtxRangePush("partition");
    partition_ = new PFP(size_, worldSize_, gpus_.size());
    nvtxRangePop();
    elapsed = MPI_Wtime() - start;
    printf("time.partition [%d] %fs\n", rank_, elapsed);

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank_) {
      std::cerr << "split " << size_ << " into " << partition_->rank_dim()
                << "x" << partition_->gpu_dim() << "\n";
    }
  }

  ~DistributedDomain() { delete partition_; }

  std::vector<LocalDomain> &domains() { return domains_; }

  void set_radius(size_t r) { radius_ = r; }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
  }

  void realize(bool useUnified = false) {

    // create local domains
    double start = MPI_Wtime();
    for (int i = 0; i < gpus_.size(); i++) {

      Dim3 idx = partition_->dom_idx(rank_, i);
      Dim3 ldSize = partition_->local_domain_size(idx);

      LocalDomain ld(ldSize, gpus_[i]);
      ld.radius_ = radius_;
      for (size_t dataIdx = 0; dataIdx < dataElemSize_.size(); ++dataIdx) {
        ld.add_data(dataElemSize_[dataIdx]);
      }

      domains_.push_back(ld);

      printf("rank=%d gpu=%d (cuda id=%d) => [%ld,%ld,%ld]\n", rank_, i,
             gpus_[i], idx.x, idx.y, idx.z);
      domainIdx_.push_back(idx);
    }

    // realize local domains
    for (auto &d : domains_) {
      if (useUnified)
        d.realize_unified();
      else
        d.realize();
      printf("DistributedDomain.realize(): finished creating LocalDomain\n");
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.local_realize [%d] %fs\n", rank_, elapsed);

    start = MPI_Wtime();
    nvtxRangePush("comm plan");

    // one outbox and sender for all local exchanges
    std::vector<Message> peerAccessOutbox;

    // one inbox for each remote domain my domains recv from
    std::vector<std::map<Dim3, std::vector<Message>>>
        remoteInboxes; // remoteOutboxes_[domain][srcIdx] = messages
    // one outbox for each remote domain my domains send to
    std::vector<std::map<Dim3, std::vector<Message>>>
        remoteOutboxes; // remoteOutboxes[domain][dstIdx] = messages

    remoteOutboxes.resize(gpus_.size());
    remoteSenders_.resize(gpus_.size());
    remoteInboxes.resize(gpus_.size());
    remoteRecvers_.resize(gpus_.size());

    // create all remote senders/recvers

    const Dim3 globalDim = partition_->gpu_dim() * partition_->rank_dim();
    for (size_t di = 0; di < domains_.size(); ++di) {
      const Dim3 myIdx = partition_->dom_idx(rank_, di);
      for (int z = -1; z < 1; ++z) {
        for (int y = -1; y < 1; ++y) {
          for (int x = -1; x < 1; ++x) {
            Dim3 dir(x, y, z);
	    if (Dim3(0,0,0) == dir) {
               continue; // no communication in this direction
	    }
            Dim3 srcIdx = (myIdx - dir).wrap(globalDim);
            Dim3 dstIdx = (myIdx + dir).wrap(globalDim);
            const int srcRank = partition_->get_rank(srcIdx);
            const int dstRank = partition_->get_rank(dstIdx);
            const int srcGPU = partition_->get_gpu(srcIdx);
            const int dstGPU = partition_->get_gpu(dstIdx);

            if (rank_ != srcRank) {
              if (0 == remoteRecvers_[di].count(srcIdx)) {
                remoteRecvers_[di][srcIdx] =
                    RemoteRecver(rank_, di, dstRank, dstGPU, domains_[di]);
                remoteInboxes[di][srcIdx] = std::vector<Message>();
              }
            }

            if (rank_ != dstRank) {
              // TODO: don't reconstruct
              remoteSenders_[di][dstIdx] =
                  RemoteSender(srcRank, srcGPU, rank_, di, domains_[di]);
              remoteOutboxes[di][dstIdx] = std::vector<Message>();
            }
          }
        }
      }
    }

    // plan messages
    nvtxRangePush("DistributedDomain::realize() plan messages");
    for (size_t di = 0; di < domains_.size(); ++di) {
      const Dim3 myIdx = partition_->dom_idx(rank_, di);
      for (int z = -1; z < 1; ++z) {
        for (int y = -1; y < 1; ++y) {
          for (int x = -1; x < 1; ++x) {
            const Dim3 dir(x, y, z);
	    if (Dim3(0,0,0) == dir) {
              continue; // no message
	    }

            const Dim3 dstIdx = (myIdx + dir).wrap(globalDim);
            const int dstRank = partition_->get_rank(dstIdx);
            const int dstGPU = partition_->get_gpu(dstIdx);

            Message sMsg(dir, di, dstGPU);
            if (rank_ == dstRank) {
              const int myDev = domains_[di].gpu();
              const int dstDev = domains_[dstGPU].gpu();
              if (peerAccess_[myDev][dstDev]) {
                peerAccessOutbox.push_back(sMsg);
              } else {
                remoteOutboxes[di][dstIdx].push_back(sMsg);
              }
            } else {
              remoteOutboxes[di][dstIdx].push_back(sMsg);
            }

            const Dim3 srcIdx = (myIdx - dir).wrap(globalDim);
            const int srcRank = partition_->get_rank(srcIdx);
            const int srcGPU = partition_->get_gpu(srcIdx);
            Message rMsg(dir, srcGPU, di);
            if (rank_ == srcRank) {
              const int myDev = domains_[di].gpu();
              const int srcDev = domains_[srcGPU].gpu();
              if (peerAccess_[srcDev][myDev]) {
                // no recver needed for peer access
              } else {
                remoteInboxes[di][srcIdx].push_back(rMsg);
              }
            } else {
              remoteInboxes[di][srcIdx].push_back(rMsg);
            }
          }
        }
      }
    }
    nvtxRangePop();

    // prepare senders and receivers
    peerAccessSender_.prepare(peerAccessOutbox, domains_);
    for (size_t di = 0; di < domains_.size(); ++di) {
      for (auto &kv : remoteSenders_[di]) {
        const Dim3 dstIdx = kv.first;
        auto &sender = kv.second;
        sender.prepare(remoteOutboxes[di][dstIdx]);
      }
      for (auto &kv : remoteRecvers_[di]) {
        const Dim3 srcIdx = kv.first;
        auto &recver = kv.second;
        recver.prepare(remoteInboxes[di][srcIdx]);
      }
    }

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
    // printf("rank=%d send remote d2h\n", rank_);
    for (size_t di = 0; di < domains_.size(); ++di) {
      for (auto &kv : remoteSenders_[di]) {
        Dim3 dstIdx = kv.first;
        RemoteSender &sender = kv.second;
        sender.send_d2h();
      }
    }

    // start remote recv h2h
    // printf("rank=%d recv remote h2h\n", rank_);
    for (size_t di = 0; di < domains_.size(); ++di) {
      for (auto &kv : remoteRecvers_[di]) {
        Dim3 srcIdx = kv.first;
        RemoteRecver &recver = kv.second;
        recver.recv_h2h();
      }
    }

    // send local messages
    // printf("rank=%d send peer access\n", rank_);
    peerAccessSender_.send();

    // poll senders and recvers to move onto next step until all are done
    nvtxRangePush("poll");
    bool pending = true;
    while (pending) {
      pending = false;
      for (size_t di = 0; di < domains_.size(); ++di) {
        // move recvers from h2h to h2d
        for (auto &kv : remoteRecvers_[di]) {
          Dim3 srcIdx = kv.first;
          RemoteRecver &recver = kv.second;
          if (recver.is_h2h()) {
            pending = true;
            if (recver.h2h_done()) {
              recver.recv_h2d();
            }
          }
        }
        // move senders from d2h to h2h
        for (auto &kv : remoteSenders_[di]) {
          Dim3 dstIdx = kv.first;
          RemoteSender &sender = kv.second;
          if (sender.is_d2h()) {
            pending = true;
            if (sender.d2h_done()) {
              sender.send_h2h();
            }
          }
        }
      }
    }
    nvtxRangePop();

    // wait for sends
    // printf("rank=%d wait for sameRankSender\n", rank_);
    peerAccessSender_.wait();

    // wait for remote senders and recvers
    // printf("rank=%d wait for RemoteRecver/RemoteSender\n", rank_);
    for (size_t di = 0; di < domains_.size(); ++di) {
      for (auto &kv : remoteRecvers_[di]) {
        Dim3 srcIdx = kv.first;
        RemoteRecver &recver = kv.second;
        recver.wait();
      }
      for (auto &kv : remoteSenders_[di]) {
        Dim3 dstIdx = kv.first;
        RemoteSender &sender = kv.second;
        sender.wait();
      }
    }

    double elapsed = MPI_Wtime() - start;
    printf("time.exchange [%d] %fs\n", rank_, elapsed);

    // wait for all ranks to be done
    MPI_Barrier(MPI_COMM_WORLD);
  }
};
