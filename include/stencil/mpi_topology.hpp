#pragma once

#include <mpi.h>

#include <set>

class MpiTopology {
private:
  MPI_Comm comm_;
  MPI_Comm shmComm_;

  // colocated global ranks
  std::set<int> colocated_;

public:
  /* Should be called by all processes in comm
   */
  MpiTopology(MPI_Comm comm) : comm_(comm), shmComm_{} {
    if (comm_) {
      MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmComm_);

      LOG_DEBUG("MpiTopology: shmcomm rank " << colocated_rank() << "/" << colocated_size());

      // Give every rank a list of co-located ranks

      std::vector<int> colocated(colocated_size());
      int rank;
      MPI_Comm_rank(comm_, &rank);
      MPI_Allgather(&rank, 1, MPI_INT, colocated.data(), 1, MPI_INT, shmComm_);
      for (auto &r : colocated) {
        colocated_.insert(r);
      }
    } else {
      colocated_.insert(0);
    }
  }
  MpiTopology() : MpiTopology({}) {}
  MpiTopology(const MpiTopology &other) = delete;
  MpiTopology(MpiTopology &&other) = delete;
  ~MpiTopology() {
    if (shmComm_) {
      MPI_Comm_free(&shmComm_);
    }
  }

  // move assignment
  MpiTopology &operator=(MpiTopology &&other) {
    comm_ = std::move(other.comm_);
    shmComm_ = std::move(other.shmComm_);
    other.shmComm_ = 0;
    colocated_ = std::move(other.colocated_);
    return *this;
  }

  int rank() const noexcept {
    assert(comm_);
    int ret;
    MPI_Comm_rank(comm_, &ret);
    return ret;
  }

  int size() const noexcept {
    assert(comm_);
    int ret;
    MPI_Comm_size(comm_, &ret);
    return ret;
  }

  MPI_Comm comm() const noexcept { return comm_; }

  MPI_Comm colocated_comm() const noexcept {
    assert(shmComm_);
    return shmComm_;
  }
  int colocated_rank() const noexcept {
    if (shmComm_) {
      int ret;
      MPI_Comm_rank(shmComm_, &ret);
      return ret;
    } else {
      return 0;
    }
  }
  int colocated_size() const noexcept {
    if (shmComm_) {
      int ret;
      MPI_Comm_size(shmComm_, &ret);
      return ret;
    } else {
      return 1;
    }
  }

  // true if the calling rank in comm is colocated with `rank`
  bool colocated(int rank) const noexcept { return 0 != colocated_.count(rank); }
};
