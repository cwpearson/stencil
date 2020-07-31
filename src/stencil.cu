#include "stencil/stencil.hpp"

#include <vector>

#ifndef STENCIL_OUTPUT_LEVEL
#define STENCIL_OUTPUT_LEVEL 0
#endif

#if STENCIL_OUTPUT_LEVEL <= 0
#define LOG_SPEW(x)                                                                                                    \
  if (0 == rank_)                                                                                                      \
    std::cerr << "SPEW[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_SPEW(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 1
#define LOG_DEBUG(x)                                                                                                   \
  if (0 == rank_)                                                                                                      \
    std::cerr << "DEBUG[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_DEBUG(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 2
#define LOG_INFO(x)                                                                                                    \
  if (0 == rank_)                                                                                                      \
    std::cerr << "INFO[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_INFO(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 3
#define LOG_WARN(x) std::cerr << "WARN[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_WARN(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 4
#define LOG_ERROR(x) std::cerr << "ERROR[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";
#else
#define LOG_ERROR(x)
#endif

#if STENCIL_OUTPUT_LEVEL <= 5
#define LOG_FATAL(x)                                                                                                   \
  std::cerr << "FATAL[" << __FILE__ << ":" << __LINE__ << "] " << x << "\n";                                           \
  exit(1);
#else
#define LOG_FATAL(x) exit(1);
#endif

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
      Rect3 extReg(Dim3(intReg.hi.x, comReg.lo.y, comReg.lo.z),
                   Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.x = intReg.hi.x; // slide face in
      ret[di].push_back(extReg);
    }
    // +y
    if (intReg.hi.y != comReg.hi.y) {
      Rect3 extReg(Dim3(comReg.lo.x, intReg.hi.y, comReg.lo.z),
                   Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.y = intReg.hi.y; // slide face in
      ret[di].push_back(extReg);
    }
    // +z
    if (intReg.hi.z != comReg.hi.z) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, intReg.hi.z), 
                   Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.z = intReg.hi.z; // slide face in
      ret[di].push_back(extReg);
    }
    // -x
    if (intReg.lo.x != comReg.lo.x) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z),
                   Dim3(intReg.lo.x, comReg.hi.y, comReg.hi.z));
      comReg.lo.x = intReg.lo.x; // slide face in
      ret[di].push_back(extReg);
    }
    // -y
    if (intReg.lo.y != comReg.lo.y) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), 
                   Dim3(comReg.hi.x, intReg.lo.y, comReg.hi.z));
      comReg.lo.y = intReg.lo.y; // slide face in
      ret[di].push_back(extReg);
    }
    // -z
    if (intReg.lo.z != comReg.lo.z) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z),
                   Dim3(comReg.hi.x, comReg.hi.y, intReg.lo.z));
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
  LOG_DEBUG("rank=" << rank_ << " send peer copy");
  nvtxRangePush("DD::exchange: peer copy send");
  for (auto &src : peerCopySenders_) {
    for (auto &kv : src) {
      PeerCopySender &sender = kv.second;
      sender.send();
    }
  }
  nvtxRangePop();

  // start colocated Senders
  LOG_DEBUG("rank=" << rank_ << " start colo send");
  nvtxRangePush("DD::exchange: colo send");
  for (auto &domSenders : coloSenders_) {
    for (auto &kv : domSenders) {
      ColocatedHaloSender &sender = kv.second;
      sender.send();
    }
  }
  // FIXME: make sure colocated sends happen before anyone starts to recv
  MPI_Barrier(MPI_COMM_WORLD); 
  nvtxRangePop();

  // send self messages
  LOG_DEBUG("rank=" << rank_ << " send peer access");
  nvtxRangePush("DD::exchange: peer access send");
  peerAccessSender_.send();
  nvtxRangePop();

  // start colocated recvers
  LOG_DEBUG("rank=" << rank_ << " start colo recv");
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
      LOG_SPEW("rank=" << rank_ << " domain=" << kv.first << " wait colocated sender");
      ColocatedHaloSender &sender = kv.second;
      sender.wait();
    }
  }
  for (auto &domRecvers : coloRecvers_) {
    for (auto &kv : domRecvers) {
      LOG_SPEW("rank=" << rank_ << " domain=" << kv.first << " wait colocated recver");
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
      LOG_SPEW("rank=" << rank_ << " domain=" << kv.first << " wait remote recver");
      StatefulRecver *recver = kv.second;
      assert(recver);
      recver->wait();
    }
  }
  for (auto &domSenders : remoteSenders_) {
    for (auto &kv : domSenders) {
      LOG_SPEW("rank=" << rank_ << " domain=" << kv.first << " wait remote sender");
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

  // wait for all ranks to be done
  // TODO: this should be safe to remove
  LOG_SPEW("rank=" << rank_ << " post-exchange barrier");
  nvtxRangePush("barrier");
  MPI_Barrier(MPI_COMM_WORLD);
  nvtxRangePop(); // barrier
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
    const std::string path = prefix + std::to_string(id) + ".txt";

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