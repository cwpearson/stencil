#pragma once

#if STENCIL_USE_MPI == 1
#include <mpi.h>
#endif

#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"

#include <nvml.h>

#include <cstring>
#include <ostream>
#include <string>
#include <vector>

struct UUID {

  UUID() : bytes_{} {}
  UUID(const char bytes[16]) { std::memcpy(bytes_, bytes, 16); }
  /* copy up to 16 bytes to form a UUID
   */
  UUID(const void *bytes, size_t n) {
    if (n > 16) {
      n = 16;
    }
    std::memcpy(bytes_, bytes, n);
  }

  char bytes_[16]; // if this is char printing is weird sometimes

  bool operator<(const UUID &rhs) const { return std::strncmp(bytes_, rhs.bytes_, 16) < 0; }

  friend std::ostream &operator<<(std::ostream &os, const UUID &uuid);

  explicit operator std::string() const {
    char buf[32+4+1];

    unsigned char uc[16];
    std::memcpy(uc, bytes_, 16);
    sprintf(&buf[0], "%02x", uc[0]);
    sprintf(&buf[2], "%02x", uc[1]);
    sprintf(&buf[4], "%02x", uc[2]);
    sprintf(&buf[6], "%02x", uc[3]);
    sprintf(&buf[8], "-");
    sprintf(&buf[9], "%02x", uc[4]);
    sprintf(&buf[11], "%02x", uc[5]);
    sprintf(&buf[13], "-");
    sprintf(&buf[14], "%02x%02x", uc[6], uc[7]);
    sprintf(&buf[18], "-");
    sprintf(&buf[19], "%02x%02x", uc[8], uc[9]);
    sprintf(&buf[23], "-");
    sprintf(&buf[24], "%02x%02x%02x%02x%02x%02x", uc[10], uc[11], uc[12], uc[13], uc[14], uc[15]);
    // sprintf writes trailing 0
    return std::string(buf);
  }
};



class GPU {
public:
  typedef unsigned index_t;

  GPU() = default;
  GPU(const UUID &uuid, const std::vector<int> &ranks) : uuid_(uuid), ranks_(ranks) {}

  /* retrieve an nvmlDevice_t for this GPU.
   */
  nvmlDevice_t nvml_device();

  /* retireve an nvml index for this GPU
   */
  unsigned int nvml_index();

  /* retrieve a CUDA index for this GPU
   */
  int cuda_index();

  /* retrieve the ranks that can access this GPU
   */
  const std::vector<int> ranks() const { return ranks_; }

  const UUID &uuid() const noexcept { return uuid_; }

private:
  // cuda/nvml UUID for this GPU
  UUID uuid_;
  std::vector<int> ranks_; // ranks that can access this GPU
};
#if 0
      // hex str of uuid. 2 hex chars per byte
      char uuidStr[sizeof(prop.uuid.bytes) * 2 + 1] = {};

      for (unsigned i = 0; i < sizeof(prop.uuid.bytes); ++i) {
        snprintf(&uuidStr[2 * i], 3 /*max 2 bytes,+1 NULL*/, "%02x", prop.uuid.bytes[i]);
      }
#endif

/* some information about the machine we're running on
   nodes indexed as 0..<N

    gpus are indexed 0..<N

*/
class Machine {
public:
  struct Distance {
    double bandwidth;
    double latency;
  };

private:
  std::vector<std::string> hostnames_; // hostname of each node
  std::vector<int> nodeOfRank_;        // node of each rank
  std::vector<GPU> gpus_;              // all GPUs in the machine

public:
#if STENCIL_USE_MPI == 1
  /* build a model of a machine visible to `comm`
   */
  static Machine build(MPI_Comm comm);
#endif

  /* build a model of a machine without using MPI
   */
  static Machine build();

  /* return the distance between two GPUs in the machine
   */
  Distance gpu_distance(const unsigned srcId, const unsigned dstId) const;

  int node_of_gpu(GPU::index_t i) const noexcept { return nodeOfRank_[gpus_[i].ranks()[0]]; }

  int num_nodes() const noexcept { return hostnames_.size(); }
  int num_ranks() const noexcept { return nodeOfRank_.size(); }
  int num_gpus() const noexcept { return gpus_.size(); }
  int node_of_rank(const int rank) const noexcept { return nodeOfRank_[rank]; }
  const GPU &gpu(const GPU::index_t i) const noexcept { return gpus_[i]; }
};