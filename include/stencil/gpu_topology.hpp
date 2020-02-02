#pragma once

#include <iostream>
#include <map>
#include <set>

#include "stencil/cuda_runtime.hpp"
#include "stencil/mat2d.hpp"
#include "stencil/nvml.hpp"

/* return the distance between two GPUs
 */
inline double get_gpu_distance(const int src, const int dst) {
  const double SAME_DISTANCE = 0.1;
  const double NVLINK_DISTANCE = 1.0;
  const double INTERNAL_DISTANCE = 2.0;
  const double SINGLE_DISTANCE = 3.0;
  const double MULTIPLE_DISTANCE = 4.0;
  const double HOSTBRIDGE_DISTANCE = 5.0;
  const double NODE_DISTANCE = 6.0;
  const double SYSTEM_DISTANCE = 7.0;

  if (src == dst) {
    return SAME_DISTANCE;
  }

  // get NVML device handles
  nvmlDevice_t srcDev, dstDev;
  NVML(nvmlDeviceGetHandleByIndex(src, &srcDev));
  NVML(nvmlDeviceGetHandleByIndex(dst, &dstDev));

  for (int link = 0; link < 6; ++link) {
    nvmlPciInfo_t pci;
    nvmlReturn_t ret = nvmlDeviceGetNvLinkRemotePciInfo(srcDev, link, &pci);
    if (NVML_SUCCESS == ret) {
      // use remote PCI to see if remote device is dst
      nvmlDevice_t remote;
      nvmlReturn_t getRet = nvmlDeviceGetHandleByPciBusId(pci.busId, &remote);
      if (NVML_SUCCESS == getRet) {
        if (remote == dstDev) {
          return NVLINK_DISTANCE;
        }
      } else if (NVML_ERROR_NOT_FOUND == getRet) {
        // not attached to a GPU
        continue; // try next link
      } else {
        NVML(getRet);
      }

    } else if (NVML_ERROR_INVALID_ARGUMENT == ret) {
      // device is invalid
      // link is invalid
      // pci is null
      continue;
    } else if (NVML_ERROR_NOT_SUPPORTED == ret) {
      // device does not support
      break;
    } else {
      NVML(ret);
    }
  }

  // no nvlink, try other methods
  nvmlGpuTopologyLevel_t pathInfo;
  NVML(nvmlDeviceGetTopologyCommonAncestor(srcDev, dstDev, &pathInfo));
  switch (pathInfo) {
  case NVML_TOPOLOGY_INTERNAL:
    return INTERNAL_DISTANCE;
  case NVML_TOPOLOGY_SINGLE:
    return SINGLE_DISTANCE;
  case NVML_TOPOLOGY_MULTIPLE:
    return MULTIPLE_DISTANCE;
  case NVML_TOPOLOGY_HOSTBRIDGE:
    return HOSTBRIDGE_DISTANCE;
  case NVML_TOPOLOGY_NODE:
    return NODE_DISTANCE;
  case NVML_TOPOLOGY_SYSTEM:
    return SYSTEM_DISTANCE;
  default:
    assert(0);
  }

  assert(0);
  return -1;
}

class GpuTopology {
private:
  std::vector<int> ids_; // cuda device IDs, not necessarily unique

  std::map<int, std::map<int, bool>> peer_;

public:
  /*! gather GPU topology information for the provided GPUs
   */
  GpuTopology(const std::vector<int> &ids) {

    auto unique = std::set<int>(ids.begin(), ids.end());
    ids_ = std::vector<int>(unique.begin(), unique.end());

    for (auto src : ids_) {
      for (auto dst : ids_) {
        if (src == dst) {
          peer_[src][dst] = true;
        } else {
          peer_[src][dst] = false;
        }
      }
    }
  }
  GpuTopology() : GpuTopology(std::vector<int>()) {}

  const std::vector<int> &devices() const noexcept { return ids_; }

  /* enable peer access where possible
   */
  void enable_peer() {

    for (auto src : ids_) {
      for (auto dst : ids_) {
        if (src == dst) {
          peer_[src][dst] = true;
          peer_[dst][src] = true;
          std::cerr << src << " -> " << dst << " peer access\n";
          std::cerr << dst << " -> " << src << " peer access\n";
        } else {
          int canAccess;
          CUDA_RUNTIME(cudaDeviceCanAccessPeer(&canAccess, src, dst));
          if (canAccess) {
            CUDA_RUNTIME(cudaSetDevice(src))
            cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0 /*flags*/);
            if (cudaSuccess == err ||
                cudaErrorPeerAccessAlreadyEnabled == err) {
              peer_[src][dst] = true;
              std::cout << src << " -> " << dst << " peer access\n";
            } else if (cudaErrorInvalidDevice) {
              peer_[src][dst] = false;
            } else {
              assert(0);
              peer_[src][dst] = false;
            }
          } else {
            peer_[src][dst] = false;
          }
        }
      }
    }
  }

  bool peer(int src, int dst) { return peer_[src][dst]; }

  double bandwidth(int src, int dst) {
    return 1.0 / get_gpu_distance(src, dst);
  }

  Mat2D<double> bandwidth_matrix() {
    std::cerr << "bw matrix for";
    for (auto &id : ids_) {
      std::cerr << " " << id;
    }
    std::cerr << "\n";

    Mat2D<double> ret(ids_.size());
    for (auto &r : ret) {
      r.resize(ids_.size());
    }

    for (size_t i = 0; i < ids_.size(); ++i) {
      for (size_t j = 0; j < ids_.size(); ++j) {
        double bw = bandwidth(ids_[i], ids_[j]);
        ret[i][j] = bw;
        ret[j][i] = bw;
      }
    }
    return ret;
  }
};


