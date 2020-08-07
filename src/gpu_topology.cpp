#include "stencil/gpu_topology.hpp"
#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"
#include "stencil/mat2d.hpp"
#include "stencil/nvml.hpp"

#include <iostream>
#include <map>
#include <set>

namespace gpu_topo {
namespace detail {
std::map<int, std::map<int, bool>> peer_;

/* return the distance between two GPUs
 */
inline double get_gpu_distance(const int src, const int dst) {
  nvml::lazy_init();

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

}; // namespace detail

double bandwidth(int src, int dst) { return 1.0 / detail::get_gpu_distance(src, dst); }

void enable_peer(const int src, const int dst) {
  if (src == dst) {
    detail::peer_[src][dst] = true;
    detail::peer_[dst][src] = true;
    LOG_DEBUG(src << " -> " << dst << " peer access (same)");
  } else {
    int canAccess;
    LOG_SPEW("check peer " << src << "->" << dst);
    CUDA_RUNTIME(cudaDeviceCanAccessPeer(&canAccess, src, dst));
    if (canAccess) {
      CUDA_RUNTIME(cudaSetDevice(src));
      cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0 /*flags*/);
      if (cudaSuccess == err || cudaErrorPeerAccessAlreadyEnabled == err) {
        cudaGetLastError(); // clear the error
        detail::peer_[src][dst] = true;
        detail::peer_[dst][src] = true; // assume this will be enabled
        LOG_DEBUG(src << " -> " << dst << " peer access (peer)");
      } else if (cudaErrorInvalidDevice == err) {
        cudaGetLastError(); // clear the error
        detail::peer_[src][dst] = false;
        LOG_WARN(src << " -> " << dst << " (invalid device)");
      } else {
        detail::peer_[src][dst] = false;
        CUDA_RUNTIME(err);
        LOG_FATAL("Unexpected CUDA runtime error" << err);
      }
    } else {
      detail::peer_[src][dst] = false;
      LOG_DEBUG(src << " -> " << dst << " (no access)");
    }
  }
  CUDA_RUNTIME(cudaGetLastError());
};

bool peer(const int src, const int dst) {
  if (0 == detail::peer_.count(src)) {
    return false;
  } else if (0 == detail::peer_[src].count(dst)) {
    return false;
  } else {
    return detail::peer_[src][dst];
  }
}

} // namespace gpu_topo
