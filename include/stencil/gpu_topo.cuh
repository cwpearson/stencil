#pragma once

#include "stencil/cuda_runtime.hpp"
#include "stencil/nvml.hpp"

using Mat2D = std::vector<std::vector<double>>;

inline double get_gpu_distance(const int src, const int dst) {
  const double SAME_DISTANCE = 0.0;
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
      NVML(nvmlDeviceGetHandleByPciBusId(pci.busId, &remote));
      if (remote == dstDev) {
        return NVLINK_DISTANCE;
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

inline Mat2D get_gpu_distance_matrix() {
  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));
  Mat2D dist(devCount, std::vector<double>(devCount));
  // build a distance matrix for GPUs
  for (int src = 0; src < devCount; ++src) {
    for (int dst = 0; dst < devCount; ++dst) {
      dist[src][dst] = get_gpu_distance(src, dst);
    }
  }
  return dist;
}
