#pragma once

#include <string>

enum class Method : int {
  None = 0,
  CudaMpi = 1,
  CudaAwareMpi = 2,
  ColoPackMemcpyUnpack = 4,
  ColoDirectAccess = 8,
  CudaMemcpyPeer = 16,
  CudaKernel = 32,
#if STENCIL_USE_CUDA_AWARE_MPI == 1
  All = 1 + 2 + 4 + 16 + 32
#else
  All = 1 + 4 + 16 + 32
#endif
};

inline Method operator|(Method a, Method b) { return static_cast<Method>(static_cast<int>(a) | static_cast<int>(b)); }

inline Method &operator|=(Method &a, Method b) {
  a = a | b;
  return a;
}

inline Method operator&(Method a, Method b) { return static_cast<Method>(static_cast<int>(a) & static_cast<int>(b)); }

inline bool operator&&(Method a, Method b) { return (a & b) != Method::None; }

inline bool any(Method a) noexcept { return a != Method::None; }

inline std::string to_string(const Method &m) {

  std::string ret;
  const std::string sep("|");

  if (m == Method::None) {
    return "";
  }

  if (m && Method::CudaMpi) {
    ret += ret.empty() ? "" : sep;
    ret += "staged";
  }
  if (m && Method::CudaAwareMpi) {
    ret += ret.empty() ? "" : sep;
    ret += "cuda-aware";
  }
  if (m && Method::ColoPackMemcpyUnpack) {
    ret += ret.empty() ? "" : sep;
    ret += "colo-pmu";
  }
  if (m && Method::ColoDirectAccess) {
    ret += ret.empty() ? "" : sep;
    ret += "colo-da";
  }
  if (m && Method::CudaMemcpyPeer) {
    ret += ret.empty() ? "" : sep;
    ret += "peer";
  }
  if (m && Method::CudaKernel) {
    ret += ret.empty() ? "" : sep;
    ret += "kernel";
  }

  return ret;
}