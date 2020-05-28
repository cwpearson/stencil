#pragma once

#include <cstdio>

#include <nvml.h>

inline void checkNvml(nvmlReturn_t result, const char *file, const int line) {
  if (result != NVML_SUCCESS) {
    fprintf(stderr, "nvml Error: %s in %s : %d\n", nvmlErrorString(result),
            file, line);
    exit(-1);
  }
}

#define NVML(stmt) checkNvml(stmt, __FILE__, __LINE__);

namespace nvml{

inline void lazy_init() {
  static bool once = false;
  if (!once) {
    NVML(nvmlInit());
    once = true;
  }
}

}
