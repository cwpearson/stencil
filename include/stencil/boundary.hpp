#pragma once

#include <cassert>

#include "stencil/direction_map.hpp"

class Boundary {

private:

  DirectionMap<bool> dirs_;

public:

  /* periodic boundary by default */
  Boundary() : dirs_(true) {}

  void set_dir(int x, int y, int z, bool b) noexcept {
    dirs_.at_dir(x, y, z) = b;
    dirs_.at_dir(-1 * x, -1 * y, -1 * z) = b;
  }

  bool wraps(int x, int y, int z) const {
    assert(x != 0 && y != 0 && z != 0);
    return dirs_.at_dir(x, y, z);
  }
};