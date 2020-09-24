
#pragma once

#include "stencil/dim3.hpp"
#include "stencil/logging.hpp"

/*! \class provide information about distributed stencil layout
 */
class Topology {

public:
  enum class Boundary {
    NONE, // invalid
    PERIODIC
  };

  struct OptionalNeighbor {
    Dim3 index;
    bool exists;
  };

  Topology();
  Topology(const Dim3 &extent, const Boundary &boundary) : extent_(extent), boundary_(boundary) {}

  OptionalNeighbor get_neighbor(const Dim3 &index, const Dim3 &dir) const noexcept;

private:
  Dim3 extent_;
  Boundary boundary_;
};