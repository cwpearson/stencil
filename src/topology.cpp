#include "stencil/topology.hpp"

Topology::Topology() : Topology(Dim3(0, 0, 0), Boundary::NONE) {}

Topology::OptionalNeighbor Topology::get_neighbor(const Dim3 &index, const Dim3 &dir) const noexcept {
  assert(dir.all_lt(2));
  assert(dir.all_gt(-2));
  assert(index.all_ge(0));

  if (Boundary::PERIODIC == boundary_) {
    OptionalNeighbor nbr;
    nbr.exists = true; // everyone has a nbr in every dir for period boundary conditions
    nbr.index = (index + dir).wrap(extent_);
    return nbr;
  } else {
    LOG_FATAL("unexpected Boundary type");
  }
}