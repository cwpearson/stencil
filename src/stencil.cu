#include "stencil/stencil.hpp"

#include <vector>

std::vector<std::vector<Rect3>> DistributedDomain::interior() {

  // one sparse domain for each LocalDomain
  std::vector<std::vector<Rect3>> ret(domains_.size());

  // direction of our halo
  for (size_t di = 0; di < domains_.size(); ++di) {
    LocalDomain &dom = domains_[di];
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {

          const Dim3 dir(dx, dy, dz);
          // some parts of the exterior may actually be interior if we dont talk to nbr
          const Rect3 rect = dom.halo_coords(dir, false /*exterior*/);

          // include interior
          if (Dim3(0, 0, 0) == dir) {
            ret[di].push_back(rect);
            continue;
          }

          // if we don't receive data, then we are interior
          // TODO: if no neighbor then part of interior
          // our +x boundary recieves data if the +x stencil radius is not 0
          if (0 == radius_.dir(dir)) {
            ret[di].push_back(rect);
            continue;
          }
        }
      }
    }
  }
  return ret;
}

std::vector<std::vector<Rect3>> DistributedDomain::exterior() {

  // one sparse domain for each LocalDomain
  std::vector<std::vector<Rect3>> ret(domains_.size());

  // direction of our halo
  for (size_t di = 0; di < domains_.size(); ++di) {
    LocalDomain &dom = domains_[di];
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {

          const Dim3 dir(dx, dy, dz);

          // skip interior
          if (Dim3(0, 0, 0) == dir) {
            continue; // skip self
          }

          // if we don't receive data, then not exterior
          // TODO: if no neighbor then part of interior
          // our +x boundary recieves data if the +x stencil radius is not 0
          else if (0 == radius_.dir(dir)) {
            continue;
          }

          else {
            ret[di].push_back(dom.halo_coords(dir, false /*exterior*/));
          }
        }
      }
    }
  }
  return ret;
}
