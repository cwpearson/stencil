#include "stencil/stencil.hpp"

#include <vector>

/* start with the whole compute region, and check each direction of the
stencil. move the correponding face/edge/corner inward enough to compensate
an access in that direction
*/
std::vector<Rect3> DistributedDomain::get_interior() const {

  // one sparse domain for each LocalDomain
  std::vector<Rect3> ret(domains_.size());

  // direction of our halo
  for (size_t di = 0; di < domains_.size(); ++di) {
    const LocalDomain &dom = domains_[di];

    const Rect3 comReg = dom.get_compute_region();
    Rect3 intReg = dom.get_compute_region();

    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const Dim3 dir(dx, dy, dz);
          if (Dim3(0, 0, 0) == dir) {
            continue;
          }

          // if the radius is non-zero in a negative direction,
          // move the lower corner of that direction inward
          if (dir.x < 0) {
            intReg.lo.x = std::max(comReg.lo.x + int64_t(radius_.dir(dir)), intReg.lo.x);
          } else if (dir.x > 0) {
            intReg.hi.x = std::min(comReg.hi.x - int64_t(radius_.dir(dir)), intReg.hi.x);
          }
          if (dir.y < 0) {
            intReg.lo.y = std::max(comReg.lo.y + int64_t(radius_.dir(dir)), intReg.lo.y);
          } else if (dir.y > 0) {
            intReg.hi.y = std::min(comReg.hi.y - int64_t(radius_.dir(dir)), intReg.hi.y);
          }
          if (dir.z < 0) {
            intReg.lo.z = std::max(comReg.lo.z + int64_t(radius_.dir(dir)), intReg.lo.z);
          } else if (dir.z > 0) {
            intReg.hi.z = std::min(comReg.hi.z - int64_t(radius_.dir(dir)), intReg.hi.z);
          }
        }
      }
    }
    ret[di] = intReg;
  }
  return ret;
}

/* the exterior is everything that is not in the interior.
   build non-overlapping regions by sliding faces of the compute region in
   until they reach the interior
*/
std::vector<std::vector<Rect3>> DistributedDomain::get_exterior() const {

  // one sparse domain for each LocalDomain
  std::vector<std::vector<Rect3>> ret(domains_.size());

  const std::vector<Rect3> intRegs = get_interior();

  for (size_t di = 0; di < domains_.size(); ++di) {
    const LocalDomain &dom = domains_[di];
    const Rect3 &intReg = intRegs[di];
    Rect3 comReg = dom.get_compute_region();

    // +x
    if (intReg.hi.x != comReg.hi.x) {
      Rect3 extReg(Dim3(intReg.hi.x, intReg.lo.y, intReg.lo.z), Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.x = intReg.hi.x; // slide face in
      ret[di].push_back(extReg);
    }
    // +y
    if (intReg.hi.y != comReg.hi.y) {
      Rect3 extReg(Dim3(intReg.lo.x, intReg.hi.y, intReg.lo.z), Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.y = intReg.hi.y; // slide face in
      ret[di].push_back(extReg);
    }
    // +z
    if (intReg.hi.z != comReg.hi.z) {
      Rect3 extReg(Dim3(intReg.lo.x, intReg.lo.y, intReg.hi.z), Dim3(comReg.hi.x, comReg.hi.y, comReg.hi.z));
      comReg.hi.z = intReg.hi.z; // slide face in
      ret[di].push_back(extReg);
    }
    // -x
    if (intReg.lo.x != comReg.lo.x) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), Dim3(intReg.lo.x, intReg.hi.y, intReg.hi.z));
      comReg.lo.x = intReg.lo.x; // slide face in
      ret[di].push_back(extReg);
    }
    // -y
    if (intReg.lo.y != comReg.lo.y) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), Dim3(intReg.hi.x, intReg.lo.y, intReg.hi.z));
      comReg.lo.y = intReg.lo.y; // slide face in
      ret[di].push_back(extReg);
    }
    // -z
    if (intReg.lo.z != comReg.lo.z) {
      Rect3 extReg(Dim3(comReg.lo.x, comReg.lo.y, comReg.lo.z), Dim3(intReg.hi.x, intReg.hi.y, intReg.lo.z));
      comReg.lo.z = intReg.lo.z; // slide face in
      ret[di].push_back(extReg);
    }
  }
  return ret;
}

const Rect3 DistributedDomain::get_compute_region() const noexcept { return Rect3(Dim3(0, 0, 0), size_); }