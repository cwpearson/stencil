#include "stencil/local_domain.cuh"

void LocalDomain::set_device(CudaErrorsFatal fatal) {
  cudaError_t err = cudaSetDevice(dev_);
  if (CudaErrorsFatal::YES == fatal) {
    CUDA_RUNTIME(err)
  } else {
    (void) err;
  }
}

Rect3 LocalDomain::halo_coords(const Dim3 &dir, const bool halo) const {
  /* convert halo position as offset from allocation to coordinates
  */

  // get the offset of the halo from the allocation
  Dim3 pos = halo_pos(dir, halo);
  Dim3 ext = halo_extent(dir);

  // convert to offset from the origin
  // the size of the negative halo is equal to the size of the negative kernel radius
  // translate 
  pos.z -= radius_.z(-1);
  pos.y -= radius_.y(-1);
  pos.x -= radius_.x(-1);

  // shift by the origin
  pos += origin_;

  return Rect3(pos, pos+ext);
}

Rect3 LocalDomain::get_compute_region() const noexcept {
  Dim3 lo = origin();
  Dim3 hi = origin() + size();
  return Rect3(lo, hi);
}