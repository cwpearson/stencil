#include "stencil/local_domain.cuh"

#include <nvToolsExt.h>

LocalDomain::LocalDomain(Dim3 sz, Dim3 origin, int dev)
    : sz_(sz), origin_(origin), dev_(dev), devCurrDataPtrs_(nullptr), devDataElemSize_(nullptr) {}

LocalDomain::~LocalDomain() {
  CUDA_RUNTIME(cudaGetLastError());

  CUDA_RUNTIME(cudaSetDevice(dev_));
  for (auto p : currDataPtrs_) {
    if (p)
      CUDA_RUNTIME(cudaFree(p));
  }
  if (devCurrDataPtrs_)
    CUDA_RUNTIME(cudaFree(devCurrDataPtrs_));

  for (auto p : nextDataPtrs_) {
    if (p)
      CUDA_RUNTIME(cudaFree(p));
  }
  if (devDataElemSize_)
    CUDA_RUNTIME(cudaFree(devDataElemSize_));
  CUDA_RUNTIME(cudaGetLastError());
}

void LocalDomain::set_device(CudaErrorsFatal fatal) {
  cudaError_t err = cudaSetDevice(dev_);
  if (CudaErrorsFatal::YES == fatal) {
    CUDA_RUNTIME(err)
  } else {
    (void)err;
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

  return Rect3(pos, pos + ext);
}

Rect3 LocalDomain::get_compute_region() const noexcept {
  Dim3 lo = origin();
  Dim3 hi = origin() + size();
  return Rect3(lo, hi);
}

void LocalDomain::swap() noexcept {
  nvtxRangePush("swap");

  // swap the host copy of the pointers
  assert(currDataPtrs_.size() == nextDataPtrs_.size());
  for (size_t i = 0; i < currDataPtrs_.size(); ++i) {
    std::swap(currDataPtrs_[i], nextDataPtrs_[i]);
  }

  // update the device version of the pointers
  CUDA_RUNTIME(cudaMemcpy(devCurrDataPtrs_, currDataPtrs_.data(), currDataPtrs_.size() * sizeof(currDataPtrs_[0]),
                          cudaMemcpyHostToDevice));
  nvtxRangePop();
}

Dim3 LocalDomain::halo_pos(const Dim3 &dir, const bool halo) const noexcept {
  assert(dir.all_gt(-2));
  assert(dir.all_lt(2));

  Dim3 ret;

  // +xhalo is the left edge + -x radius + the interior
  // +x interior is just the left edge + interior size
  if (1 == dir.x) {
    ret.x = sz_.x + (halo ? radius_.x(-1) : 0);
  } else if (-1 == dir.x) {
    ret.x = halo ? 0 : radius_.x(-1);
  } else if (0 == dir.x) {
    ret.x = radius_.x(-1);
  } else {
    LOG_FATAL("unreachable");
  }

  if (1 == dir.y) {
    ret.y = sz_.y + (halo ? radius_.y(-1) : 0);
  } else if (-1 == dir.y) {
    ret.y = halo ? 0 : radius_.y(-1);
  } else if (0 == dir.y) {
    ret.y = radius_.y(-1);
  } else {
    LOG_FATAL("unreachable");
  }

  if (1 == dir.z) {
    ret.z = sz_.z + (halo ? radius_.z(-1) : 0);
  } else if (-1 == dir.z) {
    ret.z = halo ? 0 : radius_.z(-1);
  } else if (0 == dir.z) {
    ret.z = radius_.z(-1);
  } else {
    LOG_FATAL("unreachable");
  }

  return ret;
}

std::vector<unsigned char> LocalDomain::region_to_host(const Dim3 &pos, const Dim3 &ext,
                                                       const size_t qi // quantity index
                                                       ) const {

  const size_t bytes = elem_size(qi) * ext.flatten();

  // pack quantity
  CUDA_RUNTIME(cudaSetDevice(gpu()));
  void *devBuf = nullptr;
  CUDA_RUNTIME(cudaMalloc(&devBuf, bytes));
  const dim3 dimBlock = Dim3::make_block_dim(ext, 512);
  const dim3 dimGrid = (ext + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
  pack_kernel<<<dimGrid, dimBlock>>>(devBuf, curr_data(qi), raw_size(), pos, ext, elem_size(qi));
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // copy quantity to host
  std::vector<unsigned char> hostBuf(bytes);
  CUDA_RUNTIME(cudaMemcpy(hostBuf.data(), devBuf, hostBuf.size(), cudaMemcpyDefault));

  float *ptr = reinterpret_cast<float *>(hostBuf.data());

  // free device buffer
  CUDA_RUNTIME(cudaFree(devBuf));

  return hostBuf;
}

void LocalDomain::realize() {
  LOG_SPEW("in realize()");
  CUDA_RUNTIME(cudaGetLastError());
  assert(currDataPtrs_.size() == nextDataPtrs_.size());
  assert(dataElemSize_.size() == nextDataPtrs_.size());

  LOG_INFO("origin is " << origin_);

  // allocate each data region
  CUDA_RUNTIME(cudaSetDevice(dev_));
  // int rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // std::cerr << "r" << rank << " dev=" << dev_ << "\n";
  for (int64_t i = 0; i < num_data(); ++i) {
    assert(i < dataElemSize_.size());
    int64_t elemSz = dataElemSize_[i];
    LOG_SPEW("elemSz=" << elemSz);
    LOG_SPEW("radius +x=" << radius_.x(1));
    LOG_SPEW("radius -x=" << radius_.x(-1));
    LOG_SPEW("radius +y=" << radius_.y(1));
    LOG_SPEW("radius -y=" << radius_.y(-1));
    LOG_SPEW("radius +z=" << radius_.z(1));
    LOG_SPEW("radius -z=" << radius_.z(-1));

    int64_t elemBytes = ((sz_.x + radius_.x(-1) + radius_.x(1)) * (sz_.y + radius_.y(-1) + radius_.y(1)) *
                         (sz_.z + radius_.z(-1) + radius_.z(1))) *
                        elemSz;
    LOG_SPEW("allocate " << elemBytes << " bytes");
    char *c = nullptr;
    char *n = nullptr;
    CUDA_RUNTIME(cudaMalloc(&c, elemBytes));
    CUDA_RUNTIME(cudaMalloc(&n, elemBytes));
    assert(uintptr_t(c) % elemSz == 0 && "allocation should be aligned");
    assert(uintptr_t(n) % elemSz == 0 && "allocation should be aligned");
    currDataPtrs_[i] = c;
    nextDataPtrs_[i] = n;
  }

  CUDA_RUNTIME(cudaMalloc(&devCurrDataPtrs_, currDataPtrs_.size() * sizeof(currDataPtrs_[0])));
  CUDA_RUNTIME(cudaMalloc(&devDataElemSize_, dataElemSize_.size() * sizeof(dataElemSize_[0])));
  CUDA_RUNTIME(cudaMemcpy(devCurrDataPtrs_, currDataPtrs_.data(), currDataPtrs_.size() * sizeof(currDataPtrs_[0]),
                          cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(devDataElemSize_, dataElemSize_.data(), dataElemSize_.size() * sizeof(dataElemSize_[0]),
                          cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaGetLastError());
}