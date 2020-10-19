#include "stencil/tx_cuda.cuh"

void CudaAwareMpiSender::wait() {
  assert(State::Send == state_);
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));
  MPI_Status status;
  LOG_SPEW("CudaAwareMpiSender::wait(): dstRank=" << dstRank_);
  MPI_Wait(&req_, &status);
  state_ = State::None;
}

void CudaAwareMpiSender::send_pack() {
  nvtxRangePush("CudaAwareMpiSender::send_pack");
  assert(packer_.data());
  packer_.pack();
  nvtxRangePop(); // CudaAwareMpiSender::send_pack
}

bool CudaAwareMpiSender::pack_done() {
  assert(packer_.size());
  cudaError_t err = rt::time(cudaStreamQuery, stream_);
  if (cudaSuccess == err) {
    return true;
  } else if (cudaErrorNotReady == err) {
    return false;
  } else {
    CUDA_RUNTIME(err);
    LOG_FATAL("cuda error");
  }
}

void CudaAwareMpiSender::send_d2d() {
  assert(packer_.size());
  nvtxRangePush("CudaAwareMpiSender::send_d2d");
  assert(packer_.data());
  assert(srcGPU_ < 8);
  assert(dstGPU_ < 8);
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));
  const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
  size_t numBytes = packer_.size();
  assert(numBytes <= std::numeric_limits<int>::max());
  LOG_SPEW("CudaAwareMpiSender::send_d2d() MPI_Isend t=" << tag << " r=" << dstRank_);
  mpirt::time(MPI_Isend, packer_.data(), int(numBytes), MPI_BYTE, dstRank_, tag, MPI_COMM_WORLD, &req_);
  nvtxRangePop(); // CudaAwareMpiSender::send_d2d
}

void CudaAwareMpiRecver::recv_d2d() {
  assert(unpacker_.size());
  nvtxRangePush("CudaAwareMpiRecver::recv_d2d");
  assert(unpacker_.data());
  assert(srcGPU_ < 8);
  assert(dstGPU_ < 8);
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));
  const int tag = ((srcGPU_ & 0xF) << 4) | (dstGPU_ & 0xF);
  LOG_SPEW("CudaAwareMpiSender::recv_d2d() MPI_Irecv t=" << tag << " r=" << srcRank_);
  mpirt::time(MPI_Irecv, unpacker_.data(), int(unpacker_.size()), MPI_BYTE, srcRank_, tag, MPI_COMM_WORLD, &req_);
  nvtxRangePop(); // CudaAwareMpiRecver::recv_d2d
}

void CudaAwareMpiRecver::recv_unpack() {
  assert(unpacker_.size());
  nvtxRangePush("CudaAwareMpiRecver::recv_unpack");
  unpacker_.unpack();
  nvtxRangePop(); // CudaAwareMpiRecver::recv_unpack
}

bool CudaAwareMpiRecver::d2d_done() {
  assert(unpacker_.size());
  int flag;
  CUDA_RUNTIME(rt::time(cudaSetDevice, domain_->gpu()));
  mpirt::time(MPI_Test, &req_, &flag, MPI_STATUS_IGNORE);
  if (flag) {
    return true;
  } else {
    return false;
  }
}