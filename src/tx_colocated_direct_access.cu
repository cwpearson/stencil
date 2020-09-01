#include "stencil/tx_colocated_direct_access.cuh"

#include "stencil/copy.cuh"

#include <nvToolsExt.h>

#include <algorithm>

ColocatedDirectAccessSender::~ColocatedDirectAccessSender() {
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));

  // free cuda malloc
  CUDA_RUNTIME(cudaFree(dstDomCurrDatasDev_));

  // free mem handle
  for (void *ptr : dstDomCurrDatas_) {
    CUDA_RUNTIME(cudaIpcCloseMemHandle(ptr));
  }
  dstDomCurrDatas_.clear();
}

// FIXME: one of these per rank, so needs to work for all domains
ColocatedDirectAccessSender::ColocatedDirectAccessSender(int srcRank, int srcDom, int dstRank, int dstDom,
                                                         LocalDomain &domain, Placement *placement)
    : srcRank_(srcRank), dstRank_(dstRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain), placement_(placement),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcSender_(srcRank, srcDom, dstRank, dstDom, domain.gpu()),
      dstDomCurrDatasDev_(nullptr) {}

void ColocatedDirectAccessSender::start_prepare() {
  // Post recieve the memhandles for the destination buffers
  const int memHandleTag = make_tag<MsgKind::ColocatedMem>(ipc_tag_payload(srcDom_, dstDom_));
  handles_.resize(domain_->num_data());
  MPI_Irecv(handles_.data(), handles_.size() * sizeof(handles_[0]), MPI_BYTE, dstRank_, memHandleTag, MPI_COMM_WORLD,
            &memReq_);

  ipcSender_.async_prepare();
}

void ColocatedDirectAccessSender::finish_prepare() {
  // recieve mem handles
  MPI_Wait(&memReq_, MPI_STATUS_IGNORE);

  // convert to pointers
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  for (cudaIpcMemHandle_t &handle : handles_) {
    void *ptr;
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    dstDomCurrDatas_.push_back(ptr);
  }

  // push pointers to device so they can be used in the kernel
  CUDA_RUNTIME(cudaMalloc(&dstDomCurrDatasDev_, dstDomCurrDatas_.size() * sizeof(dstDomCurrDatas_[0])));
  CUDA_RUNTIME(cudaMemcpy(dstDomCurrDatasDev_, dstDomCurrDatas_.data(),
                          dstDomCurrDatas_.size() * sizeof(dstDomCurrDatas_[0]), cudaMemcpyHostToDevice));

  ipcSender_.wait_prepare();
}

void ColocatedDirectAccessSender::send() {

  nvtxRangePush("ColocatedDirectAccessSender::send");

  // get the souce & dst index
  const Dim3 srcIdx = placement_->get_idx(srcRank_, srcDom_);
  const Dim3 dstIdx = placement_->get_idx(dstRank_, dstDom_);

  // get direction
  const Dim3 dir = (dstIdx - srcIdx).wrap(placement_->dim());
  assert(dir.all_gt(-2));
  assert(dir.all_lt(2));

  // determine the size of the destination
  const Dim3 dstSz = placement_->subdomain_size(dstIdx);
  const Dim3 dstPos = LocalDomain::halo_pos(dir * -1, dstSz, domain_->radius(), true /*exterior*/);

  const Dim3 srcSz = domain_->raw_size();
  const Dim3 srcPos = domain_->halo_pos(dir, false /*interior*/);
  const Dim3 extent = domain_->halo_extent(dir);

  const dim3 dimBlock = Dim3::make_block_dim(extent, 512 /*threads per block*/);
  const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
  assert(stream_.device() == domain_->gpu());
  CUDA_RUNTIME(cudaSetDevice(stream_.device()));
  LOG_SPEW("multi_translate grid=" << dimGrid << " block=" << dimBlock);
  multi_translate<<<dimGrid, dimBlock, 0, stream_>>>(dstDomCurrDatasDev_, dstPos, dstSz, domain_->dev_curr_datas(),
                                                     srcPos, srcSz, extent, domain_->dev_elem_sizes(),
                                                     domain_->num_data());
  CUDA_RUNTIME(cudaGetLastError());
  CUDA_RUNTIME(cudaEventRecord(ipcSender_.event(), stream_));
  ipcSender_.async_notify();

  nvtxRangePop(); // ColocatedDirectAccessSender::send
}

void ColocatedDirectAccessSender::wait() {
  ipcSender_.wait_notify();
  CUDA_RUNTIME(cudaEventSynchronize(ipcSender_.event()));
}

ColocatedDirectAccessRecver::ColocatedDirectAccessRecver(int srcRank, int srcDom, int dstRank, int dstDom,
                                                         LocalDomain &domain)
    : srcRank_(srcRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcRecver_(srcRank, srcDom, dstRank, dstDom, domain.gpu()),
      state_(State::NONE) {}

ColocatedDirectAccessRecver::~ColocatedDirectAccessRecver() {}

void ColocatedDirectAccessRecver::start_prepare() {
  // convert quantity pointers to handles
  for (void *ptr : domain_->curr_datas()) {
    cudaIpcMemHandle_t handle;
    CUDA_RUNTIME(cudaIpcGetMemHandle(&handle, ptr));
    handles_.push_back(handle);
  }

  // post send handles to source rank
  const int memTag = make_tag<MsgKind::ColocatedMem>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Isend(handles_.data(), handles_.size() * sizeof(handles_[0]), MPI_BYTE, srcRank_, memTag, MPI_COMM_WORLD,
            &memReq_);

  ipcRecver_.async_prepare();
}

void ColocatedDirectAccessRecver::finish_prepare() {

  // wait for mem handles to be sent
  MPI_Wait(&memReq_, MPI_STATUS_IGNORE);

  ipcRecver_.wait_prepare();
}

void ColocatedDirectAccessRecver::recv() {
  assert(State::NONE == state_);
  ipcRecver_.async_listen();
  state_ = State::WAIT_NOTIFY;
}

// once we are in the WAIT_KERNEL state, there's nothing else we need to do
bool ColocatedDirectAccessRecver::active() { return state_ == State::WAIT_NOTIFY; }

bool ColocatedDirectAccessRecver::next_ready() {
  if (State::WAIT_NOTIFY == state_) {
    return ipcRecver_.test_listen();
  } else { // should only be asked this in active() states
    LOG_FATAL("unexpected state");
  }
}

void ColocatedDirectAccessRecver::next() {
  if (State::WAIT_NOTIFY == state_) {
    state_ = State::WAIT_KERNEL;
  }
}

void ColocatedDirectAccessRecver::wait() {
  // wait on the event that the sender recorded after the kernel
  assert(stream_.device() == domain_->gpu());
  CUDA_RUNTIME(cudaEventSynchronize(ipcRecver_.event()));
  state_ = State::NONE;
}