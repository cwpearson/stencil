#include "stencil/tx_colocated_direct_access.cuh"

ColocatedDirectAccessSender::ColocatedDirectAccessSender(int srcRank, int srcDom, int dstRank, int dstDom,
                                                         LocalDomain &domain)
    : srcRank_(srcRank), dstRank_(dstRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcSender_(srcRank, srcDom, dstRank, dstDom, domain.gpu()) {}

void ColocatedDirectAccessSender::start_prepare(const std::vector<Message> &outbox) {
  if (0 == outbox.size()) {
    LOG_WARN("0-size ColocatedDirectAccessSender was created\n");
  }

  // Recieve the memhandles for the destination buffers
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
  for (auto cudaIpcMemHandle_t &handle : handles_) {
    void *ptr;
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    bufs_.push_back(ptr);
  }

  ipcSender_.wait_prepare();
}

void ColocatedDirectAccessSender::send() {

    nvtxRangePush("PeerSender::send");

    // translate data with kernel
    for (auto &msg : outbox_) {
      const LocalDomain *srcDomain = domains_[msg.srcGPU_];
      const LocalDomain *dstDomain = domains_[msg.dstGPU_];
      const Dim3 dstSz = dstDomain->raw_size();
      const Dim3 srcSz = srcDomain->raw_size();
      const Dim3 srcPos = srcDomain->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 dstPos = dstDomain->halo_pos(msg.dir_ * -1, true /*exterior*/);
      const Dim3 extent = srcDomain->halo_extent(msg.dir_);
      RcStream &stream = streams_[srcDomain->gpu()];
      const dim3 dimBlock = Dim3::make_block_dim(extent, 512 /*threads per block*/);
      const dim3 dimGrid = (extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      assert(stream.device() == srcDomain->gpu());
      CUDA_RUNTIME(cudaSetDevice(stream.device()));
      assert(srcDomain->num_data() == dstDomain->num_data());
      LOG_SPEW("multi_translate grid=" << dimGrid << " block=" << dimBlock);
      multi_translate<<<dimGrid, dimBlock, 0, stream>>>(dstDomain->dev_curr_datas(), dstPos, dstSz,
                                                        srcDomain->dev_curr_datas(), srcPos, srcSz, extent,
                                                        srcDomain->dev_elem_sizes(), srcDomain->num_data());
      CUDA_RUNTIME(cudaGetLastError());
    }

    nvtxRangePop(); // PeerSender::send
  }

  void ColocatedDirectAccessSender::wait() {

    for (auto &kv : streams_) {
      CUDA_RUNTIME(cudaSetDevice(kv.second.device()));
      CUDA_RUNTIME(cudaStreamSynchronize(kv.second));
    }
  }

ColocatedDirectAccessRecver::ColocatedDirectAccessRecver(int srcRank, int srcDom, int dstRank, int dstDom,
                                                         LocalDomain &domain)
    : srcRank_(srcRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcRecver_(srcRank, srcDom, dstRank, dstDom, domain.gpu()),
      state_(State::NONE) {}

void ColocatedDirectAccessRecver::start_prepare(const std::vector<Message> &inbox) {
  if (0 == inbox.size()) {
    LOG_WARN("a 0-size ColocatedDirectAccessRecver was created");
  }

  // convert quantity pointers to handles
  for (void *ptr : domain_->curr_datas()) {
    cudaIpcMemHandle_t handle;
    CUDA_RUNTIME(cudaIpcGetMemHandle(&handle, ptr));
    handles_.push_back(handle);
  }

  // send handles to source rank
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
    CUDA_RUNTIME(cudaStreamWaitEvent(stream_, ipcRecver_.event(), 0));
  }
}

void ColocatedDirectAccessRecver::wait() {
  // wait on unpacker
  assert(stream_.device() == domain_->gpu());
  CUDA_RUNTIME(cudaSetDevice(stream_.device()));
  CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  state_ = State::NONE;
}