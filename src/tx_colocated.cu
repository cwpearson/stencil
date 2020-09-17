#include "stencil/tx_colocated.cuh"

#include "stencil/copy.cuh"

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include <algorithm>

ColoHaloSender::ColoHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                               Placement *placement)
    : srcRank_(srcRank), dstRank_(dstRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain), placement_(placement),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcSender_(srcRank, srcDom, dstRank, dstDom, domain.gpu()),
      translate_(nullptr), // derived class picks translator implementation
      dstDomCurrDatasDev_(nullptr) {
  std::string streamName("ColoHaloSender_");
  streamName += "r" + std::to_string(srcRank);
  streamName += "d" + std::to_string(srcDom);
  streamName += "->r" + std::to_string(dstRank);
  streamName += "d" + std::to_string(dstDom);
  nvtxNameCudaStreamA(stream_, streamName.c_str());
}

ColoHaloSender::~ColoHaloSender() {
  assert(domain_);
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  // free cuda malloc
  CUDA_RUNTIME(cudaFree(dstDomCurrDatasDev_));
  dstDomCurrDatasDev_ = nullptr;

  // free mem handle
  for (cudaPitchedPtr &p : dstDomCurrDatas_) {
    CUDA_RUNTIME(cudaIpcCloseMemHandle(p.ptr));
  }
  dstDomCurrDatas_.clear();

  delete translate_;
  translate_ = nullptr;
}

void ColoHaloSender::start_prepare(const std::vector<Message> &outbox) {
  nvtxRangePush("ColoHaloSender::start_prepare");
  outbox_ = outbox;
  std::sort(outbox_.begin(), outbox_.end());
  // outbox should only have messages for our domain and the dst domain
  for (const Message &msg : outbox_) {
    assert(msg.srcGPU_ == srcDom_ && "outbox has a wrong message");
    assert(msg.dstGPU_ == dstDom_ && "outbox has a wrong message");
  }

  // Post recieve the memhandles for the destination buffers
  const int memHandleTag = make_tag<MsgKind::ColocatedMem>(ipc_tag_payload(srcDom_, dstDom_));
  memHandles_.resize(domain_->num_data());
  MPI_Irecv(memHandles_.data(), memHandles_.size() * sizeof(memHandles_[0]), MPI_BYTE, dstRank_, memHandleTag,
            MPI_COMM_WORLD, &memReq_);

  // Post recieve for the pitch information
  const int ptrHandleTag = make_tag<MsgKind::ColocatedPtr>(ipc_tag_payload(srcDom_, dstDom_));
  dstDomCurrDatas_.resize(domain_->num_data());
  MPI_Irecv(dstDomCurrDatas_.data(), dstDomCurrDatas_.size() * sizeof(dstDomCurrDatas_[0]), MPI_BYTE, dstRank_,
            ptrHandleTag, MPI_COMM_WORLD, &ptrReq_);

  ipcSender_.async_prepare();
  nvtxRangePop();
}

void ColoHaloSender::finish_prepare() {
  LOG_SPEW("ColoHaloSender::finish_prepare: waiting for mem handles...");
  // recieve mem handles
  MPI_Wait(&memReq_, MPI_STATUS_IGNORE);
  LOG_SPEW("ColoHaloSender::finish_prepare: got mem handles");

  // recieve pitch information
  LOG_SPEW("ColoHaloSender::finish_prepare: waiting for pitch information");
  MPI_Wait(&ptrReq_, MPI_STATUS_IGNORE);
  LOG_SPEW("ColoHaloSender::finish_prepare: got pitch info");

  // convert to pointers
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  for (size_t i = 0; i < memHandles_.size(); ++i) {
    void *ptr = nullptr;
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&ptr, memHandles_[i], cudaIpcMemLazyEnablePeerAccess));
    dstDomCurrDatas_[i].ptr = ptr; // overwrite with ptr that is valid in this address space
  }
  LOG_SPEW("ColoHaloSender::finish_prepare: converted to pointers");

  // push pointers to device so they can be used in the kernel
  CUDA_RUNTIME(cudaMalloc(&dstDomCurrDatasDev_, dstDomCurrDatas_.size() * sizeof(dstDomCurrDatas_[0])));
  CUDA_RUNTIME(cudaMemcpy(dstDomCurrDatasDev_, dstDomCurrDatas_.data(),
                          dstDomCurrDatas_.size() * sizeof(dstDomCurrDatas_[0]), cudaMemcpyHostToDevice));
  LOG_SPEW("ColoHaloSender::finish_prepare: pushed pointers");

  {
    std::vector<Translator::RegionParams> params;
    // get the dst idx;
    const Dim3 dstIdx = placement_->get_idx(dstRank_, dstDom_);

    for (const Message &msg : outbox_) {

      // the direction is not necessarily dst - src, since these domains could be neighbors in multiple directions
      // so use msg.dir_

      // determine the size of the destination
      const Dim3 dstSz = placement_->subdomain_size(dstIdx);
      const Dim3 dstPos = LocalDomain::halo_pos(msg.dir_ * -1, dstSz, domain_->radius(), true /*exterior*/);

      const Dim3 srcPos = domain_->halo_pos(msg.dir_, false /*interior*/);
      const Dim3 extent = domain_->halo_extent(msg.dir_);

      Translator::RegionParams p{.dstPtrs = dstDomCurrDatas_.data(),
                                 .dstPos = dstPos,
                                 .srcPtrs = domain_->curr_datas().data(),
                                 .srcPos = srcPos,
                                 .extent = extent,
                                 .elemSizes = domain_->elem_sizes().data(),
                                 .n = domain_->num_data()};
      params.push_back(p);
    }
    LOG_SPEW("ColoHaloSender::finish_prepare: cvt outbox to params");
    assert(translate_);
    translate_->prepare(params);
  }
  LOG_SPEW("ColoHaloSender::finish_prepare: prepared translator");

  ipcSender_.wait_prepare();
}

void ColoHaloSender::send() {
  nvtxRangePush("ColoHaloSender::send");
  assert(translate_);
  translate_->async(stream_);
  CUDA_RUNTIME(cudaEventRecord(ipcSender_.event(), stream_));
  ipcSender_.async_notify();
  nvtxRangePop(); // ColoHaloSender::send
}

void ColoHaloSender::wait() {
  ipcSender_.wait_notify();
  CUDA_RUNTIME(cudaEventSynchronize(ipcSender_.event()));
}

ColoMemcpy3dHaloSender::ColoMemcpy3dHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                                               Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!translate_);
  translate_ = new TranslatorMemcpy3D();
}

ColoQuantityKernelSender::ColoQuantityKernelSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                                   Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!translate_);
  translate_ = new TranslatorKernel(domain.gpu());
}

ColoRegionKernelSender::ColoRegionKernelSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                                             Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!translate_);
  translate_ = new TranslatorMultiKernel(domain.gpu());
}

ColoHaloRecver::ColoHaloRecver(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain)
    : srcRank_(srcRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcRecver_(srcRank, srcDom, dstRank, dstDom, domain.gpu()),
      state_(State::NONE) {}

ColoHaloRecver::~ColoHaloRecver() {}

void ColoHaloRecver::start_prepare(const std::vector<Message> &inbox) {

  // we don't do anything to recv messages
  (void)inbox;

  // convert quantity pointers to handles
  for (const cudaPitchedPtr &p : domain_->curr_datas()) {
    cudaIpcMemHandle_t handle;
    CUDA_RUNTIME(cudaIpcGetMemHandle(&handle, p.ptr));
    handles_.push_back(handle);
  }

  // post send handles to source rank
  const int memTag = make_tag<MsgKind::ColocatedMem>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Isend(handles_.data(), handles_.size() * sizeof(handles_[0]), MPI_BYTE, srcRank_, memTag, MPI_COMM_WORLD,
            &memReq_);

  // post send of pitch information
  const int ptrTag = make_tag<MsgKind::ColocatedPtr>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Isend(domain_->curr_datas().data(), domain_->curr_datas().size() * sizeof(domain_->curr_datas()[0]), MPI_BYTE,
            srcRank_, ptrTag, MPI_COMM_WORLD, &ptrReq_);

  ipcRecver_.async_prepare();
}

void ColoHaloRecver::finish_prepare() {

  // wait for mem handles to be sent
  MPI_Wait(&memReq_, MPI_STATUS_IGNORE);

  ipcRecver_.wait_prepare();
}

void ColoHaloRecver::recv() {
  assert(State::NONE == state_);
  ipcRecver_.async_listen();
  state_ = State::WAIT_NOTIFY;
}

// once we are in the WAIT_KERNEL state, there's nothing else we need to do
bool ColoHaloRecver::active() { return state_ == State::WAIT_NOTIFY; }

bool ColoHaloRecver::next_ready() {
  if (State::WAIT_NOTIFY == state_) {
    return ipcRecver_.test_listen();
  } else { // should only be asked this in active() states
    LOG_FATAL("unexpected state");
  }
}

void ColoHaloRecver::next() {
  if (State::WAIT_NOTIFY == state_) {
    state_ = State::WAIT_KERNEL;
  }
}

void ColoHaloRecver::wait() {
  // wait on the event that the sender recorded after the kernel
  assert(stream_.device() == domain_->gpu());
  CUDA_RUNTIME(cudaEventSynchronize(ipcRecver_.event()));
  state_ = State::NONE;
}
