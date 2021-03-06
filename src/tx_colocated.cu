#include "stencil/tx_colocated.cuh"

#include "stencil/copy.cuh"

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include <algorithm>

ColoHaloSender::ColoHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                               Placement *placement)
    : srcRank_(srcRank), dstRank_(dstRank), srcDom_(srcDom), dstDom_(dstDom), domain_(&domain), placement_(placement),
      stream_(domain.gpu(), RcStream::Priority::HIGH), ipcSender_(srcRank, srcDom, dstRank, dstDom, domain.gpu()),
      currTranslator_(nullptr), // derived class picks translator implementation
      nextTranslator_(nullptr)  // derived class picks translator implementation
{
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

  // free mem handle
  for (cudaPitchedPtr &p : dstDomCurrDatas_) {
    CUDA_RUNTIME(cudaIpcCloseMemHandle(p.ptr));
  }
  for (cudaPitchedPtr &p : dstDomNextDatas_) {
    CUDA_RUNTIME(cudaIpcCloseMemHandle(p.ptr));
  }
  dstDomCurrDatas_.clear();
  dstDomNextDatas_.clear();

  delete currTranslator_;
  currTranslator_ = nullptr;
  delete nextTranslator_;
  nextTranslator_ = nullptr;
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

  // Post recieve the memhandles for the curr and next destination buffers
  {
    const int memHandleTag = make_tag<MsgKind::ColocatedCurrMem>(ipc_tag_payload(srcDom_, dstDom_));
    currMemHandles_.resize(domain_->num_data());
    MPI_Irecv(currMemHandles_.data(), currMemHandles_.size() * sizeof(currMemHandles_[0]), MPI_BYTE, dstRank_,
              memHandleTag, MPI_COMM_WORLD, &currMemReq_);
  }
  {
    const int memHandleTag = make_tag<MsgKind::ColocatedNextMem>(ipc_tag_payload(srcDom_, dstDom_));
    nextMemHandles_.resize(domain_->num_data());
    MPI_Irecv(nextMemHandles_.data(), currMemHandles_.size() * sizeof(nextMemHandles_[0]), MPI_BYTE, dstRank_,
              memHandleTag, MPI_COMM_WORLD, &nextMemReq_);
  }

  // Post recieve for the pitch information. the curr and next data will have the same pitch, so we just get one copy.
  // the actual pointer values will be overwritten later after we recv mem handles
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
  MPI_Wait(&currMemReq_, MPI_STATUS_IGNORE);
  MPI_Wait(&nextMemReq_, MPI_STATUS_IGNORE);
  LOG_SPEW("ColoHaloSender::finish_prepare: got mem handles");

  // recieve pitch information
  LOG_SPEW("ColoHaloSender::finish_prepare: waiting for pitch information");
  MPI_Wait(&ptrReq_, MPI_STATUS_IGNORE);
  LOG_SPEW("ColoHaloSender::finish_prepare: got pitch info");

  // same pitch for curr and next data
  dstDomNextDatas_ = dstDomCurrDatas_;

  // convert to pointers.
  CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
  for (size_t i = 0; i < currMemHandles_.size(); ++i) {
    void *ptr = nullptr;
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&ptr, currMemHandles_[i], cudaIpcMemLazyEnablePeerAccess));
    dstDomCurrDatas_[i].ptr = ptr; // overwrite with ptr that is valid in this address space
  }
  LOG_SPEW("ColoHaloSender::finish_prepare: converted curr handles to pointers");
  for (size_t i = 0; i < nextMemHandles_.size(); ++i) {
    void *ptr = nullptr;
    CUDA_RUNTIME(cudaIpcOpenMemHandle(&ptr, nextMemHandles_[i], cudaIpcMemLazyEnablePeerAccess));
    dstDomNextDatas_[i].ptr = ptr; // overwrite with ptr that is valid in this address space
  }
  LOG_SPEW("ColoHaloSender::finish_prepare: converted next handles to pointers");

  std::vector<Translator::RegionParams> currParams, nextParams;
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

    // current
    Translator::RegionParams cp{.dstPtrs = dstDomCurrDatas_.data(),
                                .dstPos = dstPos,
                                .srcPtrs = domain_->curr_datas().data(),
                                .srcPos = srcPos,
                                .extent = extent,
                                .elemSizes = domain_->elem_sizes().data(),
                                .n = domain_->num_data()};
    currParams.push_back(cp);
    // next
    Translator::RegionParams np{.dstPtrs = dstDomNextDatas_.data(),
                                .dstPos = dstPos,
                                .srcPtrs = domain_->next_datas().data(),
                                .srcPos = srcPos,
                                .extent = extent,
                                .elemSizes = domain_->elem_sizes().data(),
                                .n = domain_->num_data()};
    nextParams.push_back(np);
  }
  LOG_SPEW("ColoHaloSender::finish_prepare: cvt outbox to params");
  assert(currTranslator_);
  currTranslator_->prepare(currParams);
  assert(nextTranslator_);
  nextTranslator_->prepare(nextParams);
  LOG_SPEW("ColoHaloSender::finish_prepare: prepared translators");

  ipcSender_.wait_prepare();
}

void ColoHaloSender::send() {
  nvtxRangePush("ColoHaloSender::send");
  assert(currTranslator_);
  currTranslator_->async(stream_);
  CUDA_RUNTIME(cudaEventRecord(ipcSender_.event(), stream_));
  ipcSender_.async_notify();
  nvtxRangePop(); // ColoHaloSender::send
}

void ColoHaloSender::wait() {
  ipcSender_.wait_notify();
  CUDA_RUNTIME(cudaEventSynchronize(ipcSender_.event()));
}

void ColoHaloSender::swap() {
  assert(!dstDomCurrDatas_.empty());
  assert(dstDomCurrDatas_.size() == dstDomNextDatas_.size());
  std::swap(dstDomCurrDatas_, dstDomNextDatas_);
  assert(currTranslator_);
  assert(nextTranslator_);
  std::swap(currTranslator_, nextTranslator_);
}

ColoMemcpy3dHaloSender::ColoMemcpy3dHaloSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                                               Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!currTranslator_);
  assert(!nextTranslator_);
  currTranslator_ = new TranslatorMemcpy3D();
  nextTranslator_ = new TranslatorMemcpy3D();
}

ColoQuantityKernelSender::ColoQuantityKernelSender(int srcRank, int srcDom, int dstRank, int dstDom,
                                                   LocalDomain &domain, Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!currTranslator_);
  assert(!nextTranslator_);
  currTranslator_ = new TranslatorKernel(domain.gpu());
  nextTranslator_ = new TranslatorKernel(domain.gpu());
}

ColoRegionKernelSender::ColoRegionKernelSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                                               Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!currTranslator_);
  assert(!nextTranslator_);
  currTranslator_ = new TranslatorMultiKernel(domain.gpu());
  nextTranslator_ = new TranslatorMultiKernel(domain.gpu());
}

ColoDomainKernelSender::ColoDomainKernelSender(int srcRank, int srcDom, int dstRank, int dstDom, LocalDomain &domain,
                                               Placement *placement)
    : ColoHaloSender(srcRank, srcDom, dstRank, dstDom, domain, placement) {
  assert(!currTranslator_);
  assert(!nextTranslator_);
  currTranslator_ = new TranslatorDomainKernel(domain.gpu());
  nextTranslator_ = new TranslatorDomainKernel(domain.gpu());
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
    currHandles_.push_back(handle);
  }
  for (const cudaPitchedPtr &p : domain_->next_datas()) {
    cudaIpcMemHandle_t handle;
    CUDA_RUNTIME(cudaIpcGetMemHandle(&handle, p.ptr));
    nextHandles_.push_back(handle);
  }

  // post send handles to source rank
  {
    const int memTag = make_tag<MsgKind::ColocatedCurrMem>(ipc_tag_payload(srcDom_, dstDom_));
    MPI_Isend(currHandles_.data(), currHandles_.size() * sizeof(currHandles_[0]), MPI_BYTE, srcRank_, memTag,
              MPI_COMM_WORLD, &currMemReq_);
  }
  LOG_SPEW("ColoHaloRecver::start_prepare posted curr mem handles to r=" << srcRank_);
  {
    const int memTag = make_tag<MsgKind::ColocatedNextMem>(ipc_tag_payload(srcDom_, dstDom_));
    MPI_Isend(nextHandles_.data(), nextHandles_.size() * sizeof(nextHandles_[0]), MPI_BYTE, srcRank_, memTag,
              MPI_COMM_WORLD, &nextMemReq_);
  }
  LOG_SPEW("ColoHaloRecver:start_prepare: posted next mem handles to r=" << srcRank_);

  // post send of pitch information. Only send one for curr/next because pitch will be the same for both, and the dst
  // will overwrite the pointers
  const int ptrTag = make_tag<MsgKind::ColocatedPtr>(ipc_tag_payload(srcDom_, dstDom_));
  MPI_Isend(domain_->curr_datas().data(), domain_->curr_datas().size() * sizeof(domain_->curr_datas()[0]), MPI_BYTE,
            srcRank_, ptrTag, MPI_COMM_WORLD, &ptrReq_);

  ipcRecver_.async_prepare();
}

void ColoHaloRecver::finish_prepare() {

  // wait for mem handles to be sent
  MPI_Wait(&currMemReq_, MPI_STATUS_IGNORE);
  MPI_Wait(&nextMemReq_, MPI_STATUS_IGNORE);

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
