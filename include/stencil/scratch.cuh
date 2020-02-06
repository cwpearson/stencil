/*! Send a LocalDomain region using Sender
 */
template <typename Sender> class HaloAnySender : public HaloSender {
private:
  const LocalDomain *domain_; // the domain we are sending from
  int dstRank_;
  int dstGPU_;
  Dim3 dir_; // the direction vector of the send

  std::future<void> fut_;

  std::vector<RcStream> streams_;

  // one flattened device buffer per domain data
  std::vector<char *> devBufs_;
  // one flattened host buffer per domain data
  std::vector<std::vector<char>> hostBufs_;
  // whether a stream has completed
  std::vector<bool> streamComplete_;

  std::vector<MPI_Request> reqs_;

public:
  HaloAnySender(const LocalDomain &domain, size_t srcRank, size_t srcGPU,
                size_t dstRank, size_t dstGPU, Dim3 dir)
      : domain_(&domain), dstRank_(dstRank), dstGPU_(dstGPU), dir_(dir),
        streams_(domain.num_data(), domain.gpu()), reqs_(domain.num_data()),
        streamComplete_(domain.num_data()) {

    assert(dir_.x >= -1 && dir.x <= 1);
    assert(dir_.y >= -1 && dir.y <= 1);
    assert(dir_.z >= -1 && dir.z <= 1);
  }

  virtual void allocate() override {

    const int gpu = domain_->gpu();
    CUDA_RUNTIME(cudaSetDevice(gpu));

    for (size_t i = 0; i < domain_->num_data(); ++i) {
      size_t numBytes = domain_->halo_bytes(dir_, i);
      char *buf = nullptr;
      CUDA_RUNTIME(cudaMalloc(&buf, numBytes));
      devBufs_.push_back(buf);
      hostBufs_.push_back(std::vector<char>(numBytes));
    }
  }

  void send_impl() {
    nvtxRangePush("HaloAnySender::send_impl");
    assert(bufs_.size() == senders_.size() && "was allocate called?");
    const Dim3 haloPos = domain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 haloExtent = domain_->halo_extent(dir_);
    const Dim3 rawSz = domain_->raw_size();

    for (size_t i = 0; i < streamComplete_.size(); ++i) {
      streamComplete_[i] = false;
    }

    // insert packs and memcpys into streams
    for (size_t i = 0; i < domain_->num_data(); ++i) {
      const char *src = domain_->curr_data(i);
      const size_t elemSize = domain_->elem_size(i);
      const size_t numBytes = domain_->halo_bytes(dir_, i);
      RcStream &stream = streams_[i];

      dim3 dimGrid(20, 20, 20);
      dim3 dimBlock(32, 4, 4);

      assert(stream.device() == domain_->gpu());
      CUDA_RUNTIME(cudaSetDevice(domain_->gpu()));
      pack<<<dimGrid, dimBlock, 0, stream>>>(
          devBufs_[i], src, rawSz, 0 /*pitch*/, haloPos, haloExtent, elemSize);
      CUDA_RUNTIME(cudaMemcpyAsync(hostBufs_[i].data(), devBufs_[i], numBytes,
                                   cudaMemcpyDefault, stream));
    }

    // poll streams for completion and issue MPI_Isend
    while (!std::all_of(streamComplete_.begin(), streamComplete_.end(),
                        [](bool b) { return b; })) {
      for (size_t i = 0; i < streams_.size(); ++i) {
        if (!streamComplete_[i]) {
          cudaError_t err = cudaStreamQuery(streams_[i]);
          if (cudaSuccess == err) {
            streamComplete_[i] = true;
            int tag = make_tag(dstGPU_, i, dir_);
            MPI_Isend(hostBufs_[i].data(), hostBufs_[i].size(), MPI_BYTE,
                      dstRank_, tag, MPI_COMM_WORLD, &reqs_[i]);
          } else if (cudaErrorNotReady == err) {
            continue;
          } else {
            CUDA_RUNTIME(err);
          }
        }
      }

      nvtxRangePop();
    }
  }

  void send() override {
    fut_ = std::async(std::launch::async, &HaloAnySender::send_impl, this);
  }

  // wait for send() to be complete
  void wait() override {
    // wait for all sends to be issued
    if (fut_.valid()) {
      fut_.wait();
#ifdef REGION_LOUD
      std::cerr << "HaloAnySender::wait(): " << dir_ << " done\n";
#endif
    } else {
      assert(0 && "wait called before send()");
    }
    for (auto &r : reqs_) {
      MPI_Wait(&r, MPI_STATUS_IGNORE);
    }
  }
};

/*! \brief Copy a LocalDomain region using pack/memcpy/unpack
 */
class PackMemcpyCopier : public HaloSender {
private:
  const LocalDomain *srcDomain_, *dstDomain_;

  Dim3 dir_;

  // one stream per domain data
  std::vector<RcStream> srcStreams_;
  std::vector<RcStream> dstStreams_;

  // one buffer on each src and dst GPU
  std::vector<void *> dstBufs_;
  std::vector<void *> srcBufs_;

  // one event per domain data to sync src and dst GPU
  std::vector<cudaEvent_t> events_;

  /* async copy data field dataIdx
   */
  void copy_data(size_t dataIdx) {

    const Dim3 srcPos = srcDomain_->halo_pos(dir_, false /*compute region*/);
    const Dim3 dstPos = dstDomain_->halo_pos(dir_, true /*halo region*/);

    const Dim3 srcSize = srcDomain_->raw_size();
    const Dim3 dstSize = dstDomain_->raw_size();

    const Dim3 extent = srcDomain_->halo_extent(dir_);
    assert(extent == dstDomain_->halo_extent(dir_ * -1));

    assert(srcDomain_->num_data() == dstDomain_->num_data());

    char *dst = dstDomain_->curr_data(dataIdx);
    char *src = srcDomain_->curr_data(dataIdx);

    size_t elemSize = srcDomain_->elem_size(dataIdx);
    assert(elemSize == dstDomain_->elem_size(dataIdx));

    // translate halo region to other
    const dim3 dimBlock(32, 4, 4);
    const dim3 dimGrid = (extent + (Dim3(dimBlock) - 1)) / Dim3(dimBlock);

    // pack on source
    CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
    pack<<<dimGrid, dimBlock, 0, srcStreams_[dataIdx]>>>(
        srcBufs_[dataIdx], srcDomain_->curr_data(dataIdx), srcSize, 0 /*pitch*/,
        srcPos, extent, elemSize);

    // copy to dst
    size_t numBytes = srcDomain_->halo_bytes(dir_, dataIdx);
    CUDA_RUNTIME(cudaMemcpyAsync(dstBufs_[dataIdx], srcBufs_[dataIdx], numBytes,
                                 cudaMemcpyDefault, srcStreams_[dataIdx]));

    // record event in src stream
    CUDA_RUNTIME(cudaEventRecord(events_[dataIdx], srcStreams_[dataIdx]));

    // cause dst stream to wait on src event
    CUDA_RUNTIME(cudaStreamWaitEvent(dstStreams_[dataIdx], events_[dataIdx],
                                     0 /*flags*/));

    // unpack on dst
    CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
    unpack<<<dimGrid, dimBlock, 0, dstStreams_[dataIdx]>>>(
        dstDomain_->curr_data(dataIdx), dstSize, 0 /*pitch*/, dstPos, extent,
        dstBufs_[dataIdx], elemSize);
  }

public:
  PackMemcpyCopier(const LocalDomain &dstDomain, const LocalDomain &srcDomain,
                   const Dim3 &dir)
      : srcDomain_(&srcDomain), dstDomain_(&dstDomain), dir_(dir) {

    assert(dstDomain.num_data() == srcDomain.num_data());

    // associate domain data array a src stream, a dst stream, and an event
    for (size_t di = 0; di < srcDomain.num_data(); ++di) {
      int srcGpu = srcDomain_->gpu();
      CUDA_RUNTIME(cudaSetDevice(srcGpu));
      srcStreams_.push_back(RcStream(srcGpu));
      cudaEvent_t event;
      CUDA_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      events_.push_back(event);

      int dstGpu = dstDomain_->gpu();
      CUDA_RUNTIME(cudaSetDevice(dstGpu));
      dstStreams_.push_back(RcStream(dstGpu));
    }
  }

  virtual void allocate() override {
    assert(0 && "unimplemented");
    for (size_t dataIdx = 0; dataIdx < srcDomain_->num_data(); ++dataIdx) {
      size_t numBytes = srcDomain_->halo_bytes(dir_, dataIdx);
      void *p = nullptr;
      CUDA_RUNTIME(cudaSetDevice(srcDomain_->gpu()));
      CUDA_RUNTIME(cudaMalloc(&p, numBytes));
      srcBufs_.push_back(p);
      p = nullptr;
      CUDA_RUNTIME(cudaSetDevice(dstDomain_->gpu()));
      CUDA_RUNTIME(cudaMalloc(&p, numBytes));
      dstBufs_.push_back(p);
    }
  }

  /* async copy
   */
  void send() override {
    // insert copies into streams
    for (size_t i = 0; i < srcDomain_->num_data(); ++i) {
      copy_data(i);
    }
  }

  // wait for send()
  virtual void wait() override {
    for (auto &s : dstStreams_) {
      CUDA_RUNTIME(cudaStreamSynchronize(s));
    }
  }
};