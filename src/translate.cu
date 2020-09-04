#include "stencil/translate.cuh"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"

#ifdef STENCIL_USE_CUDA_GRAPH
#endif

Translate::Translate() {}

void Translate::prepare(const std::vector<Params> &params) {
  // convert all Params into individual 3D copies
  for (const Params &ps : params) {
    for (int64_t i = 0; i < ps.n; ++i) {
      Param p{.dstPtr = ps.dsts[i],
              .dstPos = ps.dstPos,
              .dstSize = ps.dstSize,
              .srcPtr = ps.srcs[i],
              .srcPos = ps.srcPos,
              .srcSize = ps.srcSize,
              .extent = ps.extent,
              .elemSize = ps.elemSizes[i]};
      params_.push_back(p);
    }
  }

  // TODO: if cudagraph, record launch here
}

void Translate::async(cudaStream_t stream) {
  launch_all(stream);
  // TODO: if cuda graph, replay here
}

void Translate::launch_all(cudaStream_t stream) {
  for (const Param &p : params_) {
    memcpy_3d_async(p, stream);
  }
}

void Translate::memcpy_3d_async(const Param &param, cudaStream_t stream) {
  cudaMemcpy3DParms p = {};

  const uint64_t es = param.elemSize;

  p.srcPos = make_cudaPos(param.srcPos.x * es, param.srcPos.y, param.srcPos.z);
  p.srcPtr.pitch = param.srcSize.x * es;
  p.srcPtr.ptr = param.srcPtr;
  p.srcPtr.xsize = param.srcSize.x * es;
  p.srcPtr.ysize = param.srcSize.y;
  p.dstPos = make_cudaPos(param.dstPos.x * es, param.dstPos.y, param.dstPos.z);
  p.dstPtr.pitch = param.dstSize.x * es;
  p.dstPtr.ptr = param.dstPtr;
  p.dstPtr.xsize = param.dstSize.x * es;
  p.dstPtr.ysize = param.dstSize.y;
  p.extent = make_cudaExtent(param.extent.x * es, param.extent.y, param.extent.z);
  p.kind = cudaMemcpyDeviceToDevice;
  LOG_SPEW("srcPtr.pitch " << p.srcPtr.pitch);
  LOG_SPEW("srcPtr.ptr " << p.srcPtr.ptr);
  LOG_SPEW("srcPos  [" << p.srcPos.x << "," << p.srcPos.y << "," << p.srcPos.z << "]");
  LOG_SPEW("dstPtr.pitch " << p.dstPtr.pitch);
  LOG_SPEW("dstPtr.ptr " << p.dstPtr.ptr);
  LOG_SPEW("dstPos  [" << p.dstPos.x << "," << p.dstPos.y << "," << p.dstPos.z << "]");
  CUDA_RUNTIME(cudaMemcpy3DAsync(&p, stream));
}

void Translate::kernel(const Param &p, const int device, cudaStream_t stream) {
  const dim3 dimBlock = Dim3::make_block_dim(p.extent, 512 /*threads per block*/);
  const dim3 dimGrid = (p.extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
  CUDA_RUNTIME(cudaSetDevice(device));
  LOG_SPEW("translate dev=" << device << " grid=" << dimGrid << " block=" << dimBlock);
  translate<<<dimGrid, dimBlock, 0, stream>>>(p.dstPtr, p.dstPos, p.dstSize, p.srcPtr, p.srcPos, p.srcSize, p.extent,
                                              p.elemSize);
}