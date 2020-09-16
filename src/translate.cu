#include "stencil/translate.cuh"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"
#include "stencil/rcstream.hpp"
#include "stencil/rt.hpp"

#include <cuda_runtime.h>

Translator::Translator() {
#ifdef STENCIL_USE_CUDA_GRAPH
  graph_ = 0;
  instance_ = 0;
#endif
}

Translator::~Translator() {
#ifdef STENCIL_USE_CUDA_GRAPH
  if (instance_) {
    CUDA_RUNTIME(cudaGraphExecDestroy(instance_));
  }
  if (graph_) {
    CUDA_RUNTIME(cudaGraphDestroy(graph_));
  }
#endif
}

void Translator::prepare(const std::vector<Params> &params) {

  LOG_SPEW("params.size()=" << params.size());

  // convert all Params into individual 3D copies
  for (const Params &ps : params) {
    LOG_SPEW("ps.n=" << ps.n);
    assert(ps.dsts);
    assert(ps.srcs);
    assert(ps.elemSizes);
    for (int64_t i = 0; i < ps.n; ++i) {
      Param p(ps.dsts[i], ps.dstPos, ps.srcs[i], ps.srcPos, ps.extent, ps.elemSizes[i]);
      params_.push_back(p);
    }
  }

// FIXME: this stream is valid on device 0, but the kernels in TDA::kernel are launched on device device_
// this may cause invalid resource handle for devices != 0
#ifdef STENCIL_USE_CUDA_GRAPH
  // create a stream to record from
  RcStream stream;

  CUDA_RUNTIME(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_all(stream);
  CUDA_RUNTIME(cudaStreamEndCapture(stream, &graph_));
  CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
#endif
}

void Translator::async(cudaStream_t stream) {
#ifdef STENCIL_USE_CUDA_GRAPH
  CUDA_RUNTIME(rt::time(cudaGraphLaunch, instance_, stream));
#else
  launch_all(stream);
#endif
}

TranslatorDirectAccess::TranslatorDirectAccess(int device) : Translator(), device_(device) {}

void TranslatorDirectAccess::launch_all(cudaStream_t stream) {
  for (const Param &p : params_) {
    kernel(p, device_, stream);
  }
#ifdef STENCIL_USE_CUDA_GRAPH
  CUDA_RUNTIME(cudaGetLastError());
#endif
}

void TranslatorDirectAccess::kernel(const Param &p, const int device, cudaStream_t stream) {
  const dim3 dimBlock = Dim3::make_block_dim(p.extent, 512 /*threads per block*/);
  const dim3 dimGrid = (p.extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
  CUDA_RUNTIME(cudaSetDevice(device));
  LOG_SPEW("translate dev=" << device << " grid=" << dimGrid << " block=" << dimBlock);
  rt::launch(translate, dimGrid, dimBlock, 0, stream, p.dstPtr, p.dstPos, p.srcPtr, p.srcPos, p.extent, p.elemSize);
#ifndef STENCIL_USE_CUDA_GRAPH
  // 900: operation not permitted while stream is capturing
  CUDA_RUNTIME(rt::time(cudaGetLastError));
#endif
}

void TranslatorMemcpy3D::launch_all(cudaStream_t stream) {
  for (const Param &p : params_) {
    memcpy_3d_async(p, stream);
  }
}

void TranslatorMemcpy3D::memcpy_3d_async(const Param &param, cudaStream_t stream) {
  cudaMemcpy3DParms p = {};

  const uint64_t es = param.elemSize;

  // "offset into the src/dst objs in units of unsigned char"
  p.srcPos = make_cudaPos(param.srcPos.x * es, param.srcPos.y, param.srcPos.z);
  p.dstPos = make_cudaPos(param.dstPos.x * es, param.dstPos.y, param.dstPos.z);

  // "dimension of the transferred area in elements of unsigned char"
  p.extent = make_cudaExtent(param.extent.x * es, param.extent.y, param.extent.z);

  // we mark our srcPtr as `const void*` since we will not modify data through it, but the cuda pitchedPtr is just
  // `void*`
  p.srcPtr = param.srcPtr;
  p.dstPtr = param.dstPtr;

  p.kind = cudaMemcpyDeviceToDevice;
  LOG_SPEW("srcPtr.pitch " << p.srcPtr.pitch);
  LOG_SPEW("srcPtr.ptr " << p.srcPtr.ptr);
  LOG_SPEW("srcPos  [" << p.srcPos.x << "," << p.srcPos.y << "," << p.srcPos.z << "]");
  LOG_SPEW("dstPtr.pitch " << p.dstPtr.pitch);
  LOG_SPEW("dstPtr.ptr " << p.dstPtr.ptr);
  LOG_SPEW("dstPos  [" << p.dstPos.x << "," << p.dstPos.y << "," << p.dstPos.z << "]");
  LOG_SPEW("extent [dhw] = [" << p.extent.depth << "," << p.extent.height << "," << p.extent.width << "]");
  CUDA_RUNTIME(rt::time(cudaMemcpy3DAsync, &p, stream));
}
