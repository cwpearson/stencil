#include "stencil/translator.cuh"

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"
#include "stencil/rcstream.hpp"
#include "stencil/rt.hpp"

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <numeric>

/* return permutation vector to std::sort(vec...)
 */
template <typename T> std::vector<size_t> sort_permutation(const std::vector<T> &vec) {
  std::vector<size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), [&](size_t i, size_t j) { return vec[i] < vec[j]; });
  return p;
}

/*return copy of `orig` permuted by `p`
 */
template <typename T> std::vector<T> apply_permutation(const std::vector<T> &orig, const std::vector<size_t> &p) {
  assert(p.size() == orig.size());
  std::vector<T> ret(orig.size());
  std::transform(p.begin(), p.end(), ret.begin(), [&](size_t i) { return orig[i]; });
  return ret;
}

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

std::vector<Translator::Param> Translator::convert(const std::vector<RegionParams> &params) {
  std::vector<Translator::Param> ret;

  // convert all RegionParams into individual 3D copies
  for (const RegionParams &ps : params) {
    assert(ps.dstPtrs);
    assert(ps.srcPtrs);
    assert(ps.elemSizes);
    for (int64_t i = 0; i < ps.n; ++i) {
      Param p(ps.dstPtrs[i], ps.dstPos, ps.srcPtrs[i], ps.srcPos, ps.extent, ps.elemSizes[i]);
      ret.push_back(p);
    }
  }
  return ret;
}

void Translator::async(cudaStream_t stream) {
#ifdef STENCIL_USE_CUDA_GRAPH
  assert(instance_);
  CUDA_RUNTIME(rt::time(cudaGraphLaunch, instance_, stream));
#else
  launch_all(stream);
#endif
}

TranslatorKernel::TranslatorKernel(int device) : Translator(), device_(device) {}

void TranslatorKernel::prepare(const std::vector<RegionParams> &params) {

  LOG_SPEW("params.size()=" << params.size());
  params_ = convert(params);

#ifdef STENCIL_USE_CUDA_GRAPH
  // create a stream to record from
  LOG_DEBUG("TranslatorKernel::prepare: record on CUDA id " << device_);
  RcStream stream(device_);
  CUDA_RUNTIME(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_all(stream);
  CUDA_RUNTIME(cudaStreamEndCapture(stream, &graph_));
  CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
  CUDA_RUNTIME(cudaGetLastError());
#endif
}

void TranslatorKernel::launch_all(cudaStream_t stream) {
  for (const Param &p : params_) {
    kernel(p, device_, stream);
  }
}

void TranslatorKernel::kernel(const Param &p, const int device, cudaStream_t stream) {
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

void TranslatorMemcpy3D::prepare(const std::vector<RegionParams> &params) {

  LOG_SPEW("params.size()=" << params.size());
  params_ = convert(params);

#ifdef STENCIL_USE_CUDA_GRAPH
  // create a stream to record from
  RcStream stream;
  CUDA_RUNTIME(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_all(stream);
  CUDA_RUNTIME(cudaStreamEndCapture(stream, &graph_));
  CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
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

TranslatorMultiKernel::TranslatorMultiKernel(int device) : Translator(), device_(device) {}
TranslatorMultiKernel::~TranslatorMultiKernel() {
  CUDA_RUNTIME(cudaSetDevice(device_));
  for (RegionParams &param : params_) {
    // FIXME: free param.srcPtrs
    // CUDA_RUNTIME(cudaFree(param.srcPtrs));
    // param.srcPtrs = nullptr;
    // CUDA_RUNTIME(cudaFree(param.elemSizes));
    // param.elemSizes = nullptr;
    LOG_WARN("did not finish cleanup");
    CUDA_RUNTIME(cudaFree(param.dstPtrs));
    param.dstPtrs = nullptr;
  }
}

void TranslatorMultiKernel::prepare(const std::vector<RegionParams> &params) {

  params_ = params;
  // overwrite each srcPtrs, dstPtrs, and elemSizes with a device version to pass to kernel later
  CUDA_RUNTIME(cudaSetDevice(device_));
  for (auto &param : params_) {
    {
      cudaPitchedPtr *ptr = nullptr;
      CUDA_RUNTIME(cudaMalloc(&ptr, param.n * sizeof(param.srcPtrs[0])));
      CUDA_RUNTIME(cudaMemcpy(ptr, param.srcPtrs, param.n * sizeof(param.srcPtrs[0]), cudaMemcpyHostToDevice));
      param.srcPtrs = ptr;

      CUDA_RUNTIME(cudaMalloc(&ptr, param.n * sizeof(param.dstPtrs[0])));
      CUDA_RUNTIME(cudaMemcpy(ptr, param.dstPtrs, param.n * sizeof(param.dstPtrs[0]), cudaMemcpyHostToDevice));
      param.dstPtrs = ptr;
    }

    {
      size_t *ptr = nullptr;
      CUDA_RUNTIME(cudaMalloc(&ptr, param.n * sizeof(param.elemSizes[0])));
      CUDA_RUNTIME(cudaMemcpy(ptr, param.elemSizes, param.n * sizeof(param.elemSizes[0]), cudaMemcpyHostToDevice));
      param.elemSizes = ptr;
    }
  }

  // overwrite

#ifdef STENCIL_USE_CUDA_GRAPH
  // create a stream to record from
  LOG_DEBUG("TranslatorMultiKernel::prepare: record on CUDA id " << device_);
  RcStream stream(device_);
  CUDA_RUNTIME(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_all(stream);
  CUDA_RUNTIME(cudaStreamEndCapture(stream, &graph_));
  CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
  CUDA_RUNTIME(cudaGetLastError());
#endif
}

void TranslatorMultiKernel::launch_all(cudaStream_t stream) {
  for (const RegionParams &p : params_) {
    kernel(p, device_, stream);
  }
}

void TranslatorMultiKernel::kernel(const RegionParams &p, const int device, cudaStream_t stream) {
  const dim3 dimBlock = Dim3::make_block_dim(p.extent, 512 /*threads per block*/);
  const dim3 dimGrid = (p.extent + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
  CUDA_RUNTIME(cudaSetDevice(device));
  LOG_SPEW("multi_translate dev=" << device << " grid=" << dimGrid << " block=" << dimBlock);
  rt::launch(multi_translate, dimGrid, dimBlock, 0, stream, p.dstPtrs, p.dstPos, p.srcPtrs, p.srcPos, p.extent,
             p.elemSizes, p.n);
#ifndef STENCIL_USE_CUDA_GRAPH
  // 900: operation not permitted while stream is capturing
  CUDA_RUNTIME(rt::time(cudaGetLastError));
#endif
}

__global__ void domain_kernel(TranslatorDomainKernel::DeviceParam params) {
  for (int q = 0; q < params.nQuants; ++q) {

    const unsigned int elemSize = params.elemSizes[q];

    // quantity source pointer
    const char *__restrict__ qsp = (const char *)params.srcPtrs[q];
    char *__restrict__ qdp = (char *)params.dstPtrs[q];

    // quantity source offsets
    const size_t *__restrict__ qso = params.srcByteOffsets[q];
    const size_t *__restrict__ qdo = params.dstByteOffsets[q];

    for (int64_t x = blockDim.x * blockIdx.x + threadIdx.x; x < params.nElems; x += gridDim.x + blockDim.x) {
      const char *sp = qsp + qso[x];
      char *dp = qdp + qdo[x];
      if (elemSize == 4) {
        *reinterpret_cast<int32_t *>(dp) = *reinterpret_cast<const int32_t *>(sp);
      } else if (elemSize == 8) {
        *reinterpret_cast<int64_t *>(dp) = *reinterpret_cast<const int64_t *>(sp);
      } else {
        memcpy(dp, sp, elemSize);
      }
    }
  }
}

TranslatorDomainKernel::TranslatorDomainKernel(int device) : Translator(), device_(device), devParam_({}) {}
TranslatorDomainKernel::~TranslatorDomainKernel() {
  CUDA_RUNTIME(cudaSetDevice(device_));
  for (void *p : toDelete_) {
    CUDA_RUNTIME(cudaFree(p));
  }
  toDelete_.clear();
  devParam_ = {};
}

void TranslatorDomainKernel::prepare(const std::vector<RegionParams> &params) {

  CUDA_RUNTIME(cudaSetDevice(device_));

  // arguments to be copied to the device later.
  std::vector<void *> srcPtrs, dstPtrs;
  std::vector<std::vector<size_t>> srcByteOffsets, dstByteOffsets;
  std::vector<size_t> elemSizes;

  // various RegionParams will have the same src/dst allocation pair.
  // merge them into a single set of offsets
  nvtxRangePush("TranslatorDomainKernel::prepare: compute offsets");
  for (const RegionParams &rp : params) {
    for (int64_t i = 0; i < rp.n; ++i) {

      const cudaPitchedPtr srcPtr = rp.srcPtrs[i];
      const cudaPitchedPtr dstPtr = rp.dstPtrs[i];
      const size_t elemSize = rp.elemSizes[i];

      // see if a matching src/dst pair has already been created
      size_t pi;
      for (pi = 0; pi < srcPtrs.size(); ++pi) {
        if (srcPtrs[pi] == srcPtr.ptr && dstPtrs[pi] == dstPtr.ptr) {
          // this is the right index
          break;
        }
      }
      // not found. add a new one
      if (srcPtrs.size() == pi) {
        srcPtrs.push_back(srcPtr.ptr);
        dstPtrs.push_back(dstPtr.ptr);
        elemSizes.push_back(elemSize);
        srcByteOffsets.push_back({});
        dstByteOffsets.push_back({});
      } else {
        // check consistency
        assert(elemSize == elemSizes[pi]);
      }

      // compute the src and destination offsets for all elements
      for (unsigned int z = 0; z < rp.extent.z; ++z) {
        unsigned int zi = z + rp.srcPos.z;
        unsigned int zo = z + rp.dstPos.z;
        for (unsigned int y = 0; y < rp.extent.y; ++y) {
          unsigned int yi = y + rp.srcPos.y;
          unsigned int yo = y + rp.dstPos.y;
          for (unsigned int x = 0; x < rp.extent.x; ++x) {
            unsigned int xi = x + rp.srcPos.x;
            unsigned int xo = x + rp.dstPos.x;
            const size_t bi = zi * srcPtr.ysize * srcPtr.pitch + yi * srcPtr.pitch + xi * elemSize;
            const size_t bo = zo * dstPtr.ysize * dstPtr.pitch + yo * dstPtr.pitch + xo * elemSize;
            srcByteOffsets[pi].push_back(bi);
            dstByteOffsets[pi].push_back(bo);
          }
        }
      }
    }
  }
  nvtxRangePop(); // "TranslatorDomainKernel::prepare: compute offsets"

  assert(srcByteOffsets.size() == srcPtrs.size());
  assert(dstByteOffsets.size() == srcPtrs.size());
  assert(dstPtrs.size() == srcPtrs.size());
  assert(elemSizes.size() == srcPtrs.size());

  LOG_SPEW("TranslatorDomainKernel: " << srcByteOffsets.size() << " dst allocation");

  // all allocations should have the same number of elements moving
  for (size_t i = 0; i < srcByteOffsets.size(); ++i) {
    LOG_SPEW("TranslatorDomainKernel: moving " << srcByteOffsets[i].size() << " elements");
    assert(srcByteOffsets[i].size() == dstByteOffsets[i].size());
    assert(srcByteOffsets[0].size() == dstByteOffsets[i].size());
  }

  // sort srcByteOffsets / dstByteOffsets by dst to improve write coalescing?
  nvtxRangePush("TranslatorDomainKernel::prepare: sort offsets");
  for (size_t i = 0; i < srcByteOffsets.size(); ++i) {
    LOG_SPEW("TranslatorDomainKernel::prepare: sort " << dstByteOffsets[i].size() << " offsets");
    std::vector<size_t> perm = sort_permutation(dstByteOffsets[i]);
    srcByteOffsets[i] = apply_permutation(srcByteOffsets[i], perm);
    dstByteOffsets[i] = apply_permutation(dstByteOffsets[i], perm);

    // if (0 == mpi::world_rank()) {
    //   for (size_t j = 0; j < srcByteOffsets[i].size(); ++j) {
    //     LOG_SPEW(srcByteOffsets[i][j] << " -> " << dstByteOffsets[i][j]);
    //   }
    // }
  }
  nvtxRangePop(); // "TranslatorDomainKernel::prepare: sort offsets"

  // move all argument data to device
  devParam_ = {};
  devParam_.nQuants = srcPtrs.size();
  assert(!srcByteOffsets.empty());
  if (srcByteOffsets.empty()) {
    devParam_.nElems = 0;
  } else {
    devParam_.nElems = srcByteOffsets[0].size();
  }

  // offsets
  {
    // copy all offsets to GPU
    std::vector<void *> devSrcOffsets(devParam_.nQuants, nullptr);
    std::vector<void *> devDstOffsets(devParam_.nQuants, nullptr);
    for (size_t i = 0; i < devParam_.nQuants; ++i) {
      const size_t nBytes = devParam_.nElems * sizeof(size_t);
      LOG_SPEW("TranslatorDomainKernel: allocate offsets: " << nBytes * 2 << "B");
      CUDA_RUNTIME(cudaMalloc(&devSrcOffsets[i], nBytes));
      toDelete_.push_back(devSrcOffsets[i]);
      CUDA_RUNTIME(cudaMemcpy(devSrcOffsets[i], srcByteOffsets[i].data(), nBytes, cudaMemcpyHostToDevice));
      CUDA_RUNTIME(cudaMalloc(&devDstOffsets[i], nBytes));
      toDelete_.push_back(devDstOffsets[i]);
      CUDA_RUNTIME(cudaMemcpy(devDstOffsets[i], dstByteOffsets[i].data(), nBytes, cudaMemcpyHostToDevice));
    }

    // copy offset pointers to GPU
    {
      const size_t nBytes = devSrcOffsets.size() * sizeof(devSrcOffsets[0]);
      assert(nBytes == devParam_.nQuants * sizeof(void *));
      CUDA_RUNTIME(cudaMalloc(&devParam_.srcByteOffsets, nBytes));
      toDelete_.push_back(devParam_.srcByteOffsets);
      CUDA_RUNTIME(cudaMemcpy(devParam_.srcByteOffsets, devSrcOffsets.data(), nBytes, cudaMemcpyHostToDevice));
      CUDA_RUNTIME(cudaMalloc(&devParam_.dstByteOffsets, nBytes));
      toDelete_.push_back(devParam_.dstByteOffsets);
      CUDA_RUNTIME(cudaMemcpy(devParam_.dstByteOffsets, devDstOffsets.data(), nBytes, cudaMemcpyHostToDevice));
    }
  }

  // pointers & elemSizes
  {
    CUDA_RUNTIME(cudaMalloc(&devParam_.srcPtrs, srcPtrs.size() * sizeof(srcPtrs[0])));
    toDelete_.push_back(devParam_.srcPtrs);
    CUDA_RUNTIME(
        cudaMemcpy(devParam_.srcPtrs, srcPtrs.data(), srcPtrs.size() * sizeof(srcPtrs[0]), cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMalloc(&devParam_.dstPtrs, dstPtrs.size() * sizeof(dstPtrs[0])));
    toDelete_.push_back(devParam_.dstPtrs);
    CUDA_RUNTIME(
        cudaMemcpy(devParam_.dstPtrs, dstPtrs.data(), dstPtrs.size() * sizeof(dstPtrs[0]), cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMalloc(&devParam_.elemSizes, elemSizes.size() * sizeof(elemSizes[0])));
    toDelete_.push_back(devParam_.elemSizes);
    CUDA_RUNTIME(cudaMemcpy(devParam_.elemSizes, elemSizes.data(), elemSizes.size() * sizeof(elemSizes[0]),
                            cudaMemcpyHostToDevice));
  }

#ifdef STENCIL_USE_CUDA_GRAPH
  // create a stream to record from
  LOG_DEBUG("TranslatorDomainKernel::prepare: record kernel on CUDA id " << device_);
  RcStream stream(device_);
  CUDA_RUNTIME(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_all(stream);
  CUDA_RUNTIME(cudaStreamEndCapture(stream, &graph_));
  CUDA_RUNTIME(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
  CUDA_RUNTIME(cudaGetLastError());
#endif
}

void TranslatorDomainKernel::launch_all(cudaStream_t stream) {
  const int dimBlock = 512;
  const int dimGrid = (devParam_.nElems + dimBlock - 1) / (dimBlock);
  CUDA_RUNTIME(cudaSetDevice(device_));
  LOG_SPEW("domain_kernel dev=" << device_ << " grid=" << dimGrid << " block=" << dimBlock);
  rt::launch(domain_kernel, dimGrid, dimBlock, 0, stream, devParam_);
#ifndef STENCIL_USE_CUDA_GRAPH
  // 900: operation not permitted while stream is capturing
  CUDA_RUNTIME(rt::time(cudaGetLastError));
#endif
}