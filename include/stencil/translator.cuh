#pragma once

#include "stencil/dim3.hpp"

#include <vector>

/* a repeating 3D translate operation

   Doing the same logical 3D translation across multiple allocations, of possibly different-sized elements

   Uses CUDA operations, and optionally accelerated with the CUDA graph API
*/
class Translator {

public:
  /* parameters for the same logical transfer across multiple allocations of different sizes
   */
  struct Params {
    cudaPitchedPtr *dsts;       // pointers to destination allocations
    Dim3 dstPos;                // position within the allocation (element size)
    const cudaPitchedPtr *srcs; // pointers to source allocations
    Dim3 srcPos;
    Dim3 extent;             // the extent of the region to be copied
    const size_t *elemSizes; // the size of the elements to copy
    int64_t n;               // the number of copies

    // nice to keep this an aggregate class for designated initializers
    Params() = default;
  };

  Translator();
  virtual ~Translator();

  /* setup to do the requested copies
   */
  virtual void prepare(const std::vector<Params> &params) = 0;

  /* launch the translate async
   */
  void async(cudaStream_t stream);

protected:
  /* Common parameter format for all 3D transfers
   */
  struct Param {
    cudaPitchedPtr dstPtr; // dst allocation
    Dim3 dstPos;           // position within the allocation (element size)
    cudaPitchedPtr srcPtr; // source allocation
    Dim3 srcPos;
    Dim3 extent;     // the extent of the region to be copied
    size_t elemSize; // the size of the elements to copy

    Param(const cudaPitchedPtr &_dstPtr, const Dim3 &_dstPos, const cudaPitchedPtr &_srcPtr, const Dim3 &_srcPos,
          const Dim3 &_extent, const size_t _elemSize)
        : dstPtr(_dstPtr), dstPos(_dstPos), srcPtr(_srcPtr), srcPos(_srcPos), extent(_extent), elemSize(_elemSize) {}
  };

  // convert Params into equivalent Param
  static std::vector<Param> convert(const std::vector<Params> &params);

  std::vector<Param> params_;

#ifdef STENCIL_USE_CUDA_GRAPH
  cudaGraph_t graph_;
  cudaGraphExec_t instance_;
#endif

private:
  /* launch all the translate operations
   if cudaGraph is used, this will occur once in the setup
   otherwise, it will be used every time
*/
  virtual void launch_all(cudaStream_t stream) = 0;
};

class TranslatorDirectAccess : public Translator {

public:
  // create a translator that will run on a device
  TranslatorDirectAccess(int device);

  void prepare(const std::vector<Params> &params) override;

private:
  void launch_all(cudaStream_t stream) override;

  // initiate a 3D transfer using a kernel
  static void kernel(const Param &param, const int device, cudaStream_t stream);

  int device_;
};

class TranslatorMemcpy3D : public Translator {

public:
  void prepare(const std::vector<Params> &params) override;

private:
  void launch_all(cudaStream_t stream) override;

  // initiate a 3D transfer using cudaMemcpy3DAsync (not Peer because we may not have an ID for both devices)
  static void memcpy_3d_async(const Param &param, cudaStream_t stream);
};