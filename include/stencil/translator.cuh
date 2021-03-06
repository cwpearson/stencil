#pragma once

#include "stencil/dim3.hpp"

#include <vector>

/* a repeating 3D translate operation

   Doing the same logical 3D translation across multiple allocations, of possibly different-sized elements

   Uses CUDA operations, and optionally accelerated with the CUDA graph API
*/
class Translator {

public:
  /* Describes a set of common transfers, out of allocations of different sizes and different elements
   */
  struct RegionParams {
    cudaPitchedPtr *dstPtrs;       // pointers to destination allocations
    Dim3 dstPos;                   // position within the allocation (element size)
    const cudaPitchedPtr *srcPtrs; // pointers to source allocations
    Dim3 srcPos;
    Dim3 extent;             // the extent of the region to be copied
    const size_t *elemSizes; // the size of the elements to copy
    int64_t n;               // the number of copies

    // nice to keep this an aggregate class for designated initializers
    RegionParams() = default;
  };

  Translator();
  virtual ~Translator();

  /* setup to do the requested copies
   */
  virtual void prepare(const std::vector<RegionParams> &params) = 0;

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
  static std::vector<Param> convert(const std::vector<RegionParams> &params);

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

/* Uses one kernel launch per transfer to write into destination memory
 */
class TranslatorKernel : public Translator {

public:
  // create a translator that will run on a device
  TranslatorKernel(int device);

  void prepare(const std::vector<RegionParams> &params) override;

private:
  void launch_all(cudaStream_t stream) override;

  // initiate a 3D transfer using a kernel
  static void kernel(const Param &param, const int device, cudaStream_t stream);

  int device_;
  std::vector<Param> params_;
};

/*! \class
    \brief Uses one Memcpy3d per transfer to copy to destination memory
*/
class TranslatorMemcpy3D : public Translator {
public:
  void prepare(const std::vector<RegionParams> &params) override;

private:
  void launch_all(cudaStream_t stream) override;

  // initiate a 3D transfer using cudaMemcpy3DAsync (not Peer because we may not have an ID for both devices)
  static void memcpy_3d_async(const Param &param, cudaStream_t stream);

  std::vector<Param> params_;
};

/*! \class
    \brief Uses one kernel per transfer group to write into destination memory
*/
class TranslatorMultiKernel : public Translator {
public:
  // create a translator that will run on a device
  TranslatorMultiKernel(int device);
  ~TranslatorMultiKernel();

  void prepare(const std::vector<RegionParams> &params) override;

private:
  void launch_all(cudaStream_t stream) override;

  // initiate a 3D transfer using a kernel
  static void kernel(const RegionParams &params, const int device, cudaStream_t stream);

  int device_;
  std::vector<RegionParams> params_;
};

/*! \class
    \brief Uses one kernel per domain to write into destination memory
*/
class TranslatorDomainKernel : public Translator {
public:
  // create a translator that will run on a device
  TranslatorDomainKernel(int device);
  ~TranslatorDomainKernel();
  TranslatorDomainKernel(const TranslatorDomainKernel &other) = delete;

  void prepare(const std::vector<RegionParams> &params) override;

  /* Each domain consists of one unique set of src/dst pointer, src/dst offsets, and element size per quantity.
  The number of elements in each copy is the same
   */

  struct DeviceParam {
    void **srcPtrs;          // nQuants pointers
    void **dstPtrs;          // nQuants pointers
    size_t **srcByteOffsets; // nQuants offset arrays
    size_t **dstByteOffsets; // nQuants offset arrays
    size_t *elemSizes;       // nQuants element sizes
    size_t nQuants;
    size_t nElems;
  };

private:
  void launch_all(cudaStream_t stream) override;

  int device_;

  std::vector<void*> toDelete_; // cuda allocations
  DeviceParam devParam_; // kernel params
};
