#pragma once

#include "stencil/dim3.hpp"

#include <vector>

/* a repeating 3D translate operation

   Doing the same logical 3D translation across multiple allocations, of possibly different-sized elements

   Uses cudaGraph API to accelerate multiple transfers
*/
class Translate {

public:
  /* parameters for the same logical transfer across multiple allocations of different sizes
   */
  struct Params {
    void **dsts;  // pointers to destination allocations
    Dim3 dstPos;  // position within the allocation (element size)
    Dim3 dstSize; // size of the destination allocation (elements)
    void **srcs;  // pointers to source allocations
    Dim3 srcPos;
    Dim3 srcSize;
    Dim3 extent;       // the extent of the region to be copied
    size_t *elemSizes; // the size of the elements to copy
    int64_t n;          // the number of copies

    // nice to keep this an aggregate class for designated initializers
    Params() = default;
  };

  Translate();

  /* setup to do the requested copies
   */
  void prepare(const std::vector<Params> &params);

  /* launch the translate async
   */
  void async(cudaStream_t stream);

private:
  /* launch all the translate operations
     if cudaGraph is used, this will occur once in the setup
     otherwise, it will be used every time
  */
  void launch_all(cudaStream_t stream);

  /* Common parameters format for all 3D transfers
   */
  struct Param {
    void *dstPtr; // dst allocation
    Dim3 dstPos;  // position within the allocation (element size)
    Dim3 dstSize; // size of the destination allocation (elements)
    void *srcPtr; // source allocation
    Dim3 srcPos;
    Dim3 srcSize;
    Dim3 extent;     // the extent of the region to be copied
    size_t elemSize; // the size of the elements to copy
  };

  // initiate a 3D transfer using cudaMemcpy3DAsync (not Peer because we may not see both devices)
  static void memcpy_3d_async(const Param &param, cudaStream_t stream);

  // initiate a 3D transfer using a kernel
  static void kernel(const Param &param, const int device, cudaStream_t stream);

  std::vector<Param> params_;
};