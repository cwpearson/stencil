#pragma once
#include "astaroth.h"

#if AC_MPI_ENABLED
#include <mpi.h>
#include <stdbool.h>

#define MPI_GPUDIRECT_DISABLED (0)
#endif // AC_MPI_ENABLED

#include "stencil/stencil.hpp"

typedef AcReal AcRealPacked;

typedef struct {
  int3 dims;
  AcRealPacked *data;

  AcRealPacked *data_pinned;
  bool pinned = false; // Set if data was received to pinned memory
} PackedData;

typedef struct {
  AcReal *in[NUM_VTXBUF_HANDLES];
  AcReal *out[NUM_VTXBUF_HANDLES];

  AcReal *profiles[NUM_SCALARARRAY_HANDLES];
} VertexBufferArray;

struct device_s {
  int id;
  AcMeshInfo local_config;

  //   Concurrency
  // TODO set this
  cudaStream_t streams[NUM_STREAMS];

  // Memory
  VertexBufferArray vba;
  AcReal *reduce_scratchpad;
  AcReal *reduce_result;
};

#ifdef __cplusplus
extern "C" {
#endif

AcResult integrate_substep(const int stepNumber, // integration.cuh::acKenrelIntegrateSubset::step_number
                           cudaStream_t stream,
                           Rect3 cr, // compute region
                           VertexBufferArray vba);

AcResult acDeviceLoadDefaultUniforms(const int device);
AcResult acDeviceLoadMeshInfo(const int device, const AcMeshInfo meshInfo);

AcResult acDeviceLoadScalarUniform(const int device, cudaStream_t stream, const AcRealParam param, const AcReal value);


#if 0
/** */
AcResult acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                    AcReal* vtxbuf);
/** */
AcResult acKernelGeneralBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                   AcReal* vtxbuf, const VertexBufferHandle vtxbuf_handle,
                                   const AcMeshInfo config, const int3 bindex);


/** */
AcResult acKernelDummy(void);

/** */
AcResult acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba);

/** */
AcResult acKernelIntegrateSubstep(const cudaStream_t stream, const int step_number,
                                  const int3 start, const int3 end, VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, PackedData packed);

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const PackedData packed,
                            const int3 vba_start, VertexBufferArray vba);

/** */
AcReal acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                          const int3 end, const AcReal* vtxbuf, AcReal* scratchpad,
                          AcReal* reduce_result);

/** */
AcReal acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                         const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                         const AcReal* vtxbuf2, AcReal* scratchpad, AcReal* reduce_result);

/** */
AcReal acKernelReduceVecScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                             const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                             const AcReal* vtxbuf2, const AcReal* vtxbuf3, AcReal* scratchpad, AcReal* reduce_result);
#endif

#ifdef __cplusplus
} // extern "C"
#endif
