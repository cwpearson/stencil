#include "kernels.h"

#include <assert.h>
#include <cuComplex.h>

#include "errchk.h"
#include "math_utils.h"

__device__ __constant__ AcMeshInfo d_mesh_info;

static int __device__ __forceinline__ DCONST(const AcIntParam param) { return d_mesh_info.int_params[param]; }
static int3 __device__ __forceinline__ DCONST(const AcInt3Param param) { return d_mesh_info.int3_params[param]; }
static AcReal __device__ __forceinline__ DCONST(const AcRealParam param) { return d_mesh_info.real_params[param]; }
static AcReal3 __device__ __forceinline__ DCONST(const AcReal3Param param) { return d_mesh_info.real3_params[param]; }
static __device__ constexpr VertexBufferHandle DCONST(const VertexBufferHandle handle) { return handle; }
#define DEVICE_VTXBUF_IDX(i, j, k) ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))
#define DEVICE_1D_COMPDOMAIN_IDX(i, j, k) ((i) + (j)*DCONST(AC_nx) + (k)*DCONST(AC_nxy))
#define globalGridN (d_mesh_info.int3_params[AC_global_grid_n])
//#define globalMeshM // Placeholder
//#define localMeshN // Placeholder
//#define localMeshM // Placeholder
//#define localMeshN_min // Placeholder
//#define globalMeshN_min // Placeholder
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])
//#define d_multinode_offset (d_mesh_info.int3_params[AC_multinode_offset]) // Placeholder

static __device__ constexpr int IDX(const int i) { return i; }

static __device__ __forceinline__ int IDX(const int i, const int j, const int k) { return DEVICE_VTXBUF_IDX(i, j, k); }

static __device__ __forceinline__ int IDX(const int3 idx) { return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z); }

#if AC_DOUBLE_PRECISION == 1
typedef cuDoubleComplex acComplex;
#define acComplex(x, y) make_cuDoubleComplex(x, y)
#else
typedef cuFloatComplex acComplex;
#define acComplex(x, y) make_cuFloatComplex(x, y)
#endif

static __device__ inline acComplex exp(const acComplex &val) {
  return acComplex(exp(val.x) * cos(val.y), exp(val.x) * sin(val.y));
}
static __device__ inline acComplex operator*(const AcReal &a, const acComplex &b) {
  return (acComplex){a * b.x, a * b.y};
}

static __device__ inline acComplex operator*(const acComplex &b, const AcReal &a) {
  return (acComplex){a * b.x, a * b.y};
}

static __device__ inline acComplex operator*(const acComplex &a, const acComplex &b) {
  return (acComplex){a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// Kernels /////////////////////////////////////////////////////////////////////
#include "boundconds.cuh"
#include "integration.cuh"
#include "packing.cuh"
#include "reductions.cuh"

AcResult integrate_substep(const int stepNumber, // integration.cuh::acKenrelIntegrateSubset::step_number
                           cudaStream_t stream,
                           Rect3 cr, // compute region
                           VertexBufferArray vba) {

  dim3 dimBlock(32, 1, 4);
  dim3 dimGrid = ((cr.hi - cr.lo) + Dim3(dimBlock) - 1) / (Dim3(dimBlock)); // ceil

  int3 start{}, end{};
  start.x = cr.lo.x;
  start.y = cr.lo.y;
  start.z = cr.lo.z;
  end.x = cr.hi.x;
  end.y = cr.hi.y;
  end.z = cr.hi.z;

  if (stepNumber == 0)
    solve<0><<<dimGrid, dimBlock, 0, stream>>>(start, end, vba);
  else if (stepNumber == 1)
    solve<1><<<dimGrid, dimBlock, 0, stream>>>(start, end, vba);
  else
    solve<2><<<dimGrid, dimBlock, 0, stream>>>(start, end, vba);
  CUDA_RUNTIME(cudaDeviceSynchronize());

  return AC_SUCCESS;
}

AcResult acDeviceLoadScalarUniform(const int device, cudaStream_t stream, const AcRealParam param, const AcReal value) {
  cudaSetDevice(device);
  if (param < 0 || param >= NUM_REAL_PARAMS) {
    fprintf(stderr, "WARNING: invalid AcRealParam %d.\n", param);
    return AC_FAILURE;
  }

  if (!is_valid(value)) {
    fprintf(stderr, "WARNING: Passed an invalid value %g to device constant %s. Skipping.\n", (double)value,
            realparam_names[param]);
    return AC_FAILURE;
  }

  const size_t offset = (size_t)&d_mesh_info.real_params[param] - (size_t)&d_mesh_info;
  ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
  return AC_SUCCESS;
}

AcResult acDeviceLoadVectorUniform(const int device, cudaStream_t stream, const AcReal3Param param,
                                   const AcReal3 value) {
  cudaSetDevice(device);
  if (param < 0 || param >= NUM_REAL3_PARAMS) {
    fprintf(stderr, "WARNING: invalid AcReal3Param %d\n", param);
    return AC_FAILURE;
  }

  if (!is_valid(value)) {
    fprintf(stderr, "WARNING: Passed an invalid value (%g, %g, %g) to device constant %s. Skipping.\n", (double)value.x,
            (double)value.y, (double)value.z, real3param_names[param]);
    return AC_FAILURE;
  }

  const size_t offset = (size_t)&d_mesh_info.real3_params[param] - (size_t)&d_mesh_info;
  ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
  return AC_SUCCESS;
}

AcResult acDeviceLoadIntUniform(const int device, cudaStream_t stream, const AcIntParam param, const int value) {
  cudaSetDevice(device);
  if (param < 0 || param >= NUM_INT_PARAMS) {
    fprintf(stderr, "WARNING: invalid AcIntParam %d\n", param);
    return AC_FAILURE;
  }

  if (!is_valid(value)) {
    fprintf(stderr, "WARNING: Passed an invalid value %d to device constant %s. Skipping.\n", value,
            intparam_names[param]);
    return AC_FAILURE;
  }

  const size_t offset = (size_t)&d_mesh_info.int_params[param] - (size_t)&d_mesh_info;
  ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
  return AC_SUCCESS;
}

AcResult acDeviceLoadInt3Uniform(const int device, cudaStream_t stream, const AcInt3Param param, const int3 value) {
  cudaSetDevice(device);
  if (param < 0 || param >= NUM_INT3_PARAMS) {
    fprintf(stderr, "WARNING: invalid AcInt3Param %d\n", param);
    return AC_FAILURE;
  }

  if (!is_valid(value.x) || !is_valid(value.y) || !is_valid(value.z)) {
    fprintf(stderr,
            "WARNING: Passed an invalid value (%d, %d, %def) to device constant %s. "
            "Skipping.\n",
            value.x, value.y, value.z, int3param_names[param]);
    return AC_FAILURE;
  }

  const size_t offset = (size_t)&d_mesh_info.int3_params[param] - (size_t)&d_mesh_info;
  ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
  return AC_SUCCESS;
}

AcResult acDeviceLoadMeshInfo(const int device, const AcMeshInfo meshInfo) {
  cudaSetDevice(device);

//   ERRCHK_ALWAYS(meshInfo.int_params[AC_nx] == device->local_config.int_params[AC_nx]);
//   ERRCHK_ALWAYS(meshInfo.int_params[AC_ny] == device->local_config.int_params[AC_ny]);
//   ERRCHK_ALWAYS(meshInfo.int_params[AC_nz] == device->local_config.int_params[AC_nz]);
//   ERRCHK_ALWAYS(meshInfo.int_params[AC_multigpu_offset] == device->local_config.int_params[AC_multigpu_offset]);

  for (int i = 0; i < NUM_INT_PARAMS; ++i)
    acDeviceLoadIntUniform(device, STREAM_DEFAULT, (AcIntParam)i, meshInfo.int_params[i]);

  for (int i = 0; i < NUM_INT3_PARAMS; ++i)
    acDeviceLoadInt3Uniform(device, STREAM_DEFAULT, (AcInt3Param)i, meshInfo.int3_params[i]);

  for (int i = 0; i < NUM_REAL_PARAMS; ++i)
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, (AcRealParam)i, meshInfo.real_params[i]);

  for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
    acDeviceLoadVectorUniform(device, STREAM_DEFAULT, (AcReal3Param)i, meshInfo.real3_params[i]);

  return AC_SUCCESS;
}

AcResult
acDeviceLoadDefaultUniforms(const int device)
{
    cudaSetDevice(device);

    // clang-format off
    // Scalar
    #define LOAD_DEFAULT_UNIFORM(X) acDeviceLoadScalarUniform(device, STREAM_DEFAULT, X, X##_DEFAULT_VALUE);
    AC_FOR_USER_REAL_PARAM_TYPES(LOAD_DEFAULT_UNIFORM)
    #undef LOAD_DEFAULT_UNIFORM

    // Vector
    #define LOAD_DEFAULT_UNIFORM(X) acDeviceLoadVectorUniform(device, STREAM_DEFAULT, X, X##_DEFAULT_VALUE);
    AC_FOR_USER_REAL3_PARAM_TYPES(LOAD_DEFAULT_UNIFORM)
    #undef LOAD_DEFAULT_UNIFORM

    // Int
    #define LOAD_DEFAULT_UNIFORM(X) acDeviceLoadIntUniform(device, STREAM_DEFAULT, X, X##_DEFAULT_VALUE);
    AC_FOR_USER_INT_PARAM_TYPES(LOAD_DEFAULT_UNIFORM)
    #undef LOAD_DEFAULT_UNIFORM

    // Int3
    #define LOAD_DEFAULT_UNIFORM(X) acDeviceLoadInt3Uniform(device, STREAM_DEFAULT, X, X##_DEFAULT_VALUE);
    AC_FOR_USER_INT3_PARAM_TYPES(LOAD_DEFAULT_UNIFORM)
    #undef LOAD_DEFAULT_UNIFORM
    // clang-format on

    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}
