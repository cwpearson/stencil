#pragma once

static_assert(NUM_VTXBUF_HANDLES > 0, "ERROR: At least one uniform ScalarField must be declared.");

// Device info
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#define MAX_THREADS_PER_BLOCK (1024)
#define WARP_SIZE (32)

#define make_int3(a, b, c)                                                                         \
    (int3) { (int)a, (int)b, (int)c }

template <int step_number>
static __device__ __forceinline__ AcReal
rk3_integrate(const AcReal state_previous, const AcReal state_current, const AcReal rate_of_change,
              const AcReal dt)
{
    // Williamson (1980)
    const AcReal alpha[] = {0, AcReal(.0), AcReal(-5. / 9.), AcReal(-153. / 128.)};
    const AcReal beta[]  = {0, AcReal(1. / 3.), AcReal(15. / 16.), AcReal(8. / 15.)};

    // Note the indexing: +1 to avoid an unnecessary warning about "out-of-bounds"
    // access (when accessing beta[step_number-1] even when step_number >= 1)
    switch (step_number) {
    case 0:
        return state_current + beta[step_number + 1] * rate_of_change * dt;
    case 1: // Fallthrough
    case 2:
        return state_current +
               beta[step_number + 1] * (alpha[step_number + 1] * (AcReal(1.) / beta[step_number]) *
                                            (state_current - state_previous) +
                                        rate_of_change * dt);
    default:
        return NAN;
    }
}

template <int step_number>
static __device__ __forceinline__ AcReal3
rk3_integrate(const AcReal3 state_previous, const AcReal3 state_current,
              const AcReal3 rate_of_change, const AcReal dt)
{
    return (AcReal3){
        rk3_integrate<step_number>(state_previous.x, state_current.x, rate_of_change.x, dt),
        rk3_integrate<step_number>(state_previous.y, state_current.y, rate_of_change.y, dt),
        rk3_integrate<step_number>(state_previous.z, state_current.z, rate_of_change.z, dt),
    };
}

#define rk3(state_previous, state_current, rate_of_change, dt)                                     \
    rk3_integrate<step_number>(state_previous, value(state_current), rate_of_change, dt)

static __device__ void
write(AcReal* __restrict__ out[], const int handle, const int idx, const AcReal value)
{
    out[handle][idx] = value;
}

static __device__ __forceinline__ void
write(AcReal* __restrict__ out[], const int3 vec, const int idx, const AcReal3 value)
{
    write(out, vec.x, idx, value.x);
    write(out, vec.y, idx, value.y);
    write(out, vec.z, idx, value.z);
}

static __device__ __forceinline__ AcReal
read_out(const int idx, AcReal* __restrict__ field[], const int handle)
{
    return field[handle][idx];
}

static __device__ __forceinline__ AcReal3
read_out(const int idx, AcReal* __restrict__ field[], const int3 handle)
{
    return (AcReal3){read_out(idx, field, handle.x), read_out(idx, field, handle.y),
                     read_out(idx, field, handle.z)};
}

#define WRITE_OUT(handle, value) (write(buffer.out, handle, idx, value))
#define READ(handle) (read_data(vertexIdx, globalVertexIdx, buffer.in, handle))
#define READ_OUT(handle) (read_out(idx, buffer.out, handle))

#define GEN_PREPROCESSED_PARAM_BOILERPLATE const int3 &vertexIdx, const int3 &globalVertexIdx
#define GEN_KERNEL_PARAM_BOILERPLATE const int3 start, const int3 end, VertexBufferArray buffer

#define GEN_KERNEL_BUILTIN_VARIABLES_BOILERPLATE()                                                 \
    const int3 vertexIdx       = (int3){threadIdx.x + blockIdx.x * blockDim.x + start.x,           \
                                  threadIdx.y + blockIdx.y * blockDim.y + start.y,           \
                                  threadIdx.z + blockIdx.z * blockDim.z + start.z};          \
    const int3 globalVertexIdx = (int3){d_multigpu_offset.x + vertexIdx.x,                         \
                                        d_multigpu_offset.y + vertexIdx.y,                         \
                                        d_multigpu_offset.z + vertexIdx.z};                        \
    (void)globalVertexIdx;                                                                         \
    if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)                      \
        return;                                                                                    \
                                                                                                   \
    assert(vertexIdx.x < DCONST(AC_nx_max) && vertexIdx.y < DCONST(AC_ny_max) &&                   \
           vertexIdx.z < DCONST(AC_nz_max));                                                       \
                                                                                                   \
    assert(vertexIdx.x >= DCONST(AC_nx_min) && vertexIdx.y >= DCONST(AC_ny_min) &&                 \
           vertexIdx.z >= DCONST(AC_nz_min));                                                      \
                                                                                                   \
    const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);

#define GEN_DEVICE_FUNC_HOOK(identifier)                                                           \
    template <int step_number>                                                                     \
    AcResult acDeviceKernel_##identifier(const cudaStream_t stream, const int3 start,              \
                                         const int3 end, VertexBufferArray vba)                    \
    {                                                                                              \
                                                                                                   \
        const dim3 tpb(32, 1, 4);                                                                  \
                                                                                                   \
        const int3 n = end - start;                                                                \
        const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)),                                    \
                       (unsigned int)ceil(n.y / AcReal(tpb.y)),                                    \
                       (unsigned int)ceil(n.z / AcReal(tpb.z)));                                   \
                                                                                                   \
        identifier<step_number><<<bpg, tpb, 0, stream>>>(start, end, vba);                         \
        ERRCHK_CUDA_KERNEL();                                                                      \
                                                                                                   \
        return AC_SUCCESS;                                                                         \
    }

#include "user_kernels.h"

static dim3 rk3_tpb(32, 1, 4);

AcResult
acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba)
{
    printf("Autotuning... ");
    // RK3
    dim3 best_dims(0, 0, 0);
    float best_time          = INFINITY;
    const int num_iterations = 10;

    for (int z = 1; z <= MAX_THREADS_PER_BLOCK; ++z) {
        for (int y = 1; y <= MAX_THREADS_PER_BLOCK; ++y) {
            for (int x = WARP_SIZE; x <= MAX_THREADS_PER_BLOCK; x += WARP_SIZE) {

                if (x > end.x - start.x || y > end.y - start.y || z > end.z - start.z)
                    break;
                if (x * y * z > MAX_THREADS_PER_BLOCK)
                    break;

                if (x * y * z * REGISTERS_PER_THREAD > MAX_REGISTERS_PER_BLOCK)
                    break;

                if (((x * y * z) % WARP_SIZE) != 0)
                    continue;

                const dim3 tpb(x, y, z);
                const int3 n = end - start;
                const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                               (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                               (unsigned int)ceil(n.z / AcReal(tpb.z)));

                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) // resets the error if any
                    continue;

                // printf("(%d, %d, %d)\n", x, y, z);

                cudaEvent_t tstart, tstop;
                cudaEventCreate(&tstart);
                cudaEventCreate(&tstop);

                // #ifdef AC_dt
                // acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, FLT_EPSILON); // TODO
                // note, temporarily disabled
                /*#else
                                ERROR("FATAL ERROR: acDeviceAutoOptimize() or
                acDeviceIntegrateSubstep() was " "called, but AC_dt was not defined. Either define
                it or call the generated " "device function acDeviceKernel_<kernel name> which does
                not require the " "timestep to be defined.\n"); #endif*/

                cudaEventRecord(tstart); // ---------------------------------------- Timing start
                for (int i = 0; i < num_iterations; ++i)
                    solve<2><<<bpg, tpb>>>(start, end, vba);

                cudaEventRecord(tstop); // ----------------------------------------- Timing end
                cudaEventSynchronize(tstop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, tstart, tstop);

                ERRCHK_CUDA_KERNEL_ALWAYS();
                if (milliseconds < best_time) {
                    best_time = milliseconds;
                    best_dims = tpb;
                }
            }
        }
    }
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);
#if AC_VERBOSE
    printf("Auto-optimization done. The best threadblock dimensions for rkStep: (%d, %d, %d) %f "
           "ms\n",
           best_dims.x, best_dims.y, best_dims.z, double(best_time) / num_iterations);
#endif
    /*
    FILE* fp = fopen("../config/rk3_tbdims.cuh", "w");
    ERRCHK(fp);
    fprintf(fp, "%d, %d, %d\n", best_dims.x, best_dims.y, best_dims.z);
    fclose(fp);
    */

    rk3_tpb = best_dims;

    // Failed to find valid thread block dimensions
    ERRCHK_ALWAYS(rk3_tpb.x * rk3_tpb.y * rk3_tpb.z > 0);
    return AC_SUCCESS;
}

AcResult
acKernelIntegrateSubstep(const cudaStream_t stream, const int step_number, const int3 start,
                         const int3 end, VertexBufferArray vba)
{
    ERRCHK_ALWAYS(step_number >= 0);
    ERRCHK_ALWAYS(step_number < 3);
    const dim3 tpb = rk3_tpb;

    const int3 n = end - start;
    const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                   (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                   (unsigned int)ceil(n.z / AcReal(tpb.z)));

    //#ifdef AC_dt
    // acDeviceLoadScalarUniform(device, stream, AC_dt, dt);
    /*#else
        (void)dt;
        ERROR("FATAL ERROR: acDeviceAutoOptimize() or acDeviceIntegrateSubstep() was "
              "called, but AC_dt was not defined. Either define it or call the generated "
              "device function acDeviceKernel_<kernel name> which does not require the "
              "timestep to be defined.\n");
    #endif*/
    if (step_number == 0)
        solve<0><<<bpg, tpb, 0, stream>>>(start, end, vba);
    else if (step_number == 1)
        solve<1><<<bpg, tpb, 0, stream>>>(start, end, vba);
    else
        solve<2><<<bpg, tpb, 0, stream>>>(start, end, vba);

    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    acComplex a = exp(AcReal(1) * acComplex(1, 1) * AcReal(1));
    a* a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}
