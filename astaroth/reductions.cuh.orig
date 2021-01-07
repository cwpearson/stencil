#pragma once

/*
Reduction steps:
 1 of 3: Compute the initial value (a, a*a or exp(a)*exp(a)) and put the result in scratchpad
 2 of 3: Compute most of the reductions into a single block of data
 3 of 3: After all results have been stored into the final block, reduce the data in the final block
*/

// Function pointer definitions
typedef AcReal (*FilterFunc)(const AcReal&);
typedef AcReal (*FilterFuncVec)(const AcReal&, const AcReal&, const AcReal&);
typedef AcReal (*FilterFuncVecScal)(const AcReal&, const AcReal&, const AcReal&, const AcReal&);
typedef AcReal (*ReduceFunc)(const AcReal&, const AcReal&);

// clang-format off
/* Comparison funcs */
static __device__ inline AcReal
dmax(const AcReal& a, const AcReal& b) { return a > b ? a : b; }

static __device__ inline AcReal
dmin(const AcReal& a, const AcReal& b) { return a < b ? a : b; }

static __device__ inline AcReal
dsum(const AcReal& a, const AcReal& b) { return a + b; }

/* Function used to determine the values used during reduction */
static __device__ inline AcReal
dvalue(const AcReal& a) { return AcReal(a); }

static __device__ inline AcReal
dsquared(const AcReal& a) { return (AcReal)(a*a); }

static __device__ inline AcReal
dexp_squared(const AcReal& a) { return exp(a)*exp(a); }

static __device__ inline AcReal
dlength_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return sqrt(a*a + b*b + c*c); }

static __device__ inline AcReal
dsquared_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return dsquared(a) + dsquared(b) + dsquared(c); }

static __device__ inline AcReal
dexp_squared_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return dexp_squared(a) + dexp_squared(b) + dexp_squared(c); }

static __device__ inline AcReal
dlength_alf(const AcReal& a, const AcReal& b, const AcReal& c, const AcReal& d) { return sqrt(a*a + b*b + c*c)/sqrt(AcReal(4.0)*M_PI*exp(d)); }

static __device__ inline AcReal
dsquared_alf(const AcReal& a, const AcReal& b, const AcReal& c, const AcReal& d) { return (dsquared(a) + dsquared(b) + dsquared(c))/(AcReal(4.0)*M_PI*exp(d)); }

// clang-format on

#include <assert.h>
template <FilterFunc filter>
static __global__ void
kernel_filter(const __restrict__ AcReal* src, const int3 start, const int3 end, AcReal* dst)
{
    const int3 src_idx = (int3){start.x + threadIdx.x + blockIdx.x * blockDim.x,
                                start.y + threadIdx.y + blockIdx.y * blockDim.y,
                                start.z + threadIdx.z + blockIdx.z * blockDim.z};

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;
    (void)nz; // Suppressed unused variable warning when not compiling with debug flags

    const int3 dst_idx = (int3){threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z};

    assert(src_idx.x < DCONST(AC_nx_max) && src_idx.y < DCONST(AC_ny_max) &&
           src_idx.z < DCONST(AC_nz_max));
    assert(dst_idx.x < nx && dst_idx.y < ny && dst_idx.z < nz);
    assert(dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny < nx * ny * nz);

    dst[dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny] = filter(src[IDX(src_idx)]);
}

template <FilterFuncVec filter>
static __global__ void
kernel_filter_vec(const __restrict__ AcReal* src0, const __restrict__ AcReal* src1,
                  const __restrict__ AcReal* src2, const int3 start, const int3 end, AcReal* dst)
{
    const int3 src_idx = (int3){start.x + threadIdx.x + blockIdx.x * blockDim.x,
                                start.y + threadIdx.y + blockIdx.y * blockDim.y,
                                start.z + threadIdx.z + blockIdx.z * blockDim.z};

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;
    (void)nz; // Suppressed unused variable warning when not compiling with debug flags

    const int3 dst_idx = (int3){threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z};

    assert(src_idx.x < DCONST(AC_nx_max) && src_idx.y < DCONST(AC_ny_max) &&
           src_idx.z < DCONST(AC_nz_max));
    assert(dst_idx.x < nx && dst_idx.y < ny && dst_idx.z < nz);
    assert(dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny < nx * ny * nz);

    dst[dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny] = filter(src0[IDX(src_idx)],
                                                                   src1[IDX(src_idx)],
                                                                   src2[IDX(src_idx)]);
}

template <FilterFuncVecScal filter>
static __global__ void
kernel_filter_vec_scal(const __restrict__ AcReal* src0, const __restrict__ AcReal* src1,
                       const __restrict__ AcReal* src2, const __restrict__ AcReal* src3,  
                       const int3 start, const int3 end, AcReal* dst)
{
    const int3 src_idx = (int3){start.x + threadIdx.x + blockIdx.x * blockDim.x,
                                start.y + threadIdx.y + blockIdx.y * blockDim.y,
                                start.z + threadIdx.z + blockIdx.z * blockDim.z};

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;
    (void)nz; // Suppressed unused variable warning when not compiling with debug flags

    const int3 dst_idx = (int3){threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z};

    assert(src_idx.x < DCONST(AC_nx_max) && src_idx.y < DCONST(AC_ny_max) &&
           src_idx.z < DCONST(AC_nz_max));
    assert(dst_idx.x < nx && dst_idx.y < ny && dst_idx.z < nz);
    assert(dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny < nx * ny * nz);

    dst[dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny] = filter(
        src0[IDX(src_idx)], src1[IDX(src_idx)], src2[IDX(src_idx)], src3[IDX(src_idx)]);
}


template <ReduceFunc reduce>
static __global__ void
kernel_reduce(AcReal* scratchpad, const int num_elems)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ AcReal smem[];
    if (idx < num_elems) {
        smem[threadIdx.x] = scratchpad[idx];
    }
    else {
        smem[threadIdx.x] = NAN;
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    assert(offset % 2 == 0);
    while (offset > 0) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] = reduce(smem[threadIdx.x], smem[threadIdx.x + offset]);
        }
        offset /= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scratchpad[idx] = smem[threadIdx.x];
    }
}

template <ReduceFunc reduce>
static __global__ void
kernel_reduce_block(const __restrict__ AcReal* scratchpad, const int num_blocks,
                    const int block_size, AcReal* result)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) {
        return;
    }

    AcReal res = scratchpad[0];
    for (int i = 1; i < num_blocks; ++i) {
        res = reduce(res, scratchpad[i * block_size]);
    }
    *result = res;
}

AcReal
acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                   const int3 end, const AcReal* vtxbuf, AcReal* scratchpad, AcReal* reduce_result)
{
    const unsigned nx        = end.x - start.x;
    const unsigned ny        = end.y - start.y;
    const unsigned nz        = end.z - start.z;
    const unsigned num_elems = nx * ny * nz;

    const dim3 tpb_filter(32, 4, 1);
    const dim3 bpg_filter((unsigned int)ceil(nx / AcReal(tpb_filter.x)),
                          (unsigned int)ceil(ny / AcReal(tpb_filter.y)),
                          (unsigned int)ceil(nz / AcReal(tpb_filter.z)));

    const int tpb_reduce = 128;
    const int bpg_reduce = num_elems / tpb_reduce;

    ERRCHK(nx >= tpb_filter.x);
    ERRCHK(ny >= tpb_filter.y);
    ERRCHK(nz >= tpb_filter.z);
    ERRCHK(tpb_reduce <= num_elems);
    ERRCHK(nx * ny * nz % 2 == 0);

    // clang-format off
    if (rtype == RTYPE_MAX) {
        kernel_filter<dvalue><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dmax><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmax><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_MIN) {
        kernel_filter<dvalue><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dmin><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmin><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS) {
        kernel_filter<dsquared><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS_EXP) {
        kernel_filter<dexp_squared><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_SUM) {
        kernel_filter<dvalue><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else {
        ERROR("Unrecognized rtype");
    }
    // clang-format on
    cudaStreamSynchronize(stream);
    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    return result;
}

AcReal
acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                  const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                  const AcReal* vtxbuf2, AcReal* scratchpad, AcReal* reduce_result)
{
    const unsigned nx        = end.x - start.x;
    const unsigned ny        = end.y - start.y;
    const unsigned nz        = end.z - start.z;
    const unsigned num_elems = nx * ny * nz;

    const dim3 tpb_filter(32, 4, 1);
    const dim3 bpg_filter((unsigned int)ceil(nx / AcReal(tpb_filter.x)),
                          (unsigned int)ceil(ny / AcReal(tpb_filter.y)),
                          (unsigned int)ceil(nz / AcReal(tpb_filter.z)));

    const int tpb_reduce = 128;
    const int bpg_reduce = num_elems / tpb_reduce;

    ERRCHK(nx >= tpb_filter.x);
    ERRCHK(ny >= tpb_filter.y);
    ERRCHK(nz >= tpb_filter.z);
    ERRCHK(tpb_reduce <= num_elems);
    ERRCHK(nx * ny * nz % 2 == 0);

    // clang-format off
    if (rtype == RTYPE_MAX) {
        kernel_filter_vec<dlength_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dmax><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmax><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_MIN) {
        kernel_filter_vec<dlength_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dmin><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmin><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS) {
        kernel_filter_vec<dsquared_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS_EXP) {
        kernel_filter_vec<dexp_squared_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_SUM) {
        kernel_filter_vec<dlength_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else {
        ERROR("Unrecognized rtype");
    }
    // clang-format on

    cudaStreamSynchronize(stream);
    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    return result;
}

AcReal
acKernelReduceVecScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                      const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                      const AcReal* vtxbuf2, const AcReal* vtxbuf3, AcReal* scratchpad, AcReal* reduce_result)
{
    const unsigned nx        = end.x - start.x;
    const unsigned ny        = end.y - start.y;
    const unsigned nz        = end.z - start.z;
    const unsigned num_elems = nx * ny * nz;

    const dim3 tpb_filter(32, 4, 1);
    const dim3 bpg_filter((unsigned int)ceil(nx / AcReal(tpb_filter.x)),
                          (unsigned int)ceil(ny / AcReal(tpb_filter.y)),
                          (unsigned int)ceil(nz / AcReal(tpb_filter.z)));

    const int tpb_reduce = 128;
    const int bpg_reduce = num_elems / tpb_reduce;

    ERRCHK(nx >= tpb_filter.x);
    ERRCHK(ny >= tpb_filter.y);
    ERRCHK(nz >= tpb_filter.z);
    ERRCHK(tpb_reduce <= num_elems);
    ERRCHK(nx * ny * nz % 2 == 0);

    //NOTE: currently this has been made to only calculate afven speeds from the diagnostics. 

    // clang-format off
    if (rtype == RTYPE_ALFVEN_MAX) {
        kernel_filter_vec_scal<dlength_alf><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3, start, end, scratchpad);
        kernel_reduce<dmax><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmax><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_ALFVEN_MIN) {
        kernel_filter_vec_scal<dlength_alf><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3, start, end, scratchpad);
        kernel_reduce<dmin><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmin><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_ALFVEN_RMS) {
        kernel_filter_vec_scal<dsquared_alf><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else {
        ERROR("Unrecognized rtype");
    }
    // clang-format on

    cudaStreamSynchronize(stream);
    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    return result;
}

