#pragma once

static __global__ void
kernel_symmetric_boundconds(const int3 start, const int3 end, AcReal* vtxbuf, const int3 bindex, const int sign)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    // If destination index is inside the computational domain, return since
    // the boundary conditions are only applied to the ghost zones
    if (i_dst >= DCONST(AC_nx_min) && i_dst < DCONST(AC_nx_max) && j_dst >= DCONST(AC_ny_min) &&
        j_dst < DCONST(AC_ny_max) && k_dst >= DCONST(AC_nz_min) && k_dst < DCONST(AC_nz_max))
        return;

    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src, j_src, k_src, boundlocx0, boundlocx1, boundlocy0, boundlocy1, boundlocz0, boundlocz1;
    int bsize = STENCIL_ORDER/(int) 2;

    //if (bindex.x != 0)
    //if (bindex.y != 0)
    //if (bindex.z != 0)

    //Location of central border point.
    boundlocx0 = bsize;
    boundlocy0 = bsize; 
    boundlocz0 = bsize;
    boundlocx1 = DCONST(AC_nx_max) - 1; 
    boundlocy1 = DCONST(AC_ny_max) - 1;
    boundlocz1 = DCONST(AC_nz_max) - 1;
    
    //Defaults
    i_src = i_dst;
    j_src = j_dst;
    k_src = k_dst;

    if (bindex.x < 0)
    {

        // Pick up the mirroring value.
        if ((i_dst < boundlocx0))
        {
            i_src = 2.0f*boundlocx0 - i_dst;

        } else if ((i_dst > boundlocx1))
        {
            i_src = 2.0f*boundlocx1 - i_dst;
        }
        
        // Pick up the mirroring value.
        if ((j_dst < boundlocy0))
        {
            j_src = 2.0f*boundlocy0 - j_dst;
        } else if ((j_dst > boundlocx1))
        {
            j_src = 2.0f*boundlocy1 - j_dst;
        }
        
        // Pick up the mirroring value.
        if ((k_dst < boundlocz0))
        {
            k_src = 2.0f*boundlocz0 - k_dst;
        } else if ((k_dst > boundlocz1))
        {
            k_src = 2.0f*boundlocz1 - k_dst;
        }

        //Edges
        if (       (i_dst < boundlocx0) && (j_dst < boundlocy0)                         )
        {
            i_src = 2.0f*boundlocx0 - i_dst;
            j_src = 2.0f*boundlocy0 - j_dst;
            //if ((k_dst == 50)) printf("i_dst %i j_dst %i k_dst %i i_src %i j_src %i k_src %i bsize %i \n", i_dst, j_dst, k_dst, i_src, j_src, k_src, bsize);
        } else if ((i_dst < boundlocx0)                         && (k_dst < boundlocz0) )
        {
            i_src = 2.0f*boundlocx0 - i_dst;
            k_src = 2.0f*boundlocz0 - k_dst;
        } else if (                        (j_dst < boundlocy0) && (k_dst < boundlocz0) )
        {
            j_src = 2.0f*boundlocy0 - j_dst;
            k_src = 2.0f*boundlocz0 - k_dst;

        } else if ((i_dst > boundlocx1) && (j_dst > boundlocx1)                         )
        {
            i_src = 2.0f*boundlocx1 - i_dst;
            j_src = 2.0f*boundlocy1 - j_dst;
        } else if ( (i_dst > boundlocx1)                        && (k_dst > boundlocz1) )
        {
            i_src = 2.0f*boundlocx1 - i_dst;
            k_src = 2.0f*boundlocz1 - k_dst;
        } else if (                        (j_dst > boundlocy1) && (k_dst > boundlocz1) )
        {
            j_src = 2.0f*boundlocy1 - j_dst;
            k_src = 2.0f*boundlocz1 - k_dst;
        } else if ( (i_dst > boundlocx1)                        && (k_dst < boundlocz0) )
        {
            i_src = 2.0f*boundlocx1 - i_dst;
            k_src = 2.0f*boundlocz0 - k_dst;
        } else if ( (i_dst > boundlocx1) && (j_dst < bsize)                             )
        {
            i_src = 2.0f*boundlocx1 - i_dst;
            j_src = 2.0f*boundlocy0 - j_dst;
        } else if ( (i_dst < boundlocx0)                        && (k_dst > boundlocz1) )
        {
            i_src = 2.0f*boundlocx0 - i_dst;
            k_src = 2.0f*boundlocz1 - k_dst;
        } else if ( (i_dst < boundlocx0) && (j_dst > boundlocy1)                        )
        {
            i_src = 2.0f*boundlocx0 - i_dst;
            j_src = 2.0f*boundlocy1 - j_dst;
        } else if (                        (j_dst > boundlocy1) && (k_dst < boundlocz0) )
        {
            j_src = 2.0f*boundlocy1 - j_dst;
            k_src = 2.0f*boundlocz0 - k_dst;
        }

    }

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vtxbuf[dst_idx]   = sign*vtxbuf[src_idx] *0.0 + 1.0; // sign = 1 symmetric, sign = -1 antisymmetric
}


static __global__ void
kernel_periodic_boundconds(const int3 start, const int3 end, AcReal* vtxbuf)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    // if (i_dst >= DCONST(AC_mx) || j_dst >= DCONST(AC_my) || k_dst >= DCONST(AC_mz))
    //    return;

    // If destination index is inside the computational domain, return since
    // the boundary conditions are only applied to the ghost zones
    if (i_dst >= DCONST(AC_nx_min) && i_dst < DCONST(AC_nx_max) && j_dst >= DCONST(AC_ny_min) &&
        j_dst < DCONST(AC_ny_max) && k_dst >= DCONST(AC_nz_min) && k_dst < DCONST(AC_nz_max))
        return;

    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src = i_dst - DCONST(AC_nx_min);
    int j_src = j_dst - DCONST(AC_ny_min);
    int k_src = k_dst - DCONST(AC_nz_min);

    // Translate (s.t. the index is always positive)
    i_src += DCONST(AC_nx);
    j_src += DCONST(AC_ny);
    k_src += DCONST(AC_nz);

    // Wrap
    i_src %= DCONST(AC_nx);
    j_src %= DCONST(AC_ny);
    k_src %= DCONST(AC_nz);

    // Map to mx, my, mz coordinates
    i_src += DCONST(AC_nx_min);
    j_src += DCONST(AC_ny_min);
    k_src += DCONST(AC_nz_min);

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vtxbuf[dst_idx]   = vtxbuf[src_idx];
}

AcResult
acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                           AcReal* vtxbuf)
{
    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult 
acKernelGeneralBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                          AcReal* vtxbuf, const VertexBufferHandle vtxbuf_handle, 
                          const AcMeshInfo config, const int3 bindex)   
{
    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    int3 bc_top = {config.int_params[AC_bc_type_top_x], config.int_params[AC_bc_type_top_y], 
                   config.int_params[AC_bc_type_top_z]};
    int3 bc_bot = {config.int_params[AC_bc_type_bot_x], config.int_params[AC_bc_type_bot_y], 
                   config.int_params[AC_bc_type_bot_z]};

//#if AC_MPI_ENABLED
//    printf( "WARNING : NON-PERIODIC BOUNDARY CONDITIONS NOT SUPPORTER BY MPI! Only working at node level.\n");
//    return AC_FAILURE;
//#endif

    if ( vtxbuf_handle != -1) // This is a dummy to make swithing boundary condition with respect to   more possible later  
    {

        if (bc_top.x == AC_BOUNDCOND_SYMMETRIC) 
        {
            kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf, bindex,  1);
            ERRCHK_CUDA_KERNEL();
        } 
        else if (bc_top.x == AC_BOUNDCOND_ANTISYMMETRIC) 
        {
            kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf, bindex, -1);
            ERRCHK_CUDA_KERNEL();
        } 
        else if (bc_top.x == AC_BOUNDCOND_PERIODIC) 
        {
            kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf);
            ERRCHK_CUDA_KERNEL();
        } 
        else 
        {
            printf("ERROR: Boundary condition not recognized!\n");
            printf("ERROR: bc_top = %i, %i, %i \n", bc_top.x, bc_top.y, bc_top.z);
            printf("ERROR: bc_bot = %i, %i, %i \n", bc_bot.x, bc_bot.y, bc_bot.z);

            return AC_FAILURE;
        }

    }

    return AC_SUCCESS;
}
