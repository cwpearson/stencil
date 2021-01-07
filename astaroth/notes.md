# Notes

## Get astaroth

git clone https://jpekkila@bitbucket.org/jpekkila/astaroth.git
cd astaroth && mkdir build && cd build
cmake -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON .. && make -j

## Notes


### `astaroth.h`

Seems to define some common types. we make our own smaller one

Relies on `NUM_SCALARARRAY_HANLDES`, not sure what this is

### `integration.cuh`

Relies on `NUM_VTXBUF_HANDLES`, which seems to be set by the code generators

* we assume this to be 8


### `user_kernels.h`

Astaroth generates `build/user_kernels.h` which we take as `user_kernels.h`.

It holds the kernel code from the DSL.

build/user_kernels.h seems to define the generated code
solve() may be the entry point. it does 
out_lnrho =rk3 (out_lnrho ,lnrho ,continuity (globalVertexIdx ,uu ,lnrho ,dt ),dt );
out_aa =rk3 (out_aa ,aa ,induction (uu ,aa ),dt );
out_uu =rk3 (out_uu ,uu ,momentum (globalVertexIdx ,uu ,lnrho ,ss ,aa ,dt ),dt );
out_ss =rk3 (out_ss ,ss ,entropy (ss ,uu ,lnrho ,aa ),dt );


### Other Notes

First, we will try to understand the required data structures to create them.

acHostMeshCreate() in src/astaroth.cc seems to suggest 8 handles that are nx/ny/nz * sizeof(real)
similar in acDeviceCreate in src/device.cc

We will assume real as double as the case we care about

src/core/kernels/integration.cuh defines a macro to generate
acDeviceKernel_<> which calls kernel <>

API_specification_and_use_manual says
solve() can be called with acDeviceKernel_solve()



user_kernels is included into src/core/kernels/integration.cuh

rk3 is defined in src/core/kernels/integration.cuh

defined in build/user_kernels
continuity
induction
momentum
entropy


## How this was done





