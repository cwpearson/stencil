# stencil

A prototype MPI/CUDA stencil halo exchange library

## Quick Start

Install MPI and CUDA, then

```
git clone git@github.com:cwpearson/stencil.git
cd stencil
mkdir build
cd build
cmake ..
make
mpirun -n 4 src/main
```

## Documentation

`stencil` optimized communication with CUDA, NUMA, and MPI, depending on whether those are available in the environment.
Depending on availability, the following compiler defines exist:

| Capability | Available | Not Available |
|-|-|-|
| MPI | `STENCIL_USE_MPI=1` | `STENCIL_USE_MPI=0` |
| CUDA | `STENCIL_USE_CUDA=1` | `STENCIL_USE_CUDA=0` |
| libnuma | `STENCIL_USE_NUMA=1` | `STENCIL_USE_NUMA=0` |


## Tests

Install MPI and CUDA, then

```
make && make test
```

Some tests are tagged:

MPI tests only
```
test/test_all "[mpi]"
```

CUDA tests only
```
test/test_all "[cuda]"
```

## Design Goals
  * v1 (prototype)
    * xyz radii (Astaroth)
    * joint stencils over multiple data types
    * user-defined stencil kernels (Astaroth)
    * edge communication (Astaroth)
    * Use the minimum number of GPUs
  * future
    * optimize communication for multi-GPU topology (comm-scope)
    * support corners
    * pitched arrays
    * optimize communication between ranks on the same node
      * https://blogs.fau.de/wittmann/2013/02/mpi-node-local-rank-determination/
      * https://stackoverflow.com/questions/9022496/how-to-determine-mpi-rank-process-number-local-to-a-socket-node
    * support nd instead of 3d
    * support larger halos to trade off communication with memory
