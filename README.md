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
    * joint stencils over multiple data types (Astaroth)
    * user-defined stencil kernels (Astaroth)
    * edge communication (Astaroth)
    * corner communication (Astaroth)
    * CPU stencil (HPCG)
  * v2
    * data placement in heterogeneous environments
    * overlap MPI and CUDA
    * direct GPU-GPU communication
      * https://blogs.fau.de/wittmann/2013/02/mpi-node-local-rank-determination/
      * https://stackoverflow.com/questions/9022496/how-to-determine-mpi-rank-process-number-local-to-a-socket-node
  * future
    * halo size (performance)
      * fewer, larger messages
      * less frequent barriers
    * pitched arrays (performance)
    * optimized communication (performance)
    * Stop decomposition early
