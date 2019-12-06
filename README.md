# stencil

A prototype MPI/CUDA stencil halo exchange library

## Quick Start

```
git clone git@github.com:cwpearson/stencil.git
cd stencil
mkdir build
cd build
cmake ..
make
mpirun -n 4 src/main
```

## Design Goals
  * v1 (mvp)
    * xyz radii
    * joint stencils over multiple data types
    * user-defined stencil kernels
  * future
    * optimize communication between ranks on the same node
      * https://blogs.fau.de/wittmann/2013/02/mpi-node-local-rank-determination/
    * optimize communication for multi-GPU topology (comm-scope)
    * support diagonal radii (astaroth)
    * support nd instead of 3d
    * support larger halos to trade off communication with memory