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
  * v1 (prototype)
    * xyz radii (Astaroth)
    * joint stencils over multiple data types
    * user-defined stencil kernels (Astaroth)
    * edge communication (Astaroth)
  * future
    * optimize communication for multi-GPU topology (comm-scope)
    * support corners
    * pitched arrays
    * optimize communication between ranks on the same node
      * https://blogs.fau.de/wittmann/2013/02/mpi-node-local-rank-determination/
      * https://stackoverflow.com/questions/9022496/how-to-determine-mpi-rank-process-number-local-to-a-socket-node
    * support nd instead of 3d
    * support larger halos to trade off communication with memory
