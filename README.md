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