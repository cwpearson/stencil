# stencil

A prototype MPI/CUDA stencil halo exchange library

## Quick Start

Install MPI, CUDA, and CMake 3.13+, then

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

To run specific tests
```
test/test_cpu "<case name>" -c "<section name>"
```

## Running the Astaroth-sim

```
mpirun -n 4 src/astaroth-sim
```

## Profiling MPI

```
mpirun -n <int> nvprof -o timeline_%p.nvvp ...
```

To mount a remote directory (where there are nvprof files to load):
```
sshfs -o IdentityFile=/path/to/id_rsa user@host:/path /mount/location
```

## MCA Parameters

Setting an MCA param
```
mpirun --mca mpi_show_handle_leaks 1 -np 4 a.out
```

Checking for CUDA-Aware MPI support:
```
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```


## Design Goals
  * v1 (prototype)
    * joint stencils over multiple data types (Astaroth)
    * user-defined stencil kernels (Astaroth)
    * edge communication (Astaroth)
    * corner communication (Astaroth)
    * face communication (Astaroth)
    * overlap MPI and CUDA
  * v2
    * Remove requirement of CUDA
    * data placement in heterogeneous environments
    * direct GPU-GPU communication
      * https://blogs.fau.de/wittmann/2013/02/mpi-node-local-rank-determination/
      * https://stackoverflow.com/questions/9022496/how-to-determine-mpi-rank-process-number-local-to-a-socket-node   
    * N-Dimensional data with [cuTensor](https://docs.nvidia.com/cuda/cutensor/index.html)
  * future
    * CPU stencil (HPCG)
    * halo size (performance)
      * fewer, larger messages
      * less frequent barriers
    * pitched arrays (performance)
    * optimized communication (performance)
    * Stop decomposition early


## Interesting Things

### Reference-Counted Resource Manager for CUDA Streams

`include/stencil/rcstream.hpp`

A C++-style class representing a shared CUDA stream.
The underlying stream is released when the reference count drops to zero.

### GPU Distance Matrix

`include/stencil/gpu_topo.hpp`

The Distance Between GPUs is computed by using Nvidia Management Library to determine what the common ancestor of two GPUs is.
This is combined with other NVML APIs to determine if two GPUs are directly connected by NVLink, which is considered the closest distance.

## Notes
  * [CUDA-Aware OpenMPI](https://www.open-mpi.org/faq/?category=runcuda#mpi-cuda-support)
