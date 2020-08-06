# stencil

[![Build Status](https://travis-ci.com/cwpearson/stencil.svg?token=oXpZxp44qzps6HC63xis&branch=master)](https://travis-ci.com/cwpearson/stencil)

A prototype MPI/CUDA stencil halo exchange library


## Quick Start (weak scaling)

Install MPI, CUDA, and CMake 3.13+, then

```
git clone git@github.com:cwpearson/stencil.git
cd stencil
mkdir build
cd build
cmake ..
make
mpirun -n 4 src/weak
```

## Documentation

`stencil` optimized communication with CUDA, NUMA, and MPI, depending on whether those are available in the environment.
Depending on availability, the following compiler defines exist:

| Capability | Available | Not Available |
|-|-|-|
| MPI | `STENCIL_USE_MPI=1` | `STENCIL_USE_MPI=0` |
| CUDA | `STENCIL_USE_CUDA=1` | `STENCIL_USE_CUDA=0` |

## Requirements
Tested on

* CUDA 10.1 / 10.2
* OpenMPI 2.1

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
## Profiling with nsys

With the default profiling settings, we sometimes see a crash on Nsight Systems 2019.3.7 on amd64.
Restrict profiling to CUDA, NVTX, and OS calls.

```
nsys profile -t cuda,nvtx,osrt mpirun -n <int> blah
```

to enable IP sample, backtrace, and scheduling data collection
```
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

Use the Nsight Systems application to view the resulting `qdrep` file.

Make sure your `nsight-sys` version at least matches the `nsys` version used to collect the profile.

### 2019.5.2

Nsight Systems 2019.5.2 allows `-t mpi`, but on amd64 it causes the importer to hang.

### 2020.1.1

```
nsys profile -t nvtx,cuda,mpi mpirun -n <int> blah
```

## Profiling with nvprof

```
mpirun -n <int> nvprof -o timeline_%p.nvvp ...
```

To mount a remote directory (where there are nvprof files to load):
```
sshfs -o IdentityFile=/path/to/id_rsa user@host:/path /mount/location
```
## Choosing a different MPI

```
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=path/to/mpi ..
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

## Bulding OpenMPI with CUDA Support

```
./configure --prefix="blah" --with-cuda=/path/to/cuda
```

## On NCSA Hal

Run scripts are in `scripts/hal`

```
srun --partition=gpu --time=4:00:00 --ntasks=2 --nodes=2 --ntasks-per-node=1 --sockets-per-node=1 --cores-per-socket=8 --threads-per-core=4 --mem-per-cpu=1200 --wait=0 --export=ALL --gres=gpu:v100:2 --pty /bin/bash
```

SpectrumMPI with CUDA-Aware MPI

```
module load spectrum-mpi
cmake .. -DUSE_CUDA_AWARE_MPI=ON
make
```

Show reservations: `scontrol show res`

Show queue: `swqueue`

## On OLCF Summit

* [jsrunvisualizer](jsrunvisualizer.olcf.ornl.gov)
* [job-step-viewer](https://jobstepviewer.olcf.ornl.gov)
* [jsrun arguments](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#launching-a-job-with-jsrun)

Run scripts are in `srcipts/summit`.

nsight-systems 2020.3.1.71 can crash with the `osrt` or `mpi` profiler turned on.
Disable with `nsys profile -t cuda,nvtx`.

```
module load cmake
module load cuda
```

To control the compute mode, use `bsub -alloc_flags gpudefault` (see https://www.olcf.ornl.gov/for-users/system-user-guides/summitdev-quickstart-guide/#gpu-specific-jobs)

To enable GPUDirect, do `jsrun --smpiargs="-gpu" ...` (see https://docs.olcf.ornl.gov/systems/summit_user_guide.html, "CUDA-Aware MPI")

## ParaView (5.8.0)

* `File` > `Open`. Open individually, not as a group
  * Select the delimiters, then hit apply on the properties pane
* Group the files into a single dataset.
  * Select the two files, then `Filters` > `Common` > `Group Datasets`.
  * `Apply`. This will create a new combined GroupDatasets1 in the pipeline browser.
* Select the dataset and `Filters` > `Alphabetical` > `Table to Points`
  * Select the x,y,z columns in the `Properties` pane.
  * Click on the vizualization window
  * `Apply`


## Design Goals
  * v1 (iWAPT)
    * [x] joint stencils over multiple data types (Astaroth)
    * [x] uneven partitioning
    * [x] edge communication (Astaroth)
    * [x] corner communication (Astaroth)
    * [x] face communication (Astaroth)
    * [x] overlap MPI and CUDA
    * [x] Same-GPU halo exchange with kernels
    * [x] Same-rank halo exchange with kernels
    * [x] Same-rank halo exchange with `cudaMemcpyPeer`
    * [x] co-located MPI rank halo exchange with with `cudaIpc...` and `cudaMemcpyPeer`
    * [x] automatic data placement in heterogeneous environments
    * [x] Control which GPUs a distributed domain should use
      * `DistributedDomain::use_gpus(const std::vector<int> &gpus)` 
    * [x] Control which exchange method should be used
  * v2
    * [x] ParaView output files `DistributedDomain::write_paraview(const std::string &prefix)`
    * [x] support uneven radius (branch=`feature/multi-radius`)
    * [x] "Accessor Object" for data
      * [x] Index according to point in compute domain
    * [x] Support overlapped computation and communication
      * interface for extracting interior/exterior of compute region for kernel invocations
  * v3
    * [ ] allow a manual partition before placement
      * constrain to single subdomain per GPU

  * future work
    * [ ] Autodetect CUDA-Aware MPI support
      * testing at build time with `ompi_info`
      * `MPI_T_cvar_read` / `MPI_T_cvar_get_info` ?
    * [ ] N-Dimensional data 
      * with [cuTensor](https://docs.nvidia.com/cuda/cutensor/index.html)?
    * [ ] selectable halo multiplier
      * fewer, larger messages and less frequent barriers
      * larger halo allocations
    * [ ] pitched arrays
      * hide in accessor
    * [ ] factor out placement solver code


* needs info
  * mesh refinement
    * we would rely on the user to specify the conditions
    * inferface for asking for refinement?
    * how to rebalance load and communication
  * [ ] Remove requirement of CUDA (HPCG)
    * would require test machine with non-homogeneous intra-node communication
  * mapping multiple subdomains to each GPU

* wontfix
  * `cudaMemcpy3D`
    * pros: good for up to 3D, supports pitched allocations
    * cons: 4D will become a bunch of 3D transfers.
    * fix: we will always pack/unpack into 1D buffer
  * [ ] Non-rectangular regions
    * probably getting to close to a general task-based runtime at that point (like Legion)


## Interesting Things

### Reference-Counted Resource Manager for CUDA Streams

`include/stencil/rcstream.hpp`

A C++-style class representing a shared CUDA stream.
The underlying stream is released when the reference count drops to zero.

### GPU Distance Matrix

`include/stencil/gpu_topology.hpp`

The Distance Between GPUs is computed by using Nvidia Management Library to determine what the common ancestor of two GPUs is.
This is combined with other NVML APIs to determine if two GPUs are directly connected by NVLink, which is considered the closest distance.

## C++ Guidelines

* Don't put state in abstract classes
  * [I.25: Prefer abstract classes as interfaces to class hierarchies](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#i25-prefer-abstract-classes-as-interfaces-to-class-hierarchies)
  * [C.121: If a base class is used as an interface, make it a pure abstract class](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#c121-if-a-base-class-is-used-as-an-interface-make-it-a-pure-abstract-class)
* use signed integer types for subscripts
  * [ES.107: Don't use unsigned for subscripts, prefer gsl::index](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es107-dont-use-unsigned-for-subscripts-prefer-gslindex)


## Notes
  * [Open MPI: CUDA-Aware OpenMPI](https://www.open-mpi.org/faq/?category=runcuda#mpi-cuda-support)
  * [Nvidia DevBlock: Benchmarking CUDA-Aware MPI](https://devblogs.nvidia.com/benchmarking-cuda-aware-mpi/)

## Acks
  * [cwpearson/argparse](github.com/cwpearson/argparse)
