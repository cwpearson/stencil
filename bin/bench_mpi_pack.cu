#include <chrono>
#include <sstream>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"
#include "statistics.hpp"
#include "stencil/stencil.hpp"

typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

// #define HOST

/* use vector + hvector + hvector
 */
MPI_Datatype make_vhh(const Dim3 copyExt, const Dim3 allocExt) {
  cudaExtent ce = make_cudaExtent(copyExt.x, copyExt.y, copyExt.z);
  cudaExtent ae = make_cudaExtent(allocExt.x, allocExt.y, allocExt.z);
  MPI_Datatype rowType = {};
  MPI_Datatype planeType = {};
  MPI_Datatype fullType = {};
  {
    {
      {
        // number of blocks
        int count = ce.width;
        // number of elements in each block
        int blocklength = 1;
        // number of elements between the start of each block
        const int stride = 1;
        MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &rowType);
        MPI_Type_commit(&rowType);
      }
      int count = ce.height;
      int blocklength = 1;
      // bytes between start of each block
      const int stride = ae.width;
      MPI_Type_create_hvector(count, blocklength, stride, rowType, &planeType);
      MPI_Type_commit(&planeType);
    }
    int count = ce.depth;
    int blocklength = 1;
    // bytes between start of each block
    const int stride = ae.width * ae.height;
    MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
    MPI_Type_commit(&fullType);
  }

  return fullType;
}

/* use vector + hvector
 */
MPI_Datatype make_vh(const Dim3 copyExt, const Dim3 allocExt) {
  cudaExtent ce = make_cudaExtent(copyExt.x, copyExt.y, copyExt.z);
  cudaExtent ae = make_cudaExtent(allocExt.x, allocExt.y, allocExt.z);
  MPI_Datatype planeType = {};
  MPI_Datatype fullType = {};
  {
    {
      // number of blocks
      int count = ce.height;
      // number of elements in each block
      int blocklength = ce.width;
      // elements between start of each block
      const int stride = ae.width;
      MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &planeType);
      MPI_Type_commit(&planeType);
    }
    int count = ce.depth;
    int blocklength = 1;
    // bytes between start of each block
    const int stride = ae.width * ae.height;
    MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
    MPI_Type_commit(&fullType);
  }

  return fullType;
}

struct BenchResult {
  double packTime;
};

BenchResult bench(MPI_Datatype ty, const Dim3 ext, const int nIters) {

  // allocation extent (B)
  cudaExtent allocExt = {};
  allocExt.width = 1024;
  allocExt.height = 1024;
  allocExt.depth = 1024;

  // create device allocations
#ifdef HOST
  char *src = new char[allocExt.width * allocExt.height * allocExt.depth];
#else
  cudaPitchedPtr src = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc3D(&src, allocExt));
  allocExt.width = src.pitch; // cudaMalloc3D may adjust pitch
#endif

  // copy extent (B)
  cudaExtent copyExt = {};
  copyExt.width = ext.x;
  copyExt.height = ext.y;
  copyExt.depth = ext.z;

  // create flat destination allocation
  char *dst = nullptr;
  const int dstSize = copyExt.width * copyExt.height * copyExt.depth;
#ifdef HOST
  dst = new char[dstSize];
#else
  CUDA_RUNTIME(cudaMalloc(&dst, dstSize));
#endif

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    auto start = Clock::now();
#ifdef HOST
    MPI_Pack(src, 1, ty, dst, dstSize, &position, MPI_COMM_WORLD);
#else
    MPI_Pack(src.ptr, 1, ty, dst, dstSize, &position, MPI_COMM_WORLD);
#endif
    auto stop = Clock::now();
    Duration dur = stop - start;
    stats.insert(dur.count());
  }
  nvtxRangePop();

#ifdef HOST
  delete[] src;
  delete[] dst;
#else
  CUDA_RUNTIME(cudaFree(src.ptr));
  CUDA_RUNTIME(cudaFree(dst));
#endif

  return BenchResult{.packTime = stats.trimean()};
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int nIters = 5;

  argparse::Parser p;
  p.add_option(nIters, "--iters");
  if (!p.parse(argc, argv)) {
    std::cout << p.help();
    exit(EXIT_FAILURE);
  }

  Dim3 allocExt(1024, 1024, 1024);
  BenchResult result;
  MPI_Datatype ty;

  for (Dim3 ext : {Dim3(100, 100, 100), Dim3(200, 200, 200), Dim3(256, 256, 256)}) {
    ty = make_vhh(ext, allocExt);
    result = bench(ty, ext, nIters);
    std::cout << "vhh " << ext << " " << result.packTime << " " << double(ext.flatten()) / 1024 / result.packTime
              << "KiB/s "
              << "\n";

    ty = make_vh(ext, allocExt);
    result = bench(ty, ext, nIters);
    std::cout << "vh " << ext << " " << result.packTime << " " << double(ext.flatten()) / 1024 / result.packTime
              << "KiB/s "
              << "\n";
  }

  MPI_Finalize();

  return 0;
}
