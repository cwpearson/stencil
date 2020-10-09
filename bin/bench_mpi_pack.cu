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
MPI_Datatype make_v_hv_hv(const Dim3 copyExt, const Dim3 allocExt) {
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
MPI_Datatype make_v_hv(const Dim3 copyExt, const Dim3 allocExt) {
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

/* use hindexed

  each block is a row
 */
MPI_Datatype make_hi(const Dim3 copyExt, const Dim3 allocExt) {

  MPI_Datatype fullType = {};
  // z*y rows
  const int count = copyExt.z * copyExt.y;

  // byte offset of each row
  MPI_Aint *const displacements = new MPI_Aint[count];
  for (int64_t z = 0; z < copyExt.z; ++z) {
    for (int64_t y = 0; y < copyExt.y; ++y) {
      MPI_Aint bo = z * allocExt.y * allocExt.x + y * allocExt.x;
      // std::cout << bo << "\n";
      displacements[z * copyExt.y + y] = bo;
    }
  }
  // each row is the same length
  int *const blocklengths = new int[count];
  for (int i = 0; i < count; ++i) {
    blocklengths[i] = copyExt.x;
  }

  MPI_Type_create_hindexed(count, blocklengths, displacements, MPI_BYTE, &fullType);
  MPI_Type_commit(&fullType);

  return fullType;
}

/* use hindexed_block

  each block is a row
 */
MPI_Datatype make_hib(const Dim3 copyExt, const Dim3 allocExt) {
  MPI_Datatype fullType = {};
  // z*y rows
  const int count = copyExt.z * copyExt.y;
  const int blocklength = copyExt.x;

  // byte offset of each row
  MPI_Aint *const displacements = new MPI_Aint[count];
  for (int64_t z = 0; z < copyExt.z; ++z) {
    for (int64_t y = 0; y < copyExt.y; ++y) {
      MPI_Aint bo = z * allocExt.y * allocExt.x + y * allocExt.x;
      // std::cout << bo << "\n";
      displacements[z * copyExt.y + y] = bo;
    }
  }

  MPI_Type_create_hindexed_block(count, blocklength, displacements, MPI_BYTE, &fullType);
  MPI_Type_commit(&fullType);
  return fullType;
}

double GPU_Pack(void *__restrict__ dst, const cudaPitchedPtr src,
                const Dim3 srcPos,    // logical offset into the 3D region, in elements
                const Dim3 srcExtent, // logical extent of the 3D region to pack, in elements
                const size_t elemSize)

{


  const Dim3 bd = Dim3::make_block_dim(srcExtent,512);
  const Dim3 gd = (srcExtent + bd - 1) / bd;
  // std::cerr << gd << " " << bd << "\n";


#if 0
  cudaEvent_t start, stop;
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  CUDA_RUNTIME(cudaEventRecord(start));
  pack_kernel<<<512, bd>>>(dst, src, srcPos, srcExtent, elemSize);
  CUDA_RUNTIME(cudaEventRecord(stop));
  CUDA_RUNTIME(cudaEventSynchronize(stop));

  float ms;
  CUDA_RUNTIME(cudaEventElapsedTime(&ms, start, stop));
  return double(ms) / 1000;
#else
  auto start = Clock::now();
  pack_kernel<<<gd, bd>>>(dst, src, srcPos, srcExtent, elemSize);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  auto stop = Clock::now();
  Duration dur = stop - start;
  return dur.count();
#endif
}

struct BenchResult {
  double packTime;
};

BenchResult bench_mpi_pack(MPI_Datatype ty, const Dim3 ext, const int nIters) {

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
    CUDA_RUNTIME(cudaDeviceSynchronize());
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

BenchResult bench_gpu_pack(const Dim3 ext, const int nIters) {

  // allocation extent (B)
  cudaExtent allocExt = {};
  allocExt.width = 1024;
  allocExt.height = 1024;
  allocExt.depth = 1024;

  // create device allocations
  cudaPitchedPtr src = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc3D(&src, allocExt));
  allocExt.width = src.pitch; // cudaMalloc3D may adjust pitch

  // create flat destination allocation
  char *dst = nullptr;
  const int dstSize = ext.x * ext.y * ext.z;
  CUDA_RUNTIME(cudaMalloc(&dst, dstSize));

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    double time = GPU_Pack(dst, src, Dim3(0, 0, 0), ext, 1);
    stats.insert(time);
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(src.ptr));
  CUDA_RUNTIME(cudaFree(dst));

  return BenchResult{.packTime = stats.trimean()};
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int nIters = 30;

  argparse::Parser p;
  p.add_option(nIters, "--iters");
  if (!p.parse(argc, argv)) {
    std::cout << p.help();
    exit(EXIT_FAILURE);
  }

  Dim3 allocExt(1024, 1024, 1024);
  BenchResult result;
  MPI_Datatype ty;

  std::vector<Dim3> dims = {Dim3(1, 1024, 1024), Dim3(2, 1024, 512), Dim3(4, 1024, 256),  Dim3(8, 1024, 128),
                            Dim3(16, 1024, 64),  Dim3(32, 1024, 32), Dim3(64, 1024, 16),  Dim3(128, 1024, 8),
                            Dim3(256, 1024, 4),  Dim3(512, 1024, 2), Dim3(1024, 1024, 1), Dim3(1, 1024, 1),
                            Dim3(2, 1024, 1),    Dim3(4, 1024, 1),   Dim3(8, 1024, 1),    Dim3(16, 1024, 1),
                            Dim3(32, 1024, 1),   Dim3(64, 1024, 1),  Dim3(128, 1024, 1),  Dim3(256, 1024, 1),
                            Dim3(512, 1024, 1),  Dim3(12, 512, 512), Dim3(512, 3, 512),   Dim3(512, 512, 3)};
  // dims = {Dim3(32, 1024, 32), Dim3(1024, 1024, 1)};

  // all are B/s
  std::cout << "s,x,y,z,gpu (MiB/s),hib (MiB/s),v_hv_hv (MiB/s),v_hv (MiB/s)\n";
  // std::cout << "s,x,y,z,gpu (MiB/s),hi (MiB/s),hib (MiB/s),v_hv_hv (MiB/s),v_hv (MiB/s)\n";
  for (const Dim3 &ext : dims) {

    std::string s;
    s = std::to_string(ext.x) + "/" + std::to_string(ext.y) + "/" + std::to_string(ext.z);

    std::cout << s << ",";
    std::cout << ext.x << "," << ext.y << "," << ext.z;
    std::cout << std::flush;

    nvtxRangePush(s.c_str());
    result = bench_gpu_pack(ext, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;

#if 0
    ty = make_hi(ext, allocExt);
    result = bench_mpi_pack(ty, ext, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

#if 0
    ty = make_hib(ext, allocExt);
    result = bench_mpi_pack(ty, ext, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

#if 0
    ty = make_v_hv_hv(ext, allocExt);
    result = bench_mpi_pack(ty, ext, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

#if 0
    ty = make_v_hv(ext, allocExt);
    result = bench_mpi_pack(ty, ext, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

    std::cout << "\n";
    std::cout << std::flush;
    nvtxRangePop();
  }

  MPI_Finalize();

  return 0;
}
