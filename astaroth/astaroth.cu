/* 
Try to do some rough approximation of astaroth using the stencil library.
*/

#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"
#include "stencil/stencil.hpp"


#include "kernels.h"

/*! set compute region to dst[x,y,z] = sin(x+y+z + origin.x + origin.y + origin.z)
 */
template <typename T>
__global__ void init_kernel(Accessor<T> dst,    //<! [out] pointer to beginning of allocation
                            const Rect3 cr,     //<! [in] compute region
                            const double period //<! [in] sine wave period
) {
  for (int64_t z = cr.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < cr.hi.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = cr.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < cr.hi.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = cr.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < cr.hi.x; x += gridDim.x * blockDim.x) {
        dst[Dim3(x, y, z)] = sin(2 * 3.14159 / period * x + 2 * 3.14159 / period * y + 2 * 3.14159 / period * z);
      }
    }
  }
}

/* Apply the stencil to the coordinates in `reg`
 */
__global__ void stencil_kernel(Accessor<AcReal> dst, const Accessor<AcReal> src, const Rect3 reg) {

  for (int64_t z = reg.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < reg.hi.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = reg.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < reg.hi.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = reg.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < reg.hi.x; x += gridDim.x * blockDim.x) {
        Dim3 o(x, y, z);
        float val = 0;
        val += src[o + Dim3(-1, 0, 0)];
        val += src[o + Dim3(0, -1, 0)];
        val += src[o + Dim3(0, 0, -1)];
        val += src[o + Dim3(1, 0, 0)];
        val += src[o + Dim3(0, 1, 0)];
        val += src[o + Dim3(0, 0, 1)];
        val /= 6;
        dst[o] = val;
      }
    }
  }
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  argparse::Parser p("an astaroth-like performance simulator");

  int x = 256;
  int y = 256;
  int z = 256;
  bool trivialPlacement = false;

  p.add_flag(trivialPlacement, "--trivial")->help("use trivial placement");
  p.add_positional(x)->required();
  p.add_positional(y)->required();
  p.add_positional(z)->required();

  // If there was an error during parsing, report it.
  if (!p.parse(argc, argv)) {
    if (0 == rank) {
      std::cerr << p.help();
    }
    exit(EXIT_FAILURE);
  }

  if (p.need_help()) {
    if (0 == rank) {
      std::cerr << p.help();
    }
    exit(EXIT_SUCCESS);
  }

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numSubdoms;
  {
    MpiTopology topo(MPI_COMM_WORLD);
    numSubdoms = size / topo.colocated_size() * devCount;
  }

  if (0 == rank) {
    std::cout << "assuming " << numSubdoms << " subdomains\n";
  }

  Method methods = Method::None;
  methods |= Method::CudaMpi;
  methods |= Method::ColoPackMemcpyUnpack;
  methods |= Method::CudaMemcpyPeer;
  methods |= Method::CudaKernel;
  if (Method::None == methods) {
    methods = Method::Default;
  }

  PlacementStrategy strategy = PlacementStrategy::NodeAware;
  if (trivialPlacement) {
    strategy = PlacementStrategy::Trivial;
  }

  if (0 == rank) {
    std::cout << "domain: " << x << "," << y << "," << z << "\n";
  }

  {
    size_t radius = 3;

    DistributedDomain dd(x, y, z);

    dd.set_methods(methods);
    dd.set_radius(radius);
    dd.set_placement(strategy);

    auto dh0 = dd.add_data<AcReal>("d0");
    auto dh1 = dd.add_data<AcReal>("d1");
    auto dh2 = dd.add_data<AcReal>("d2");
    auto dh3 = dd.add_data<AcReal>("d3");
    auto dh4 = dd.add_data<AcReal>("d4");
    auto dh5 = dd.add_data<AcReal>("d5");
    auto dh6 = dd.add_data<AcReal>("d6");
    auto dh7 = dd.add_data<AcReal>("d7");

    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

    // create a stream for the integration kernels to run in
    std::vector<RcStream> computeStreams(dd.domains().size());
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      computeStreams[di] = RcStream(dd.domains()[di].gpu());
    }

    std::cerr << "init\n";
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      auto &d = dd.domains()[di];
      d.set_device();
      dim3 dimBlock = Dim3::make_block_dim(d.raw_size(), 512);
      dim3 dimGrid = ((d.raw_size()) + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      init_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(d.get_curr_accessor(dh0), d.get_compute_region(), 10);
      CUDA_RUNTIME(cudaDeviceSynchronize());
    }

    if (0)
      dd.write_paraview("init");

    const std::vector<Rect3> interiors = dd.get_interior();
    const std::vector<std::vector<Rect3>> exteriors = dd.get_exterior();

    for (size_t iter = 0; iter < 5; ++iter) {

      // launch operations on interior
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Accessor<AcReal> src0 = d.get_curr_accessor<AcReal>(dh0);
        const Accessor<AcReal> dst0 = d.get_next_accessor<AcReal>(dh0);
        nvtxRangePush("launch");
        const Rect3 cr = interiors[di];
        std::cerr << rank << ": launch on region=" << cr << " (interior)\n";
        // std::cerr << src0.origin() << "=src0 origin\n";
        d.set_device();
        dim3 dimBlock = Dim3::make_block_dim(cr.hi - cr.lo, 512);
        dim3 dimGrid = ((cr.hi - cr.lo) + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
        stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst0, src0, cr);
        CUDA_RUNTIME(cudaGetLastError());
        nvtxRangePop(); // launch
                        // CUDA_RUNTIME(cudaDeviceSynchronize());
      }

      // exchange halo
      std::cerr << rank << ": exchange\n";
      dd.exchange();

      // operate on exterior
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Accessor<AcReal> src0 = d.get_curr_accessor<AcReal>(dh0);
        const Accessor<AcReal> dst0 = d.get_next_accessor<AcReal>(dh0);
        for (size_t si = 0; si < exteriors[di].size(); ++si) {
          nvtxRangePush("launch");
          const Rect3 cr = exteriors[di][si];
          std::cerr << rank << ": launch on region=" << cr << " (exterior)\n";
          // std::cerr << src0.origin() << "=src0 origin\n";
          d.set_device();
          dim3 dimBlock = Dim3::make_block_dim(cr.hi - cr.lo, 512);
          dim3 dimGrid = ((cr.hi - cr.lo) + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
          stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst0, src0, cr);
          CUDA_RUNTIME(cudaGetLastError());
          nvtxRangePop(); // launch
          // CUDA_RUNTIME(cudaDeviceSynchronize());
        }
      }

      // wait for stencil to complete
      for (auto &s : computeStreams) {
        CUDA_RUNTIME(cudaStreamSynchronize(s));
      }

      // swap
      dd.swap();
    }

    if (0)
      dd.write_paraview("final");

  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
