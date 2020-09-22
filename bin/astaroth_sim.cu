#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include <cxxopts/cxxopts.hpp>

#include "stencil/stencil.hpp"

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
__global__ void stencil_kernel(Accessor<float> dst, const Accessor<float> src, const Rect3 reg) {

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

  cxxopts::Options options("MyProgram", "One line description of MyProgram");
  // clang-format off
  options.add_options()
  ("h,help", "Show help")
  ("remote", "Enable RemoteSender/Recver")
  ("colocated", "Enable ColocatedHaloSender/Recver")
  ("peer", "Enable PeerAccessSender")
  ("kernel", "Enable PeerCopySender")
  ("trivial", "Skip node-aware placement")
  ("x", "x dim", cxxopts::value<int>()->default_value("512"))
  ("y", "y dim", cxxopts::value<int>()->default_value("512"))
  ("z", "z dim", cxxopts::value<int>()->default_value("512"))
  ("f,file", "File name", cxxopts::value<std::string>());
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result["help"].as<bool>()) {
    std::cerr << options.help();
    exit(EXIT_SUCCESS);
  }

  MPI_Init(&argc, &argv);

  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  double kernelMillis = 50;
  size_t x = result["x"].as<int>();
  size_t y = result["y"].as<int>();
  size_t z = result["z"].as<int>();

  cudaDeviceProp prop;
  CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0));
  if (std::string("Tesla V100-SXM2-32GB") == prop.name) {
    kernelMillis = 20.1;
  } else if (std::string("Tesla P100-SXM2-16GB") == prop.name) {
    kernelMillis = 34.1;
  } else {
    if (0 == rank) {
      std::cerr << "WARN: unknown GPU " << prop.name << ", using " << kernelMillis << "ms for kernel\n";
    }
  }

  /*
  Table 5
  512^3
  512^3 on Pascal 34.1ms
  512^3 on Volta  20.1ms
  */

  Method methods = Method::None;
  if (result["remote"].as<bool>()) {
    methods |= Method::CudaMpi;
  }
  if (result["colocated"].as<bool>()) {
    methods |= Method::ColoPackMemcpyUnpack;
  }
  if (result["peer"].as<bool>()) {
    methods |= Method::CudaMemcpyPeer;
  }
  if (result["kernel"].as<bool>()) {
    methods |= Method::CudaKernel;
  }
  if (Method::None == methods) {
    methods = Method::Default;
  }

  PlacementStrategy strategy = PlacementStrategy::NodeAware;
  if (result["trivial"].as<bool>()) {
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

    auto dh0 = dd.add_data<float>("d0");
    // auto dh1 = dd.add_data<float>("d1");
    // auto dh2 = dd.add_data<float>("d2");
    // auto dh3 = dd.add_data<float>("d3");

    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

    // create a compute stream for each local domain
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
        const Accessor<float> src0 = d.get_curr_accessor<float>(dh0);
        const Accessor<float> dst0 = d.get_next_accessor<float>(dh0);
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
        const Accessor<float> src0 = d.get_curr_accessor<float>(dh0);
        const Accessor<float> dst0 = d.get_next_accessor<float>(dh0);
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
