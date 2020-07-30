#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include <cxxopts/cxxopts.hpp>

#include "stencil/stencil.hpp"

/*! set compute region to zero
 */
/* Apply the stencil to the coordinates in `reg`
 */
__global__ void init_kernel(Accessor<float> dst, const Rect3 reg) {

  for (int64_t z = reg.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < reg.hi.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = reg.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < reg.hi.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = reg.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < reg.hi.x; x += gridDim.x * blockDim.x) {
        dst[Dim3(x, y, z)] = 0;
      }
    }
  }
}

__device__ int64_t dist(const Dim3 a, const Dim3 b) {
  return sqrt(float((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
}

/* Apply a 3d jacobi stencil to `reg`

   Since the library only supports periodic boundary conditions right now,
   fix part of the middle of the compute region at 1 and part at 0
 */
__global__ void stencil_kernel(Accessor<float> dst, const Accessor<float> src,
                               const Rect3 myReg, //<! the region i should modify
                               const Rect3 cReg   //<! the entire compute region
) {

  // x = 1/3, y = 1/2, z = 1/2
  const Dim3 hotCenter(cReg.lo.x + (cReg.hi.x - cReg.lo.x) / 3, (cReg.lo.y + cReg.hi.y) / 2,
                       (cReg.lo.z + cReg.hi.z) / 2);
  const Dim3 coldCenter(cReg.lo.x + (cReg.hi.x - cReg.lo.x) * 2 / 3, (cReg.lo.y + cReg.hi.y) / 2,
                        (cReg.lo.z + cReg.hi.z) / 2);
  const int64_t sphereRadius = (cReg.hi.x - cReg.lo.x) / 10;
  const float COLD_TEMP = 0;
  const float HOT_TEMP = 1;

  for (int64_t z = myReg.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < myReg.hi.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = myReg.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < myReg.hi.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = myReg.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < myReg.hi.x;
           x += gridDim.x * blockDim.x) {
        Dim3 o(x, y, z);

        /* a sphere 1/10 of the CR in radius and 1/3 of the way over is 1
           a simula
        */
        if (dist(o, hotCenter) <= sphereRadius) {
          dst[o] = HOT_TEMP;
        } else if (dist(o, coldCenter) <= sphereRadius) {
          // dst[o] = COLD_TEMP;
        } else {
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
}

int main(int argc, char **argv) {

  cxxopts::Options options("MyProgram", "One line description of MyProgram");
  // clang-format off
  options.add_options()
  ("h,help", "Show help")
  ("remote", "Enable RemoteSender/Recver")
  ("cuda-aware-mpi", "Enable CudaAwareMpiSender/Recver")
  ("colocated", "Enable ColocatedHaloSender/Recver")
  ("peer", "Enable PeerAccessSender")
  ("kernel", "Enable PeerCopySender")
  ("trivial", "Skip node-aware placement")
  ("x", "x dim", cxxopts::value<int>()->default_value("100"))
  ("y", "y dim", cxxopts::value<int>()->default_value("100"))
  ("z", "z dim", cxxopts::value<int>()->default_value("100"))
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

  size_t x = result["x"].as<int>();
  size_t y = result["y"].as<int>();
  size_t z = result["z"].as<int>();

  cudaDeviceProp prop;
  CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0));

  MethodFlags methods = MethodFlags::None;
  if (result["remote"].as<bool>()) {
    methods |= MethodFlags::CudaMpi;
  }
  if (result["cuda-aware-mpi"].as<bool>()) {
    methods |= MethodFlags::CudaAwareMpi;
  }
  if (result["colocated"].as<bool>()) {
    methods |= MethodFlags::CudaMpiColocated;
  }
  if (result["peer"].as<bool>()) {
    methods |= MethodFlags::CudaMemcpyPeer;
  }
  if (result["kernel"].as<bool>()) {
    methods |= MethodFlags::CudaKernel;
  }
  if (MethodFlags::None == methods) {
    methods = MethodFlags::All;
  }

  PlacementStrategy strategy = PlacementStrategy::NodeAware;
  if (result["trivial"].as<bool>()) {
    strategy = PlacementStrategy::Trivial;
  }

  if (0 == rank) {
    std::cout << "domain: " << x << "," << y << "," << z << "\n";
  }

  Radius radius = Radius::constant(0);
  radius.set_face(1);

  {
    DistributedDomain dd(x, y, z);

    dd.set_methods(methods);
    dd.set_radius(radius);
    dd.set_placement(strategy);

    auto dh = dd.add_data<float>("d");

    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

    Rect3 computeRegion = dd.get_compute_region();

    // create a compute stream for each local domain
    std::vector<RcStream> computeStreams(dd.domains().size());
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      computeStreams[di] = RcStream(dd.domains()[di].gpu());
    }

    std::cerr << "init\n";
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      auto &d = dd.domains()[di];
      Rect3 reg = d.get_compute_region();
      const Accessor<float> src = d.get_curr_accessor<float>(dh);
      dim3 dimBlock = Dim3::make_block_dim(reg.extent(), 512);
      dim3 dimGrid = (reg.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
      d.set_device();
      init_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(src, reg);
      CUDA_RUNTIME(cudaDeviceSynchronize());
    }

    if (0)
      dd.write_paraview("init");

    const std::vector<Rect3> interiors = dd.get_interior();
    const std::vector<std::vector<Rect3>> exteriors = dd.get_exterior();

    for (size_t iter = 0; iter < 2; ++iter) {
      // launch operations on interior
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Rect3 mr = interiors[di];
        const Accessor<float> src0 = d.get_curr_accessor<float>(dh);
        const Accessor<float> dst0 = d.get_next_accessor<float>(dh);
        nvtxRangePush("launch");
        if (0 == rank) std::cerr << rank << ": launch on region=" << mr << " (interior)\n";
        // std::cerr << src0.origin() << "=src0 origin\n";
        d.set_device();
        dim3 dimBlock = Dim3::make_block_dim(mr.extent(), 512);
        dim3 dimGrid = (mr.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
        stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst0, src0, mr, computeRegion);
        CUDA_RUNTIME(cudaGetLastError());
        nvtxRangePop(); // launch
                        // CUDA_RUNTIME(cudaDeviceSynchronize());
      }

      // exchange halo
      if (0 == rank) std::cerr << rank << ": exchange\n";
      dd.exchange();

      // operate on exterior
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Accessor<float> src0 = d.get_curr_accessor<float>(dh);
        const Accessor<float> dst0 = d.get_next_accessor<float>(dh);
        for (size_t si = 0; si < exteriors[di].size(); ++si) {
          nvtxRangePush("launch");
          const Rect3 mr = exteriors[di][si];
          if (0 == rank) std::cerr << rank << ": launch on region=" << mr << " (exterior)\n";
          // std::cerr << src0.origin() << "=src0 origin\n";
          d.set_device();
          dim3 dimBlock = Dim3::make_block_dim(mr.extent(), 512);
          dim3 dimGrid = (mr.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
          stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst0, src0, mr, computeRegion);
          CUDA_RUNTIME(cudaGetLastError());
          nvtxRangePop(); // launch
          // CUDA_RUNTIME(cudaDeviceSynchronize());
        }
      }

      // wait for stencil to complete
      for (auto &s : computeStreams) {
        CUDA_RUNTIME(cudaStreamSynchronize(s));
      }

      if (1 && (iter % 1 == 0))
        dd.write_paraview("iter_" + std::to_string(iter) + "_");

      // swap
      dd.swap();
    }

    if (0)
      dd.write_paraview("final");

  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
