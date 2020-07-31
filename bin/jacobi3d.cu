#include <chrono>
#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include <cxxopts/cxxopts.hpp>

#include "stencil/stencil.hpp"

#define OVERLAP

const float COLD_TEMP = 0;
const float HOT_TEMP = 1;

/*! set compute region to zero
 */
/* Apply the stencil to the coordinates in `reg`
 */
__global__ void init_kernel(Accessor<float> dst, const Rect3 reg, const Rect3 cReg //<! the entire compute region
) {

  for (int64_t z = reg.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < reg.hi.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = reg.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < reg.hi.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = reg.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < reg.hi.x; x += gridDim.x * blockDim.x) {
        Dim3 o(x, y, z);
        dst[o] = (HOT_TEMP + COLD_TEMP) / 2;
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

  for (int64_t z = myReg.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < myReg.hi.z; z += gridDim.z * blockDim.z) {
    for (int64_t y = myReg.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < myReg.hi.y; y += gridDim.y * blockDim.y) {
      for (int64_t x = myReg.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < myReg.hi.x;
           x += gridDim.x * blockDim.x) {
        Dim3 o(x, y, z);

        /* a sphere 1/10 of the CR in radius and x = 1/3 of the way over is set hot
           a similar sphere of cold is at x = 2/3
        */
        if (dist(o, hotCenter) <= sphereRadius) {
          dst[o] = HOT_TEMP;
        } else if (dist(o, coldCenter) <= sphereRadius) {
          dst[o] = COLD_TEMP;
        } else {

          float px = src[o + Dim3(1, 0, 0)];
          float mx = src[o + Dim3(-1, 0, 0)];
          float py = src[o + Dim3(0, 1, 0)];
          float my = src[o + Dim3(0, -1, 0)];
          float pz = src[o + Dim3(0, 0, 1)];
          float mz = src[o + Dim3(0, 0, -1)];

          float val = 0;
          val += px;
          val += mx;
          val += py;
          val += my;
          val += pz;
          val += mz;
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
  // x
  radius.dir(1, 0, 0) = 1;
  radius.dir(-1, 0, 0) = 1;
  // y
  radius.dir(0, 1, 0) = 1;
  radius.dir(0, -1, 0) = 1;
  // z
  radius.dir(0, 0, 1) = 1;
  radius.dir(0, 0, -1) = 1;
  // radius.set_face(1);

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

    // init current values
    std::cerr << "init\n";
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      auto &d = dd.domains()[di];
      Rect3 reg = d.get_compute_region();
      const Accessor<float> src = d.get_curr_accessor<float>(dh);
      dim3 dimBlock = Dim3::make_block_dim(reg.extent(), 512);
      dim3 dimGrid = (reg.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
      d.set_device();
      init_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(src, reg, computeRegion);
    }

      // wait for stencil to complete
      for (auto &s : computeStreams) {
        CUDA_RUNTIME(cudaStreamSynchronize(s));
      }

    if (1) {
#ifdef OVERLAP
      dd.write_paraview("overlap_init");
#else
      dd.write_paraview("init");
#endif
    }

#ifdef OVERLAP
    const std::vector<Rect3> interiors = dd.get_interior();
    const std::vector<std::vector<Rect3>> exteriors = dd.get_exterior();
#endif // OVERLAP

    for (size_t iter = 0; iter < 5000; ++iter) {

      if (0 == rank)
        std::cout << iter << "\n";

#ifdef OVERLAP
      // launch operations on interior, safe to compute on before exchange
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Rect3 mr = interiors[di];
        const Accessor<float> src0 = d.get_curr_accessor<float>(dh);
        const Accessor<float> dst0 = d.get_next_accessor<float>(dh);
        nvtxRangePush("launch");
        // if (0 == rank)
        //   std::cerr << rank << ": launch on region=" << mr << " (interior)\n";
        dim3 dimBlock = Dim3::make_block_dim(mr.extent(), 256);
        dim3 dimGrid = (mr.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
        d.set_device();
        stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst0, src0, mr, computeRegion);
        CUDA_RUNTIME(cudaGetLastError());
        nvtxRangePop(); // launch
                        // CUDA_RUNTIME(cudaDeviceSynchronize());
      }
#endif

      // exchange halos: update ghost elements with current values from neighbors
      // if (0 == rank)
      //   std::cerr << rank << ": exchange\n";
      dd.exchange();

#ifdef OVERLAP
      // operate on exterior now that ghost values are right
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Accessor<float> src = d.get_curr_accessor<float>(dh);
        const Accessor<float> dst = d.get_next_accessor<float>(dh);
        for (size_t si = 0; si < exteriors[di].size(); ++si) {
          nvtxRangePush("launch");
          const Rect3 mr = exteriors[di][si];
          // if (0 == rank)
          //   std::cerr << rank << ": launch on region=" << mr << " (exterior)\n";
          dim3 dimBlock = Dim3::make_block_dim(mr.extent(), 256);
          dim3 dimGrid = (mr.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
          d.set_device();
          stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst, src, mr, computeRegion);
          CUDA_RUNTIME(cudaGetLastError());
          nvtxRangePop(); // launch
          CUDA_RUNTIME(cudaDeviceSynchronize());
        }
      }
#else // OVERLAP
      // launch operations on compute region now that ghost values are right
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        auto &d = dd.domains()[di];
        const Rect3 mr = d.get_compute_region();
        const Accessor<float> src = d.get_curr_accessor<float>(dh);
        const Accessor<float> dst = d.get_next_accessor<float>(dh);
        nvtxRangePush("launch (whole)");
        // if (0 == rank)
        // std::cerr << rank << ": launch on region=" << mr << " (whole)\n";
        d.set_device();
        dim3 dimBlock = Dim3::make_block_dim(mr.extent(), 256);
        dim3 dimGrid = (mr.extent() + Dim3(dimBlock) - 1) / Dim3(dimBlock);
        stencil_kernel<<<dimGrid, dimBlock, 0, computeStreams[di]>>>(dst, src, mr, computeRegion);
        CUDA_RUNTIME(cudaGetLastError());
        nvtxRangePop(); // launch (whole)
      }
#endif
      // wait for stencil to complete
      for (auto &s : computeStreams) {
        CUDA_RUNTIME(cudaStreamSynchronize(s));
      }

      // current = next
      dd.swap();

      if (1 && (iter % 200 == 0)) {
#ifdef OVERLAP
        dd.write_paraview("overlap_iter_" + std::to_string(iter) + "_");
#else
        dd.write_paraview("iter_" + std::to_string(iter) + "_");
#endif
      }
    }

    if (1) {
#ifdef OVERLAP
      dd.write_paraview("overlap_final");
#else
      dd.write_paraview("final");
#endif
    }

  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  return 0;
}
