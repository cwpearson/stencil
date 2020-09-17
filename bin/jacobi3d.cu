#include <cmath>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"

#include "stencil/stencil.hpp"

#include "statistics.hpp"

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
  return __fsqrt_rn(float((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
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
  const int sphereRadius = (cReg.hi.x - cReg.lo.x) / 10;

  for (int z = myReg.lo.z + blockIdx.z * blockDim.z + threadIdx.z; z < myReg.hi.z; z += gridDim.z * blockDim.z) {
    for (int y = myReg.lo.y + blockIdx.y * blockDim.y + threadIdx.y; y < myReg.hi.y; y += gridDim.y * blockDim.y) {
      for (int x = myReg.lo.x + blockIdx.x * blockDim.x + threadIdx.x; x < myReg.hi.x; x += gridDim.x * blockDim.x) {
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

  bool useStaged = false;
  bool useColo = false;
  bool useMemcpyPeer = false;
  bool useKernel = false;

  bool trivial = false;
  bool noOverlap = false;
  bool paraview = false;

  size_t x = 512;
  size_t y = 512;
  size_t z = 512;

  std::string prefix;

  int iters = 5;
  int checkpointPeriod = -1;

  argparse::Parser parser("a cwpearson/argparse-powered CLI app");
  // clang-format off
  parser.add_flag(useStaged, "--staged")->help("Enable RemoteSender/Recver");
  parser.add_flag(useColo, "--colo")->help("Enable ColocatedHaloSender/Recver");
  parser.add_flag(useMemcpyPeer, "--peer")->help("Enable PeerAccessSender");
  parser.add_flag(useKernel, "--kernel")->help("Enable PeerCopySender");
  parser.add_flag(trivial, "--trivial")->help("Skip node-aware placement");
  parser.add_flag(noOverlap, "--no-overlap")->help("Don't overlap communication and computation");
  parser.add_option(prefix, "--prefix")->help("prefix for paraview files");
  parser.add_flag(paraview, "--paraview")->help("dump paraview files");
  parser.add_option(iters, "--iters", "-n")->help("number of iterations");
  parser.add_option(checkpointPeriod, "--period", "-q")->help("iterations between checkpoints");
  parser.add_positional(x)->required();
  parser.add_positional(y)->required();
  parser.add_positional(z)->required();
  // clang-format on

  if (!parser.parse(argc, argv)) {
    std::cerr << parser.help() << "\n";
    exit(EXIT_FAILURE);
  }

  if (parser.need_help()) {
    std::cerr << parser.help() << "\n";
    return 0;
  }

  // default checkpoint 10 times
  if (checkpointPeriod <= 0) {
    checkpointPeriod = iters / 10;
  }

  MPI_Init(&argc, &argv);

  const int size = mpi::world_size();
  const int rank = mpi::world_rank();

  if (0 == rank) {
#ifndef NDEBUG
    std::cout << "WARN: not release mode\n";
    std::cerr << "WARN: not release mode\n";
#endif
#ifdef STENCIL_EXCHANGE_STATS
    std::cout << "ERR: STENCIL_EXCHANGE_STATS defined\n";
    std::cerr << "ERR: STENCIL_EXCHANGE_STATS defined\n";
    exit(-1);
#endif
  }

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numSubdoms;
  int numNodes;
  {
    MpiTopology topo(MPI_COMM_WORLD);

    numNodes = size / topo.colocated_size();
    int ranksPerNode = topo.colocated_size();
    int subdomsPerRank = topo.colocated_size() > devCount ? 1 : devCount / topo.colocated_size();

    if (0 == rank) {
      std::cerr << numNodes << " nodes\n";
      std::cerr << ranksPerNode << " ranks / node\n";
      std::cerr << subdomsPerRank << " sd / rank\n";
    }

    numSubdoms = numNodes * ranksPerNode * subdomsPerRank;
  }

  if (0 == rank) {
    std::cerr << "assuming " << numSubdoms << " subdomains\n";
  }

  /*
  For a single node, scaling the compute domain as a cube while trying to keep gridpoints/GPU constant
    cause weird-to-understand scaling performance for GPU kernels because the aspect ratio affects the GPU kernel
    performance.
  So, we just recursively mulitply the prime factors into the smallest dimensions to keep the shape constant.

  For multiple nodes, it's generally impossible to guarantee a particular size of each subdomain due to the hierarchical
  partitioning, so we just scale the whole compute region to keep the gridpoints / subdomain roughly constant.
  */
  if (1 == numNodes) {
    std::vector<int64_t> pfs = prime_factors(numSubdoms);
    for (auto pf : pfs) {
      if (x <= y && x <= z) {
        x *= pf;
      } else if (y <= z) {
        y *= pf;
      } else {
        z *= pf;
      }
    }
  } else {
    x *= std::pow(double(numSubdoms), 1.0 / 3);
    y *= std::pow(double(numSubdoms), 1.0 / 3);
    z *= std::pow(double(numSubdoms), 1.0 / 3);
  }

  cudaDeviceProp prop;
  CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0));

  Method methods = Method::None;
  if (useStaged) {
    methods |= Method::CudaMpi;
  }
  if (useColo) {
    methods |= Method::ColoPackMemcpyUnpack;
  }
  if (useMemcpyPeer) {
    methods |= Method::CudaMemcpyPeer;
  }
  if (useKernel) {
    methods |= Method::CudaKernel;
  }
  if (Method::None == methods) {
    methods = Method::Default;
  }

  PlacementStrategy strategy = PlacementStrategy::NodeAware;
  if (trivial) {
    strategy = PlacementStrategy::Trivial;
  }

  bool overlap = true;
  if (noOverlap) {
    overlap = false;
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

  Statistics iterTime;

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

    // wait for init to complete
    for (auto &s : computeStreams) {
      CUDA_RUNTIME(cudaStreamSynchronize(s));
    }

    if (paraview) {
      dd.write_paraview(prefix + "jacobi3d_init");
    }

    const std::vector<Rect3> interiors = dd.get_interior();
    const std::vector<std::vector<Rect3>> exteriors = dd.get_exterior();

    for (int iter = 0; iter < iters; ++iter) {

      double elapsed = MPI_Wtime();

      if (overlap) {
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
        }
      }

      // exchange halos: update ghost elements with current values from neighbors
      // if (0 == rank)
      //   std::cerr << rank << ": exchange\n";
      dd.exchange();

      if (overlap) {
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
          }
        }
      } else {
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
      }

      // wait for stencil to complete before swapping pointers
      for (auto &s : computeStreams) {
        CUDA_RUNTIME(cudaStreamSynchronize(s));
      }

      // current = next
      dd.swap();

      elapsed = MPI_Wtime() - elapsed;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      iterTime.insert(elapsed);

      if (paraview && (checkpointPeriod > 0) && (iter % checkpointPeriod == 0)) {
        dd.write_paraview(prefix + "jacobi3d_" + std::to_string(iter));
      }
    }

    if (paraview) {
      dd.write_paraview(prefix + "jacobi3d_final");
    }

    if (0 == mpi::world_rank()) {
      const std::string methodStr = to_string(methods);

      std::cout << "jacobi3d," << methodStr << "," << size << "," << devCount << "," << x << "," << y << "," << z << ","
                << dd.exchange_bytes_for_method(Method::CudaMpi) << ","
                << dd.exchange_bytes_for_method(Method::ColoPackMemcpyUnpack) << ","
                << dd.exchange_bytes_for_method(Method::CudaMemcpyPeer) << ","
                << dd.exchange_bytes_for_method(Method::CudaKernel) << "," << iterTime.min() << ","
                << iterTime.trimean() << "\n";
    }
  } // send domains out of scope before MPI_Finalize

  MPI_Finalize();

  std::cerr << "cuda=" << timers::cudaRuntime.get_elapsed() << "\n";
  std::cerr << "mpi=" << timers::mpi.get_elapsed() << "\n";

  return 0;
}
