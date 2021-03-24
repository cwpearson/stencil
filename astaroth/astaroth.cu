/*
Try to do some rough approximation of astaroth using the stencil library.
*/

#include <cmath>
#include <thread>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"
#include "stencil/stencil.hpp"

#include "astaroth_utils.h"
#include "kernels.h"
#include "statistics.hpp"

#define OVERLAP 1

// handle IDs
const int H_LNRHO = 0;
const int H_UUX = 1;
const int H_UUY = 2;
const int H_UUZ = 3;
const int H_AX = 4;
const int H_AY = 5;
const int H_AZ = 6;
const int H_ENTROPY = 7;

template <typename T>
static __global__ void sin_ramp_init_kernel(Accessor<T> dst, //<! [out] region to fill
                                            Rect3 dstExt     //<! [in] the extent of the region to initialize
) {
  const T ripple[4] = {0, 0.25, 0, -0.25};
  const size_t period = sizeof(ripple) / sizeof(ripple[0]);

  const size_t tiz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tix = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = dstExt.lo.z + tiz; z < dstExt.hi.z; z += gridDim.z * blockDim.z) {
    for (size_t y = dstExt.lo.y + tiy; y < dstExt.hi.y; y += gridDim.y * blockDim.y) {
      for (size_t x = dstExt.lo.x + tix; x < dstExt.hi.x; x += gridDim.x * blockDim.x) {

        Dim3 p(x, y, z);
        T val = p.x + ripple[p.x % period] + p.y + ripple[p.y % period] + p.z + ripple[p.z % period];
        dst[p] = val;
      }
    }
  }
}

template <typename T>
static __global__ void sin_init_kernel(Accessor<T> dst,    //<! [out] region to fill
                                       const Rect3 dstExt, //<! [in] the extent of the region to initialize
                                       const Dim3 totExt   //<! [in] the total size of the distributed region
) {
  const T amplUU = 0.0001;
  const T period = 16;
  const T pi = 3.141592653589793238462643383;

  const size_t tiz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tix = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = dstExt.lo.z + tiz; z < dstExt.hi.z; z += gridDim.z * blockDim.z) {
    for (size_t y = dstExt.lo.y + tiy; y < dstExt.hi.y; y += gridDim.y * blockDim.y) {
      for (size_t x = dstExt.lo.x + tix; x < dstExt.hi.x; x += gridDim.x * blockDim.x) {

        T val = amplUU * sin(float(y) * 2 * pi / period);
        Dim3 p(x, y, z);
        dst[p] = val;
      }
    }
  }
}

static __device__ uint32_t hash32(uint32_t x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

static __device__ uint64_t hash64(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}

/* bad random numbers from -1 to 1 */
template <typename T>
static __global__ void hash_init_kernel(Accessor<T> dst, //<! [out] region to fill
                                        Rect3 dstExt     //<! [in] the extent of the region to initialize
) {

  const size_t tiz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tix = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = dstExt.lo.z + tiz; z < dstExt.hi.z; z += gridDim.z * blockDim.z) {
    for (size_t y = dstExt.lo.y + tiy; y < dstExt.hi.y; y += gridDim.y * blockDim.y) {
      for (size_t x = dstExt.lo.x + tix; x < dstExt.hi.x; x += gridDim.x * blockDim.x) {

        Dim3 p(x, y, z);
        // create a "random" number from 0 to 1
        T val = float(hash64(p.x) ^ hash64(p.y) ^ hash64(p.z)) / float(uint64_t(-1));
        // shift it to be -1 to 1
        val = (val - 0.5) * 2;
        dst[p] = val;
      }
    }
  }
}

template <typename T>
static __global__ void const_init_kernel(Accessor<T> dst, //<! [out] region to fill
                                         Rect3 dstExt,    //<! [in] the extent of the region to initialize
                                         T val) {

  const size_t tiz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tix = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = dstExt.lo.z + tiz; z < dstExt.hi.z; z += gridDim.z * blockDim.z) {
    for (size_t y = dstExt.lo.y + tiy; y < dstExt.hi.y; y += gridDim.y * blockDim.y) {
      for (size_t x = dstExt.lo.x + tix; x < dstExt.hi.x; x += gridDim.x * blockDim.x) {
        Dim3 p(x, y, z);
        dst[p] = val;
      }
    }
  }
}

template <typename T>
static __global__ void radial_explosion_init_kernel(Accessor<T> uux, Accessor<T> uuy, Accessor<T> uuz,
                                                    const Rect3 dstExt, //<! [in] the extent of the region to initialize
                                                    const Dim3 totExt //<! [in] the total size of the distributed region
) {
  const T amplUU = 1;
  const T shellRadiusUU = 0.8;
  const T widthUU = 0.2;

  // taken from conf
  const T dsx = 0.04908738521;
  const T dsy = 0.04908738521;
  const T dsz = 0.04908738521;

  const double3 orig{0.01, 32 * dsy, 50 * dsz};

  const size_t tiz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tix = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t z = dstExt.lo.z + tiz; z < dstExt.hi.z; z += gridDim.z * blockDim.z) {
    for (size_t y = dstExt.lo.y + tiy; y < dstExt.hi.y; y += gridDim.y * blockDim.y) {
      for (size_t x = dstExt.lo.x + tix; x < dstExt.hi.x; x += gridDim.x * blockDim.x) {

        const Dim3 p(x, y, z);
        T xx = x * dsx - orig.x;
        T yy = y * dsy - orig.y;
        T zz = z * dsz - orig.z;
        // printf("%f %f %f\n", xx, yy, zz);
        const T rr = sqrt(pow(xx, 2.0) + pow(yy, 2.0) + pow(zz, 2.0));

        T theta, phi, uu_radial;

        if (rr > 0) {
          if (zz >= 0) {
            // got a hair over 1.0 here for some reason
            theta = acos(min(1.0, zz / rr));
          } else {
            zz = -zz;
            theta = M_PI - acos(zz / rr);
          }

          if (xx != 0.0) {
            if (xx < 0.0 && yy >= 0.0) {
              //-+
              xx = -xx; // Needs a posite value for atan
              phi = M_PI - atan(yy / xx);

            } else if (xx > 0.0 && yy < 0.0) {
              //+-
              yy = -yy;
              phi = 2.0 * M_PI - atan(yy / xx);
            } else if (xx < 0.0 && yy < 0.0) {
              //--
              yy = -yy;
              xx = -xx;
              phi = M_PI + atan(yy / xx);
            } else {
              //++
              phi = atan(yy / xx);
            }
          } else { // To avoid div by zero with atan
            if (yy > 0.0) {
              phi = M_PI / 2.0;
            } else if (yy < 0.0) {
              phi = (3.0 * M_PI) / 2.0;
            } else {
              phi = 0.0;
            }
          }

          // Set zero for explicit safekeeping
          if (xx == 0.0 && yy == 0.0) {
            phi = 0.0;
          }

          // Gaussian velocity
          // uu_radial = AMPL_UU*exp( -rr2 / (2.0*pow(WIDTH_UU, 2.0)) );
          // New distribution, where that gaussion wave is not in the exact centre coordinates
          // uu_radial = AMPL_UU*exp( -pow((rr - 4.0*WIDTH_UU),2.0) / (2.0*pow(WIDTH_UU, 2.0)) ); //TODO: Parametrize
          // the peak location.
          uu_radial = amplUU * exp(-pow((rr - shellRadiusUU), 2.0) / (2.0 * pow(widthUU, 2.0)));

          if (zz < 1.0 && false)
            printf("%f %f %f -> %f / %f -> %f -> %f\n", xx, yy, zz, -pow((rr - shellRadiusUU), 2.0),
                   (2.0 * pow(widthUU, 2.0)), exp(-pow((rr - shellRadiusUU), 2.0) / (2.0 * pow(widthUU, 2.0))),
                   uu_radial);

        } else {
          uu_radial = 0.0; // TODO: There will be a discontinuity in the origin... Should the shape of the distribution
                           // be different?
        }

        // printf("%f %f %f\n", uu_radial, theta, phi);

        // Determine the cartesian velocity components and lnrho
        uux[p] = uu_radial * sin(theta) * cos(phi);
        uuy[p] = uu_radial * sin(theta) * sin(phi);
        uuz[p] = uu_radial * cos(theta);

        // if (isnan(uux[p])) {
        //   printf("ERR: %e %f %f %f\n", zz/rr, acos(zz/rr), theta, sin(theta));
        // }

        // if (uu_radial > 0) {
        //   printf("[%lu %lu %lu] %.9f %.9f %.9f\n", p.x, p.y, p.z, uu_radial, theta, phi);
        // }
      }
    }
  }
}

int3 decompose_xyz(int p) {

  int3 ret{1, 1, 1};

  for (int pf : prime_factors(p)) {
    if (ret.x <= ret.y && ret.x <= ret.z) {
      ret.x *= pf;
    } else if (ret.y <= ret.z) {
      ret.y *= pf;
    } else {
      ret.z *= pf;
    }
  }
  return ret;
}

int3 decompose_zyx(int p) {

  int3 ret{1, 1, 1};

  for (int pf : prime_factors(p)) {
    if (ret.z <= ret.y && ret.z <= ret.x) {
      ret.z *= pf;
    } else if (ret.y <= ret.x) {
      ret.y *= pf;
    } else {
      ret.x *= pf;
    }
  }
  return ret;
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  argparse::Parser p("Astaroth simulator");
  bool trivialPlacement = false;
  bool randomPlacement = false;
  bool useStaged = false;
  bool useColo = false;
  bool useMemcpyPeer = false;
  bool useKernel = false;
  bool noCompute = false;
  bool paraviewInit = false;
  bool paraviewFinal = false;

  p.add_flag(trivialPlacement, "--trivial")->help("use trivial placement");
  p.add_flag(randomPlacement, "--random")->help("use random placement");
  p.add_flag(useStaged, "--staged")->help("Enable RemoteSender/Recver");
  p.add_flag(useColo, "--colo")->help("Enable ColocatedHaloSender/Recver");
  p.add_flag(useMemcpyPeer, "--peer")->help("Enable PeerAccessSender");
  p.add_flag(useKernel, "--kernel")->help("Enable PeerCopySender");
  p.add_flag(noCompute, "--no-compute")->help("Don't launch compute kernels");
  p.add_flag(paraviewInit, "--paraview-init")->help("write paraview file after init");
  p.add_flag(paraviewFinal, "--paraview-final")->help("write paraview file at end");


  int iters = 1;
  p.add_positional(iters)->required();
  int paraviewPeriod = std::max(1, iters / 10);

  // If there was an error during parsing, report it.
  if (!p.parse(argc, argv)) {
    if (0 == rank) {
      std::cerr << p.help();
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (p.need_help()) {
    if (0 == rank) {
      std::cerr << p.help();
    }
    MPI_Finalize();
    exit(EXIT_SUCCESS);
  }

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

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numSubdoms;
  // int numNodes;
  // int ranksPerNode;
  {
    MpiTopology topo(MPI_COMM_WORLD);
    numSubdoms = size / topo.colocated_size() * devCount;
    // numNodes = size / topo.colocated_size();
    // ranksPerNode = topo.colocated_size();
  }

  if (0 == rank) {
    std::cerr << "assuming " << numSubdoms << " subdomains\n";
  }

  // load config
  // like from config_loader.cc
  AcMeshInfo info{};
  acLoadConfig(AC_DEFAULT_CONFIG, &info);

  // figure out the whole domain size
  Dim3 totExt(info.int_params[AC_nx], info.int_params[AC_ny], info.int_params[AC_nz]);
  {
#if 0
    int3 i3 = decompose_xyz(ranksPerNode);
    totExt.x *= i3.x;
    totExt.y *= i3.y;
    totExt.z *= i3.z;
    i3 = decompose_xyz(numNodes);
    totExt.x *= i3.x;
    totExt.y *= i3.y;
    totExt.z *= i3.z;
#else
    int3 i3 = decompose_zyx(size);
    totExt.x *= i3.x;
    totExt.y *= i3.y;
    totExt.z *= i3.z;
#endif
  }

  if (0 == rank) {
    std::cerr << "AC_nx=" << info.int_params[AC_nx] << "\n";
    std::cerr << "AC_ny=" << info.int_params[AC_ny] << "\n";
    std::cerr << "AC_nz=" << info.int_params[AC_nz] << "\n";
    std::cerr << "AC_mx=" << info.int_params[AC_mx] << "\n";
    std::cerr << "AC_my=" << info.int_params[AC_my] << "\n";
    std::cerr << "AC_mz=" << info.int_params[AC_mz] << "\n";
    std::cerr << "AC_nx_min=" << info.int_params[AC_nx_min] << "\n";
    std::cerr << "AC_ny_min=" << info.int_params[AC_ny_min] << "\n";
    std::cerr << "AC_nz_min=" << info.int_params[AC_nz_min] << "\n";
    std::cerr << "AC_nx_max=" << info.int_params[AC_nx_max] << "\n";
    std::cerr << "AC_ny_max=" << info.int_params[AC_ny_max] << "\n";
    std::cerr << "AC_nz_max=" << info.int_params[AC_nz_max] << "\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  PlacementStrategy strategy = PlacementStrategy::NodeAware;
  if (trivialPlacement) {
    strategy = PlacementStrategy::Trivial;
  } else if (randomPlacement) {
    strategy = PlacementStrategy::IntraNodeRandom;
  }

  Statistics iterTime, exchTime;

  { // scope domains before mpi_finalize
    size_t radius = 3;

    DistributedDomain dd(totExt.x, totExt.y, totExt.z);

    dd.set_methods(methods);
    dd.set_radius(radius);
    dd.set_placement(strategy);

    // add required data
    std::vector<DataHandle<AcReal>> handles;
    // create NUM_VTXBUF_HANDLES values
    handles.push_back(dd.add_data<AcReal>("lnrho"));
    handles.push_back(dd.add_data<AcReal>("uux"));
    handles.push_back(dd.add_data<AcReal>("uuy"));
    handles.push_back(dd.add_data<AcReal>("uuz"));
    handles.push_back(dd.add_data<AcReal>("ax"));
    handles.push_back(dd.add_data<AcReal>("ay"));
    handles.push_back(dd.add_data<AcReal>("az"));
    handles.push_back(dd.add_data<AcReal>("entropy"));
    if (handles.size() != NUM_VTXBUF_HANDLES) {
      std::cerr << "ERROR\n";
      exit(EXIT_FAILURE);
    }

    // dd.set_methods(Method::CudaMpi);

    // create arrays
    std::cerr << "realize\n";
    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);


    // set multigpu_offset for astaroth
    Dim3 orig = dd.get_origin(0);
    info.int3_params[AC_multigpu_offset] = {int(orig.x), int(orig.y), int(orig.z)};


    // create mesh info for each device
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      int device = dd.domains()[di].gpu();
      acDeviceLoadDefaultUniforms(device);
      acDeviceLoadMeshInfo(device, info);
    }

    // create the VBAs for each domain
    std::vector<VertexBufferArray> vbas(dd.domains().size());
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      VertexBufferArray &vba = vbas[di];
      LocalDomain &d = dd.domains()[di];

      // set the appropriate in and out buffers
      for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        vba.in[i] = d.get_curr_accessor<AcReal>(handles[i]).ptr().ptr;
        vba.out[i] = d.get_next_accessor<AcReal>(handles[i]).ptr().ptr;
        std::cerr << rank << ": out[" << i << "]=" << vba.out[i] << "\n";
      }
    }


    // one stream for the interior, plus one stream for each exterior
    std::vector<RcStream> cStreamInterior(dd.domains().size());
    std::vector<std::vector<RcStream>> cStreamExterior(dd.domains().size());
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      int device = dd.domains()[di].gpu();
      cStreamInterior[di] = RcStream(device);
      for (int i = 0; i < 26; ++i) { // 26 possible nbrs
        cStreamExterior[di].push_back(RcStream(device));
      }
    }


#if 1
    std::cerr << "init\n";
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      auto &d = dd.domains()[di];
      d.set_device();
      dim3 dimBlock = Dim3::make_block_dim(d.raw_size(), 512);
      dim3 dimGrid = ((d.raw_size()) + Dim3(dimBlock) - 1) / (Dim3(dimBlock));

      // initialize all
#if 0
      for (DataHandle<AcReal> dh : handles) {
        hash_init_kernel<<<dimGrid, dimBlock, 0, cStreamInterior[di]>>>(d.get_curr_accessor(dh),
                                                                        d.get_compute_region());
      }
#endif

      // sine wave lnrho
#if 0
      sin_init_kernel<<<dimGrid, dimBlock, 0, cStreamInterior[di]>>>(d.get_curr_accessor(handles[H_LNRHO]),
                                                                     d.get_compute_region(), totExt);
#endif

      // radial explostion
#if 1
      // everything to random numbers
      for (DataHandle<AcReal> dh : handles) {
        hash_init_kernel<<<dimGrid, dimBlock, 0, cStreamInterior[di]>>>(d.get_curr_accessor(dh),
                                                                        d.get_compute_region());
      }

      // constant for lnrho
      const_init_kernel<<<dimGrid, dimBlock, 0, cStreamInterior[di]>>>(d.get_curr_accessor(handles[H_LNRHO]),
                                                                       d.get_compute_region(), 0.5);

      // radial explosion for velocity
      radial_explosion_init_kernel<<<dimGrid, dimBlock, 0, cStreamInterior[di]>>>(
          d.get_curr_accessor(handles[H_UUX]), d.get_curr_accessor(handles[H_UUY]), d.get_curr_accessor(handles[H_UUZ]),
          d.get_compute_region(), totExt);
#endif

      CUDA_RUNTIME(cudaDeviceSynchronize());
      std::cerr << "init done\n";
    }
#endif

    if (paraviewInit) {
      dd.write_paraview("init");
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // TODO - add a single exchange and verification code here

    const std::vector<Rect3> interiors = dd.get_interior();
    const std::vector<std::vector<Rect3>> exteriors = dd.get_exterior();
    std::vector<Rect3> fulls;
    for (auto &d : dd.domains()) {
      fulls.push_back(d.get_compute_region());
    }
    // stencil defines compute region in terms of grid points
    // while asteroth does it in terms of memory offset.
    // we will need to add in the offset from the stencil region
    const Dim3 acOff = Dim3(STENCIL_ORDER / 2, STENCIL_ORDER / 2, STENCIL_ORDER / 2);

    for (size_t iter = 0; iter < 10; ++iter) {
      double exchElapsed = 0;
      MPI_Barrier(MPI_COMM_WORLD);
      double iterStart = MPI_Wtime();

      for (int substep = 0; substep < 3; ++substep) {

        // during no-compute, need a barrier here, otherwise we measure any load imbalance
        // as communication time. With compute, the total iteration time is more meaningful and
        // we don't want to add a barrier
        if (noCompute) {
          MPI_Barrier(MPI_COMM_WORLD);
        }

#if OVERLAP
        // launch operations on interior
        for (size_t di = 0; di < dd.domains().size(); ++di) {
          auto &d = dd.domains()[di];
          if (!noCompute) {
            nvtxRangePush("launch");
            Rect3 cr = interiors[di];
            cr.lo += acOff - dd.get_origin(di); // astaroth indexing is memory offset based
            cr.hi += acOff - dd.get_origin(di);
            // std::cerr << rank << ": launch on region=" << cr << " (interior)\n";
            // std::cerr << src0.origin() << "=src0 origin\n";
            d.set_device();
            // acDeviceLoadScalarUniform(d.gpu(), cStreamInterior[di], AC_dt, AC_REAL_EPSILON);
            acDeviceLoadScalarUniform(d.gpu(), cStreamInterior[di], AC_dt, 1e-8);
            integrate_substep(substep, cStreamInterior[di], cr, vbas[di]);
            nvtxRangePop(); // launch
          }
        }
#endif

        // exchange halo
        std::cerr << rank << ": exchange " << iter << "\n";
        double exchStart = MPI_Wtime();
        dd.exchange();
        exchElapsed += MPI_Wtime() - exchStart;

#if OVERLAP
        // launch on exteriors
        for (size_t di = 0; di < dd.domains().size(); ++di) {
          auto &d = dd.domains()[di];
          for (size_t si = 0; si < exteriors[di].size(); ++si) {
            if (!noCompute) {
              nvtxRangePush("launch");
              Rect3 cr = exteriors[di][si];
              cr.lo += acOff - dd.get_origin(di); // astaroth indexing is memory offset based
              cr.hi += acOff - dd.get_origin(di);
              // std::cerr << rank << ": launch on region=" << cr << " (exterior)\n";
              // std::cerr << src0.origin() << "=src0 origin\n";
              d.set_device();
              integrate_substep(substep, cStreamExterior[di][si], cr, vbas[di]);
              nvtxRangePop(); // launch
                              // CUDA_RUNTIME(cudaDeviceSynchronize());
            }
          }
        }
#else
        // launch operations on full region
        for (size_t di = 0; di < dd.domains().size(); ++di) {
          auto &d = dd.domains()[di];
          if (!noCompute) {
            nvtxRangePush("launch");
            Rect3 cr = fulls[di];
            cr.lo += acOff - dd.get_origin(di); // astaroth indexing is memory offset based
            cr.hi += acOff - dd.get_origin(di);
            std::cerr << rank << ": launch on region=" << cr << " (full)\n";
            std::cerr << dd.get_origin(di) << "=dd origin\n";
            d.set_device();
            // acDeviceLoadScalarUniform(d.gpu(), cStreamInterior[di], AC_dt, AC_REAL_EPSILON);
            acDeviceLoadScalarUniform(d.gpu(), cStreamInterior[di], AC_dt, 1e-8);
            integrate_substep(substep, cStreamInterior[di], cr, vbas[di]);
            nvtxRangePop(); // launch
          }
        }
#endif

        if (!noCompute) {
          // wait for stencil to complete
          for (auto &s : cStreamInterior) {
            CUDA_RUNTIME(cudaStreamSynchronize(s));
          }
          for (auto &v : cStreamExterior) {
            for (auto &s : v) {
              CUDA_RUNTIME(cudaStreamSynchronize(s));
            }
          }
        }
      }
      // swap inputs and outputs
      dd.swap();
      // swap vbas
      for (size_t di = 0; di < dd.domains().size(); ++di) {
        VertexBufferArray &vba = vbas[di];
        std::swap(vba.in, vba.out);
      }

      double iterElapsed = MPI_Wtime() - iterStart;

      MPI_Allreduce(MPI_IN_PLACE, &iterElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &exchElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      iterTime.insert(iterElapsed);
      exchTime.insert(exchElapsed);

      if (iter % paraviewPeriod == 0 && iter != 0) {
        std::stringstream ss;
        ss << "iter" << iter;
        dd.write_paraview(ss.str());
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    if (paraviewFinal) {
      dd.write_paraview("final");
      MPI_Barrier(MPI_COMM_WORLD);
    }

  } // send domains out of scope before MPI_Finalize

  if (0 == rank) {
    std::cout << size;
    std::cout << "," << info.int_params[AC_nx];
    std::cout << "," << info.int_params[AC_ny];
    std::cout << "," << info.int_params[AC_nz];
    std::cout << "," << iterTime.trimean();
    std::cout << "," << exchTime.trimean();
    std::cout << "\n";
  }

  MPI_Finalize();

  return 0;
}
