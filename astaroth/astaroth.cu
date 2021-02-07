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

int3 decompose(int p) {

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

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  argparse::Parser p("Astaroth simulator");
  bool trivialPlacement = false;

  p.add_flag(trivialPlacement, "--trivial")->help("use trivial placement");

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

  int devCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&devCount));

  int numSubdoms;
  {
    MpiTopology topo(MPI_COMM_WORLD);
    numSubdoms = size / topo.colocated_size() * devCount;
  }

  if (0 == rank) {
    std::cerr << "assuming " << numSubdoms << " subdomains\n";
  }

  // load config
  // like from config_loader.cc
  AcMeshInfo info{};
  acLoadConfig(AC_DEFAULT_CONFIG, &info);

  // figure out the whole domain size
  {
    int3 i3 = decompose(size);
    info.int_params[AC_nx] *= i3.x;
    info.int_params[AC_ny] *= i3.y;
    info.int_params[AC_nz] *= i3.z;
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
  const int x = info.int_params[AC_nx];
  const int y = info.int_params[AC_ny];
  const int z = info.int_params[AC_nz];
  MPI_Barrier(MPI_COMM_WORLD);

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

  Statistics iterTime, exchTime;

  { // scope domains before mpi_finalize
    size_t radius = 3;

    DistributedDomain dd(x, y, z);

    dd.set_methods(methods);
    dd.set_radius(radius);
    dd.set_placement(strategy);

    // add required data
    std::vector<DataHandle<AcReal>> handles;
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
      handles.push_back(dd.add_data<AcReal>(""));
    }

    // create arrays
    std::cerr << "realize\n";
    dd.realize();

    MPI_Barrier(MPI_COMM_WORLD);

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

    // create mesh info for each device
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      int device = dd.domains()[di].gpu();
      acDeviceLoadDefaultUniforms(device);

      std::cerr << info.int_params[AC_nx] << "\n";
      acDeviceLoadMeshInfo(device, info);
    }

    // create the VBAs for each domain
    std::vector<VertexBufferArray> vbas(dd.domains().size());
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      VertexBufferArray &vba = vbas[di];
      LocalDomain &d = dd.domains()[di];

      for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        vba.in[i] = d.get_curr_accessor<AcReal>(handles[i]).ptr().ptr;
        vba.out[i] = d.get_next_accessor<AcReal>(handles[i]).ptr().ptr;
      }
    }

#if 0
    std::cerr << "init\n";
    for (size_t di = 0; di < dd.domains().size(); ++di) {
      auto &d = dd.domains()[di];
      d.set_device();
      dim3 dimBlock = Dim3::make_block_dim(d.raw_size(), 512);
      dim3 dimGrid = ((d.raw_size()) + Dim3(dimBlock) - 1) / (Dim3(dimBlock));
      init_kernel<<<dimGrid, dimBlock, 0, cStreamInterior[di]>>>(d.get_curr_accessor(dh0), d.get_compute_region(), 10);
      CUDA_RUNTIME(cudaDeviceSynchronize());
    }

    if (0)
      dd.write_paraview("init");
#endif

    const std::vector<Rect3> interiors = dd.get_interior();
    const std::vector<std::vector<Rect3>> exteriors = dd.get_exterior();

    // stencil defines compute region in terms of grid points
    // while asteroth does it in terms of memory offset.
    // we will need to add in the offset from the stencil region
    const Dim3 acOff = Dim3(STENCIL_ORDER / 2, STENCIL_ORDER / 2, STENCIL_ORDER / 2);

    for (size_t iter = 0; iter < 5; ++iter) {

      double iterStart = MPI_Wtime();
      double exchElapsed = 0;

      for (int substep = 0; substep < 3; ++substep) {
        // launch operations on interior
        for (size_t di = 0; di < dd.domains().size(); ++di) {
          auto &d = dd.domains()[di];
          nvtxRangePush("launch");
          Rect3 cr = interiors[di];
          cr.lo += acOff - dd.get_origin(di); // astaroth indexing is memory offset based
          cr.hi += acOff - dd.get_origin(di);
          // std::cerr << rank << ": launch on region=" << cr << " (interior)\n";
          // std::cerr << src0.origin() << "=src0 origin\n";
          d.set_device();
          acDeviceLoadScalarUniform(d.gpu(), cStreamInterior[di], AC_dt, AC_REAL_EPSILON);
          integrate_substep(substep, cStreamInterior[di], cr, vbas[di]);
          nvtxRangePop(); // launch
        }

        // exchange halo
        std::cerr << rank << ": exchange\n";
        double exchStart = MPI_Wtime();
        dd.exchange();
        exchElapsed += MPI_Wtime() - exchStart;

        // launch on exteriors
        for (size_t di = 0; di < dd.domains().size(); ++di) {
          auto &d = dd.domains()[di];
          for (size_t si = 0; si < exteriors[di].size(); ++si) {
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

        // wait for stencil to complete
        for (auto &s : cStreamInterior) {
          CUDA_RUNTIME(cudaStreamSynchronize(s));
        }
        for (auto &v : cStreamExterior) {
          for (auto &s : v) {
            CUDA_RUNTIME(cudaStreamSynchronize(s));
          }
        }

        // swap inputs and outputs
        dd.swap();
      }

      double iterElapsed = MPI_Wtime() - iterStart;

      MPI_Allreduce(MPI_IN_PLACE, &iterElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &exchElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      iterTime.insert(iterElapsed);
      exchTime.insert(exchElapsed);
    }

    if (0)
      dd.write_paraview("final");

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
