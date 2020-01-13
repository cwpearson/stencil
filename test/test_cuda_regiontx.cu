#include "catch2/catch.hpp"

#include "stencil/local_domain.cuh"
#include "stencil/tx_cuda.cuh"

TEMPLATE_TEST_CASE("region sender recver", "[cuda][mpi][template]", int,
                   double) {

  const Dim3 sz(30, 40, 50);
  const int gpu = 0;
  const size_t radius = 4;
  LocalDomain d0(sz, gpu);
  LocalDomain d1(sz, gpu);

  d0.set_radius(radius);
  d1.set_radius(radius);

  auto h0 = d0.add_data<TestType>();
  auto h1 = d1.add_data<TestType>();

  d0.realize();
  d1.realize();

  const int srcRank = 0;
  const int dstRank = 0;
  const int srcGPU = 0;
  const int dstGPU = 0;

  SECTION("+x +y +z corner") {
    std::cerr << "start +x +y +z corner test\n";
    Dim3 dir(1, 1, 1);

    RegionSender<AnySender> sender(d0, srcRank, srcGPU, dstRank, dstGPU, dir);
    RegionRecver<AnyRecver> recver(d1, srcRank, srcGPU, dstRank, dstGPU, dir);

    sender.allocate();
    recver.allocate();
    sender.send();
    recver.recv();
    sender.wait();
    recver.wait();
  }

  SECTION("+x +y edge") {
    std::cerr << "start +x +y edge test\n";
    Dim3 dir(1, 1, 0);

    RegionSender<AnySender> sender(d0, srcRank, srcGPU, dstRank, dstGPU, dir);
    RegionRecver<AnyRecver> recver(d1, srcRank, srcGPU, dstRank, dstGPU, dir);

    sender.allocate();
    recver.allocate();
    sender.send();
    recver.recv();
    sender.wait();
    recver.wait();
  }

  SECTION("+x face") {
    std::cerr << "start +x face test\n";
    Dim3 dir(1, 0, 0);

    RegionSender<AnySender> sender(d0, srcRank, srcGPU, dstRank, dstGPU, dir);
    RegionRecver<AnyRecver> recver(d1, srcRank, srcGPU, dstRank, dstGPU, dir);

    sender.allocate();
    recver.allocate();
    sender.send();
    recver.recv();
    sender.wait();
    recver.wait();
  }
}
