#include "catch2/catch.hpp"

#include "stencil/local_domain.cuh"
#include "stencil/tx_cuda.cuh"

TEMPLATE_TEST_CASE("pack memcpy copier", "[cuda][template]", int, double) {

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

  SECTION("+x +y +z corner") {
    Dim3 dir(1, 1, 1);

    PackMemcpyCopier copier(d1, d0, dir);
    copier.allocate();
    copier.send();
    copier.wait();
  }

  SECTION("+x +y edge") {
    Dim3 dir(1, 1, 0);

    PackMemcpyCopier copier(d1, d0, dir);
    copier.allocate();
    copier.send();
    copier.wait();
  }

  SECTION("+x face") {
    Dim3 dir(1, 0, 0);

    PackMemcpyCopier copier(d1, d0, dir);
    copier.allocate();
    copier.send();
    copier.wait();
  }
}
