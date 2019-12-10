#include "catch2/catch.hpp"

#include "stencil/local_domain.cuh"

TEMPLATE_TEST_CASE("local domain", "[cuda][template]", int, double) {

  const Dim3 sz(30, 40, 50);
  const int gpu = 0;
  const size_t radius = 4;
  LocalDomain d0(sz, gpu);

  d0.set_radius(radius);
  auto handle = d0.add_data<TestType>();


  d0.realize();
  TestType *p = d0.get_curr(handle);
  REQUIRE(p != nullptr);

  REQUIRE(d0.face_bytes(0, 0) == sizeof(TestType) * sz.y * sz.z * radius);
}