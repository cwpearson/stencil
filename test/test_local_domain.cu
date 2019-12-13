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

  SECTION("edge") {
    REQUIRE(Dim3(0, 0, 0) == d0.edge_pos(0, 1, false, false)); // -x -y
    REQUIRE(Dim3(34, 0, 0) == d0.edge_pos(0, 1, true, false)); // +x -y
    REQUIRE(Dim3(0, 44, 0) == d0.edge_pos(0, 1, false, true)); // -x +y
    REQUIRE(Dim3(34, 44, 0) == d0.edge_pos(0, 1, true, true)); // +x +y
    REQUIRE(Dim3(4, 4, 50) == d0.edge_extent(0, 1));

    REQUIRE(Dim3(0, 0, 0) == d0.edge_pos(0, 2, false, false)); // -x -z
    REQUIRE(Dim3(34, 0, 0) == d0.edge_pos(0, 2, true, false)); // +x -z
    REQUIRE(Dim3(0, 0, 54) == d0.edge_pos(0, 2, false, true)); // -x +z
    REQUIRE(Dim3(34, 0, 54) == d0.edge_pos(0, 2, true, true)); // +x +z
    REQUIRE(Dim3(4, 40, 4) == d0.edge_extent(0, 2));

    REQUIRE(Dim3(0, 0, 0) == d0.edge_pos(1, 2, false, false)); // -y -z
    REQUIRE(Dim3(0, 44, 0) == d0.edge_pos(1, 2, true, false)); // +y -z
    REQUIRE(Dim3(0, 0, 54) == d0.edge_pos(1, 2, false, true)); // -y +z
    REQUIRE(Dim3(0, 44, 54) == d0.edge_pos(1, 2, true, true)); // +y +z
    REQUIRE(Dim3(30, 4, 4) == d0.edge_extent(1, 2));
  }

  SECTION("edge position dimension order doesn't matter") {
    REQUIRE(d0.edge_pos(0, 1, false, false) == d0.edge_pos(1, 0, false, false));
  }
}