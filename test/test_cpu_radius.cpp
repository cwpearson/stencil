#include "catch2/catch.hpp"

#include "stencil/radius.hpp"

TEST_CASE("radius") {

  Radius r0, r1;

  r0 = Radius::constant_radius(0);
  r1 = Radius::constant_radius(0);

  REQUIRE(r0(0, 0, 0) == r1(0, 0, 0));

  SECTION("copy") {
    Radius r3 = Radius::constant_radius(1);
    Radius r2(r3);
    REQUIRE(r3 == r2);
  }
}