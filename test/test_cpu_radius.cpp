#include "catch2/catch.hpp"

#include "stencil/radius.hpp"

TEST_CASE("radius") {

  Radius r0, r1;

  SECTION("copy") {
    Radius r3 = Radius::constant(1);
    Radius r2(r3);
    REQUIRE(r3 == r2);
  }

  SECTION("constant") {
    r0 = Radius::constant(0);
    r1 = Radius::constant(0);

    REQUIRE(r0.dir(0, 0, 0) == r1.dir(0, 0, 0));
    REQUIRE(r0.x(-1) == 0);
    REQUIRE(r0.x(1) == 0);
  }

  SECTION("face") {
    r0 = Radius::constant(0);
    r0.set_face(1);

    REQUIRE(r0.x(-1) == 1);
    REQUIRE(r0.x(1) == 1);
    REQUIRE(r0.y(-1) == 1);
    REQUIRE(r0.y(1) == 1);
    REQUIRE(r0.z(-1) == 1);
    REQUIRE(r0.z(1) == 1);
  }
}