#include "catch2/catch.hpp"

#include "stencil/tx.hpp"

TEST_CASE("tag") {

  REQUIRE(make_tag(0, 0, Dim3(0, 0, 0)) != make_tag(0, 0, Dim3(0, 0, -1)));
  REQUIRE(make_tag(0, 0, Dim3(0, 0, 0)) == make_tag(0, 0, Dim3(0, 0, 0)));
  REQUIRE(make_tag(0, 1, Dim3(0, 0, 0)) != make_tag(0, 0, Dim3(0, 0, 0)));
}