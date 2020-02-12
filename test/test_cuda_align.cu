#include "catch2/catch.hpp"

#include "stencil/align.cuh"

TEST_CASE("align", "[cuda]") {
  REQUIRE(0 == next_align_of(0, 0));
  REQUIRE(0 == next_align_of(0, 1));
  REQUIRE(0 == next_align_of(0, 8));
  REQUIRE(8 == next_align_of(1, 8));
  REQUIRE(0x7f5c746071e0 == next_align_of(0x7f5c746071dc, 8));

  REQUIRE(7 == next_align_of(7, 1));
  REQUIRE(8 == next_align_of(7, 2));
  REQUIRE(8 == next_align_of(7, 4));
  REQUIRE(8 == next_align_of(7, 8));
}
