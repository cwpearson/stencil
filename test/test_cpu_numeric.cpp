#include "catch2/catch.hpp"

#include <iostream>

#include "stencil/numeric.hpp"

TEST_CASE("prime_factors") {

  SECTION("6") {

    auto pfs = prime_factors(6);
    REQUIRE(pfs[0] == 3);
    REQUIRE(pfs[1] == 2);

  }

}