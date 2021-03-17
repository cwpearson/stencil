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

TEST_CASE("get_max_abs_error") {

  using namespace Catch::literals;

  SECTION("double 0.1") {
    std::vector<double> a{1.0, 2.0, 3.0};
    std::vector<double> b{1.1, 2.0, 3.0};
    REQUIRE(0.1_a == get_max_abs_error(a.data(), b.data(), a.size()));
  }

    SECTION("double 0.0") {
    std::vector<double> a{1.0, 2.0, 3.0};
    std::vector<double> b{1.0, 2.0, 3.0};
    REQUIRE(0.0_a == get_max_abs_error(a.data(), b.data(), a.size()));
  }
}