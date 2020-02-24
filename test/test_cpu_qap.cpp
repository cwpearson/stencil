#include "catch2/catch.hpp"

#include <iostream>

#include "stencil/mat2d.hpp"
#include "stencil/qap.hpp"

TEST_CASE("qap") {

  const double inf = std::numeric_limits<double>::infinity();

  SECTION("unbalanced triangle") {
    // high bw between 0-2
    Mat2D<double> bw = {{inf, 1, 10}, {1, inf, 1}, {10, 1, inf}};
    // high cost between  0-1
    Mat2D<double> comm = {{0, 10, 1}, {10, 0, 1}, {1, 1, 0}};

    INFO("reciprocal");
    Mat2D<double> dist = make_reciprocal(bw);
    INFO("solve");
    auto f = qap::solve(comm, dist);

    INFO("check");
    REQUIRE(f[0] == 0);
    REQUIRE(f[1] == 2);
    REQUIRE(f[2] == 1);
  }

  SECTION("p9") {
    // high bw between 0-2
    // clang-format off
    Mat2D<double> bw = {
      {900,  75,  64,  64},
      { 75, 900,  64,  64},
      { 64,  64, 900,  75},
      { 64,  64,  75, 900}
    };
    // high cost between  0-2, 1-3
    Mat2D<double> comm = {
      {  7,  5,  10,  1},
      {  5,  7,  1,  10},
      { 10,  1,  7,  5},
      { 1,  10,  5, 7}
    };
    // clang-format on

    INFO("reciprocal");
    Mat2D<double> dist = make_reciprocal(bw);
    INFO("solve");
    auto f = qap::solve(comm, dist);

    INFO("check");
    REQUIRE(f[0] == 0);
    REQUIRE(f[1] == 2);
    REQUIRE(f[2] == 1);
    REQUIRE(f[3] == 3);
  }

  SECTION("p9_catch") {
    // high bw between 0-2
    // clang-format off
    Mat2D<double> bw = {
      {900,  75,  64,  64},
      { 75, 900,  64,  64},
      { 64,  64, 900,  75},
      { 64,  64,  75, 900}
    };
    // high cost between  0-2, 1-3
    Mat2D<double> comm = {
      {  7,  5,  10,  1},
      {  5,  7,  1,  10},
      { 10,  1,  7,  5},
      { 1,  10,  5, 7}
    };
    // clang-format on

    INFO("reciprocal");
    Mat2D<double> dist = make_reciprocal(bw);
    INFO("solve");
    auto f = qap::solve_catch(comm, dist);

    INFO("check");
    REQUIRE(f[0] == 3);
    REQUIRE(f[1] == 1);
    REQUIRE(f[2] == 2);
    REQUIRE(f[3] == 0);
  }

}