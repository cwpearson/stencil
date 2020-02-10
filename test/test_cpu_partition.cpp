#include "catch2/catch.hpp"

#include <iostream>

#include "stencil/partition.hpp"

TEST_CASE("partition") {

  SECTION("10x5x5 into 2x1x1") {

    Dim3 sz(10, 5, 5);
    int n = 2;

    RankPartition part(sz, n);

    REQUIRE(Dim3(2,1,1) == part.dim());
    REQUIRE(Dim3(5, 5, 5) == part.subdomain_size(Dim3(0,0,0)));
    REQUIRE(Dim3(5, 5, 5) == part.subdomain_size(Dim3(1,0,0)));

  }

  SECTION("10x3x1 into 4x1") {

    Dim3 sz(10, 3, 1);
    int n = 4;

    RankPartition part(sz, n);

    REQUIRE(Dim3(3, 3, 1) == part.subdomain_size(Dim3(0,0,0)));
    REQUIRE(Dim3(3, 3, 1) == part.subdomain_size(Dim3(1,0,0)));
    REQUIRE(Dim3(2, 3, 1) == part.subdomain_size(Dim3(2,0,0)));
    REQUIRE(Dim3(2, 3, 1) == part.subdomain_size(Dim3(3,0,0)));
  }

  SECTION("10x5x5 into 3x1") {

    Dim3 sz(10, 5, 5);
    int n = 3;

    RankPartition part(sz, n);

    REQUIRE(Dim3(4, 5, 5) == part.subdomain_size(Dim3(0,0,0)));
    REQUIRE(Dim3(3, 5, 5) == part.subdomain_size(Dim3(1,0,0)));
    REQUIRE(Dim3(3, 5, 5) == part.subdomain_size(Dim3(2,0,0)));
  }

  SECTION("13x7x7 into 4x1") {

    Dim3 sz(13, 7, 7);
    int n = 4;

    RankPartition part(sz, n);

    REQUIRE(Dim3(4, 7, 7) == part.subdomain_size(Dim3(0,0,0)));
    REQUIRE(Dim3(3, 7, 7) == part.subdomain_size(Dim3(1,0,0)));
    REQUIRE(Dim3(3, 7, 7) == part.subdomain_size(Dim3(2,0,0)));
    REQUIRE(Dim3(3, 7, 7) == part.subdomain_size(Dim3(3,0,0)));

  }

}