#include "catch2/catch.hpp"

#include <iostream>

#include "stencil/partition.hpp"

TEST_CASE("partition") {

  SECTION("10x5x5 into 2x1") {

    Dim3 sz(10, 5, 5);
    int ranks = 2;
    int gpus = 1;

    Partition *part = new PFP(sz, ranks, gpus);

    REQUIRE(0 == part->get_rank(Dim3(0, 0, 0)));
    REQUIRE(Dim3(1, 1, 1) == part->gpu_dim());
    REQUIRE(Dim3(2, 1, 1) == part->rank_dim());

    for (int i = 0; i < ranks; ++i) {
      REQUIRE(part->rank_idx(i) < part->rank_dim());
      REQUIRE(part->rank_idx(i) >= 0);
      REQUIRE(part->get_rank(part->rank_idx(i)) == i);
    }

    for (int i = 0; i < gpus; ++i) {
      REQUIRE(part->gpu_idx(i) < part->gpu_dim());
      REQUIRE(part->gpu_idx(i) >= 0);
      REQUIRE(part->get_gpu(part->gpu_idx(i)) == i);
    }

    REQUIRE(Dim3(5, 5, 5) == part->local_domain_size(Dim3(0,0,0), Dim3(0,0,0)));

    delete part;
  }

  SECTION("10x5x5 into 3x1") {

    Dim3 sz(10, 5, 5);
    int ranks = 3;
    int gpus = 1;

    Partition *part = new PFP(sz, ranks, gpus);

    std::cerr << "here\n";
    REQUIRE(Dim3(4, 5, 5) == part->local_domain_size(Dim3(0,0,0), Dim3(0,0,0)));
    REQUIRE(Dim3(4, 5, 5) == part->local_domain_size(Dim3(1,0,0), Dim3(0,0,0)));
    REQUIRE(Dim3(3, 5, 5) == part->local_domain_size(Dim3(2,0,0), Dim3(0,0,0)));

    delete part;
  }

  SECTION("13x7x7 into 4x1") {

    Dim3 sz(13, 7, 7);
    int ranks = 4;
    int gpus = 1;

    Partition *part = new PFP(sz, ranks, gpus);

    std::cerr << "here\n";
    REQUIRE(Dim3(4, 7, 7) == part->local_domain_size(Dim3(0,0,0), Dim3(0,0,0)));
    REQUIRE(Dim3(3, 7, 7) == part->local_domain_size(Dim3(1,0,0), Dim3(0,0,0)));
    REQUIRE(Dim3(3, 7, 7) == part->local_domain_size(Dim3(2,0,0), Dim3(0,0,0)));
    REQUIRE(Dim3(3, 7, 7) == part->local_domain_size(Dim3(3,0,0), Dim3(0,0,0)));

    delete part;
  }

}