#include "catch2/catch.hpp"

#include "stencil/gpu_topology.hpp"

TEST_CASE("get_gpu_distance_matrix", "[cuda]") {
  INFO("should not fail");

  
  // expect peer access to self
  GpuTopology t({0});
  t.enable_peer();
  REQUIRE(t.peer(0,0));

}