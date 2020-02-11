#include "catch2/catch.hpp"

#include "stencil/gpu_topology.hpp"

TEST_CASE("gpu_topo", "[cuda]") {
  INFO("should not fail");

  // expect peer access to self
  gpu_topo::enable_peer(0,0);
  REQUIRE(gpu_topo::peer(0,0));

}