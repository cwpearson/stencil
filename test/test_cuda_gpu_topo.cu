#include "catch2/catch.hpp"

#include "stencil/gpu_topo.hpp"

TEST_CASE("get_gpu_distance_matrix", "[cuda]") {
  INFO("should not fail");
  auto dist = get_gpu_distance_matrix();
}