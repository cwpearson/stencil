#include "catch2/catch.hpp"

#include "stencil/rcstream.hpp"

TEST_CASE("ctor", "[cuda]") {
  RcStream s;

  SECTION("copy ctor") {
    RcStream t(s);
    REQUIRE(t == s);
  }

  SECTION("move ctor") {
    auto temp = cudaStream_t(s);
    RcStream t(std::move(s));
    REQUIRE(cudaStream_t(t) == temp);
  }

  SECTION("copy assign") {
    RcStream t;
    t = s;
    REQUIRE(t == s);
  }

  SECTION("move assign") {
    RcStream t;
    auto temp = cudaStream_t(s);
    t = std::move(s);
    REQUIRE(cudaStream_t(t) == temp);
  }
}