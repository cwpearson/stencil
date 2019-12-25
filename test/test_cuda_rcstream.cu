#include "catch2/catch.hpp"

#include "stencil/rcstream.hpp"

TEST_CASE("ctor", "[cuda]") {
  RcStream s;

  SECTION("copy ctor") {
    std::cerr << "test: copy ctor\n";
    RcStream t(s);
    REQUIRE(t == s);
  }

  SECTION("move ctor") {
    std::cerr << "test: move ctor\n";
    auto temp = cudaStream_t(s);
    RcStream t(std::move(s));
    REQUIRE(cudaStream_t(t) == temp);
  }

  SECTION("copy assign") {
    std::cerr << "test: copy assign\n";
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