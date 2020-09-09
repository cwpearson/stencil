#include "catch2/catch.hpp"

#include <cstring> // std::memcpy

#include "stencil/accessor.hpp"

TEMPLATE_TEST_CASE("accessor", "[template]", int, double) {
  Dim3 extent(10, 11, 12);
  TestType *data = new TestType[extent.x * extent.y * extent.z];
  PitchedPtr<TestType> ptr(extent.x * sizeof(TestType), data, extent.x * sizeof(TestType), extent.y);
  {
    Dim3 origin(0, 0, 0);
    Accessor<TestType> acc(ptr, origin);
    acc[Dim3(0, 0, 0)] = 10;
    REQUIRE(acc[Dim3(0, 0, 0)] == 10);
    REQUIRE(data[0] == 10);

    acc[Dim3(9, 10, 11)] = 13;
    REQUIRE(acc[Dim3(9, 10, 11)] == 13);
    REQUIRE(data[11 * (10 * 11) + 10 * (10) + 9] == 13);
  }

  // correct origin offset
  {
    Dim3 origin(1, 1, 1);
    Accessor<TestType> acc(data, origin, extent);
    REQUIRE(acc[Dim3(1, 1, 1)] == 10);
  }

  delete[] data;
}