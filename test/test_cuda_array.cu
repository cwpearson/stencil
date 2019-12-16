#include "catch2/catch.hpp"

#include "stencil/array.hpp"

TEMPLATE_TEST_CASE("array", "[cuda][template]", int, double) {
    typedef Array<TestType, device_type::gpu> Array;

    SECTION("ctor") {
        Array arr;
        REQUIRE(arr.size() == 0);
        REQUIRE(arr.data() == nullptr);
    }

    SECTION("ctor") {
        Array arr(10);
        REQUIRE(arr.size() == 10);
        REQUIRE(arr.data() != nullptr);
    }

    SECTION("resize") {
        Array arr;
        arr.resize(11);
        REQUIRE(arr.size() == 11);
        REQUIRE(arr.data() != nullptr);
        arr.resize(0);
        REQUIRE(arr.size() == 0);
        REQUIRE(arr.data() == nullptr);
    }

    SECTION("swap") {
        Array a(10), b(13);
        swap(a,b);
        REQUIRE(a.size() == 13);
        REQUIRE(b.size() == 10);
    }
}
  