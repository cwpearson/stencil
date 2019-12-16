#include "catch2/catch.hpp"

#include "stencil/array.hpp"

TEMPLATE_TEST_CASE("array", "[template]", int, double) {
    typedef Array<TestType> Array;

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

    SECTION("element access") {
        Array arr;
        arr.resize(13);
        arr[0] = 10;
        arr[12] = 27;
        REQUIRE(arr[0] == 10);
        REQUIRE(arr[12] == 27);
    }
}
  