#include "catch2/catch.hpp"

#include "stencil/mat2d.hpp"

TEST_CASE("mat2d") {

  SECTION("value ctor") {
    Mat2D<double> tmp(10, 10, 0);
    for (auto i = 0; i < tmp.shape().y; ++i) {
      for (auto j = 0; j < tmp.shape().x; ++j) {
        REQUIRE(tmp[i][j] == 0);
      }
    }
  }

  SECTION("brace list") {
    Mat2D<double> c = {{0, 10, 1}, {10, 0, 1}, {1, 1, 0}};
    REQUIRE(c.shape() == Shape(3, 3));
    REQUIRE(c[0][0] == 0);
    REQUIRE(c[0][1] == 10);
    REQUIRE(c[0][2] == 1);
    REQUIRE(c[1][0] == 10);
    REQUIRE(c[1][1] == 0);
    REQUIRE(c[1][2] == 1);
    REQUIRE(c[2][0] == 1);
    REQUIRE(c[2][1] == 1);
    REQUIRE(c[2][2] == 0);
  }

  SECTION("row push") {
    Mat2D<double> c(4, 1, 0);
    std::vector<double> v = {1, 2, 3, 4};
    c.push_back(v);
    REQUIRE(c.shape() == Shape(4, 2));
    REQUIRE(c[0][0] == 0);
    REQUIRE(c[0][1] == 0);
    REQUIRE(c[0][2] == 0);
    REQUIRE(c[0][3] == 0);
    REQUIRE(c[1][0] == 1);
    REQUIRE(c[1][1] == 2);
    REQUIRE(c[1][2] == 3);
    REQUIRE(c[1][3] == 4);

    v = {5, 6, 7, 8};
    c.push_back(v);
    REQUIRE(c.shape() == Shape(4, 3));
    REQUIRE(c[2][0] == 5);
    REQUIRE(c[2][1] == 6);
    REQUIRE(c[2][2] == 7);
    REQUIRE(c[2][3] == 8);
  }

  Mat2D<double> a(10, 10);
  for (auto i = 0; i < a.shape().y; ++i) {
    for (auto j = 0; j < a.shape().x; ++j) {
      a[i][j] = i * 10 + j;
    }
  }
  Shape ashp = a.shape();

  SECTION("copy-ctor") {
    Mat2D<double> b(a);
    REQUIRE(b == a);
  }

  SECTION("copy-assign") {
    Mat2D<double> b = a;
    REQUIRE(b == a);
  }

  SECTION("move-ctor") {
    Mat2D<double> b(std::move(a));
    REQUIRE(b.shape() == ashp);

    for (auto i = 0; i < b.shape().y; ++i) {
      for (auto j = 0; j < b.shape().x; ++j) {
        REQUIRE(b[i][j] == i * 10 + j);
      }
    }
  }

  SECTION("move-assign") {
    Mat2D<double> b = std::move(a);
    REQUIRE(b.shape() == ashp);
    for (auto i = 0; i < b.shape().y; ++i) {
      for (auto j = 0; j < b.shape().x; ++j) {
        REQUIRE(b[i][j] == i * 10 + j);
      }
    }
  }

  SECTION("row") {
    Mat2D<double> c(10, 10);
    for (auto i = 0; i < a.shape().y; ++i) {
      c[i] = a[i];
    }
    REQUIRE(c == a);
  }
}