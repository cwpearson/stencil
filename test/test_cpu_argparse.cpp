#include "catch2/catch.hpp"

#include <iostream>

#include "stencil/argparse.hpp"

TEST_CASE("argparse") {

  SECTION("types") {
    const char *argv[] = {
        "some-exe", "--campi", "--f", "10", "1.7", "1.8",
        "--", // stop looking for options
        "--a string",
        "-6",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    Parser p;

    bool campi = false;
    size_t x;
    double d;
    float f;
    int i;
    std::string s;
    p.add_flag(campi, "--campi");
    p.add_positional(x);
    p.add_positional(d);
    p.add_positional(f);
    p.add_positional(s);
    p.add_positional(i);
    REQUIRE(p.parse(argc, argv));

    REQUIRE(campi == true);
    REQUIRE(x == 10);
    REQUIRE(d == 1.7);
    REQUIRE(f == 1.8f);
    REQUIRE(s == "--a string");
    REQUIRE(i == -6);
    REQUIRE(p.need_help() == false);
  }

  SECTION("no args") {
    const char *argv[] = {};
    int argc = sizeof(argv) / sizeof(argv[0]);
    Parser p;
    REQUIRE(p.parse(argc, argv));
  }


  SECTION("description") {
    const char *argv[] = {
        "some-exe",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    Parser p("a test program");
    REQUIRE(p.parse(argc, argv));
  }

  SECTION("skip-first") {
    const char *argv[] = {"some-exe"};
    int argc = sizeof(argv) / sizeof(argv[0]);

    Parser p;
    p.no_unrecognized();
    REQUIRE(p.parse(argc, argv));
  }

  SECTION("no-unrecognized") {
    const char *argv[] = {"some-exe", "-f"};
    int argc = sizeof(argv) / sizeof(argv[0]);

    Parser p;
    p.no_unrecognized();
    REQUIRE(false == p.parse(argc, argv));
  }

  SECTION("missing-reqd-posnl") {
    const char *argv[] = {"some-exe", "a"};
    int argc = sizeof(argv) / sizeof(argv[0]);

    std::string a;
    std::string b;

    Parser p;
    p.add_positional(a)->required();
    p.add_positional(b)->required();
    REQUIRE(false == p.parse(argc, argv));
  }

  SECTION("-h") {
    const char *argv[] = {"some-exe", "-h"};
    int argc = sizeof(argv) / sizeof(argv[0]);

    std::string a;
    std::string b;

    Parser p;
    REQUIRE(true == p.parse(argc, argv));
    REQUIRE(p.need_help());

    std::cerr << p.help() << "\n";
  }

  SECTION("--help") {
    const char *argv[] = {"some-exe", "--help"};
    int argc = sizeof(argv) / sizeof(argv[0]);

    std::string a;
    std::string b;

    Parser p;
    REQUIRE(true == p.parse(argc, argv));
    REQUIRE(p.need_help());
  }

}
