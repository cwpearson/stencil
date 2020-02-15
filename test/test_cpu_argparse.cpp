#include "catch2/catch.hpp"

#include <iostream>

#include "stencil/argparse.hpp"

TEST_CASE("argparse") {
    
SECTION("") {

    
    const char* argv[] = {
        "some-exe",
        "--campi",
        "--f",
        "10",
        "1.7",
        "1.8",
        "--", // stop looking for options
        "-6",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);


    Parser p;

    bool campi = false;
    size_t x;
    double d;
    float f;
    int i;
    p.add_flag(campi, "--campi");
    p.add_positional(x);
    p.add_positional(d);
    p.add_positional(f);
    p.add_positional(i);
    p.parse(argc, argv);

    REQUIRE(campi == true);
    REQUIRE(x == 10);
    REQUIRE(d == 1.7);
    REQUIRE(f == 1.8f);
    REQUIRE(i == -6);



}

}
  