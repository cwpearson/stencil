#include "catch2/catch.hpp"

#include "stencil/tx_cuda.cuh"

#define MESSAGE(x, y, z) Message(Dim3(x, y, z), 0, 0)

TEST_CASE("message") {

  REQUIRE(!MESSAGE(0, 0, 0).contains(MESSAGE(1, 0, 0)));
  REQUIRE(MESSAGE(0, 0, 0).contains(MESSAGE(0, 0, 0)));

  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, -1, -1)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, -1, 0)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, -1, 1)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, 0, -1)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, 0, 0)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, 0, 1)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, 1, -1)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, 1, 0)));
  REQUIRE(MESSAGE(1, 0, 0).contains(MESSAGE(1, 1, 1)));

  REQUIRE(!MESSAGE(1, -1, -1).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, -1, 0).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, -1, 1).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, 0, -1).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, 0, 1).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, 1, -1).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, 1, 0).contains(MESSAGE(1, 0, 0)));
  REQUIRE(!MESSAGE(1, 1, 1).contains(MESSAGE(1, 0, 0)));

  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, -1, -1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, -1, 0)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, -1, 1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, 0, -1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, 0, 0)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, 0, 1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, 1, -1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, 1, 0)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(0, 1, 1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, -1, -1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, -1, 0)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, -1, 1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, 0, -1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, 0, 0)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, 0, 1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, 1, -1)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, 1, 0)));
  REQUIRE(!MESSAGE(1, 0, 0).contains(MESSAGE(-1, 1, 1)));

  SECTION("overlap") {
    std::vector<Message> msgs;
    msgs.push_back(MESSAGE(1, 0, 0));
    msgs.push_back(MESSAGE(1, 0, 0));
    msgs.push_back(MESSAGE(1, 1, 0));
    msgs.push_back(MESSAGE(0, 0, 0));

    auto result = Message::remove_overlapping(msgs);
    REQUIRE(result.size() == 2);
    REQUIRE(result[0] == MESSAGE(1, 0, 0));
    REQUIRE(result[1] == MESSAGE(0, 0, 0));
  }

  SECTION("real case") {
    REQUIRE(MESSAGE(0, 0, 1).contains(MESSAGE(1, 1, 1)));
    REQUIRE(MESSAGE(0, 0, 1).contains(MESSAGE(0, 1, 1)));

    std::vector<Message> msgs;
    msgs.push_back(MESSAGE(1, 1, 1));
    msgs.push_back(MESSAGE(0, 1, 1));
    msgs.push_back(MESSAGE(0, 0, 1));

    auto result = Message::remove_overlapping(msgs);

    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == MESSAGE(0, 0, 1));
  }
}