//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include "catch2/catch.hpp"

// https://github.com/catchorg/Catch2/blob/master/docs/own-main.md#top
#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include <mpi.h>

#include <cassert>
#include <iostream>

int main( int argc, char* argv[] ) {

  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess) {
    std::cerr << "couldn't get device count";
    return EXIT_FAILURE;
  }
  if (0 >= count) {
    std::cerr << "tests require at least one GPU";
    return EXIT_FAILURE;
  }

  MPI_Init(nullptr, nullptr);

  int result = Catch::Session().run( argc, argv );

  MPI_Finalize();

  return result;
}