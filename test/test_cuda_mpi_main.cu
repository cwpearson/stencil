//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
// this in one cpp file #include "catch2/catch.hpp"

// https://github.com/catchorg/Catch2/blob/master/docs/own-main.md#top
#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include <mpi.h>

int main(int argc, char *argv[]) {
  // global setup...

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (MPI_THREAD_MULTIPLE != provided) {
    assert(0 && "require MPI_THREAD_MULTIPLE");
  }

  int result = Catch::Session().run(argc, argv);

  MPI_Finalize();

  return result;
}