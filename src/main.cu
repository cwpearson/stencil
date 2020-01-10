#include "stencil/stencil.hpp"

int main(int argc, char **argv) {

  size_t x = 100;
  size_t y = 10;
  size_t z = 300;

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(provided == MPI_THREAD_MULTIPLE);

  DistributedDomain dd(x, y, z);

  dd.set_radius(2);

  dd.add_data<int64_t>();

  dd.realize();
  printf("main(): realize finished\n");

  for (auto &d : dd.domains()) {
  }

  printf("main(): call exchange\n");
  dd.exchange();
  printf("main(): done exchange\n");

  MPI_Finalize();

  return 0;
}