#include "stencil/stencil.cuh"

int main(int argc, char **argv) {

  size_t x = 100;
  size_t y = 10;
  size_t z = 300;

  MPI_Init(&argc, &argv);

  DistributedDomain dd(x, y, z);

  dd.set_radius(2);

  auto pressureHandle = dd.add_data<double>();
  auto temperatureHandle = dd.add_data<int>();

  dd.realize();
  printf("main(): realize finished\n");

  for (auto &d : dd.domains()) {
    auto *pressure = d.get_curr(pressureHandle);
    auto *temperature = d.get_curr(temperatureHandle);
    auto *np = d.get_next(pressureHandle);
    auto *nt = d.get_next(temperatureHandle);
  }

  printf("main(): call exchange\n");
  dd.exchange();
  printf("main(): call exchange\n");
  dd.exchange();

  MPI_Finalize();

  return 0;
}