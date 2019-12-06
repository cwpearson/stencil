#include "stencil/stencil.cuh"

int main(int argc, char **argv) {

  size_t x = 100;
  size_t y = 10;
  size_t z = 300;

  MPI_Init(&argc, &argv);

  DistributedDomain dd(x, y, z);

  dd.set_radius(1, 0, 3);

  auto pressureHandle = dd.add_data<double>();
  auto temperatureHandle = dd.add_data<int>();

  dd.realize();

  for (auto &d : dd.domains()) {
    auto *pressure = d.get_data(pressureHandle);
    auto *temperature = d.get_data(temperatureHandle);
  }

  dd.exchange();
  dd.exchange();

  MPI_Finalize();

  return 0;
}