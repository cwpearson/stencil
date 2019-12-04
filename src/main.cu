#include "stencil/stencil.cuh"

int main(int argc, char **argv) {

  size_t x = 100;
  size_t y = 10;
  size_t z = 300;

  MPI_Init(&argc, &argv);

  Domain d(x, y, z);

  auto r = d.add_radius(1, 0, 3);

  auto pressureHandle = d.add_data<double>();
  auto temperatureHandle = d.add_data<int>();

  d.realize();

  auto *pressure = d.get_data(pressureHandle);
  auto *temperature = d.get_data(temperatureHandle);

  d.exchange(r);
  d.exchange(r);

  MPI_Finalize();

  return 0;
}