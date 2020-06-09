#pragma once

#if STENCIL_USE_MPI == 1
#include <mpi.h>
#endif

namespace mpi {

inline int world_rank() {
#if STENCIL_USE_MPI == 1
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

inline int world_size() {
#if STENCIL_USE_MPI == 1
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
#else
  return 1;
#endif
}

} // namespace mpi