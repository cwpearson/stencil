set -x -e

source ci/env.sh

which g++
which nvcc
which cmake

g++ --version
nvcc --version
cmake --version

mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_PREFIX_PATH=$OPENMPI_PATH \
  -DMPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/
make VERBOSE=1 