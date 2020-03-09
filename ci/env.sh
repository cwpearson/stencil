OPENMPI_PREFIX=$HOME/openmpi
CMAKE_PREFIX=$HOME/cmake

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PATH=$OPENMPI_PREFIX/bin:$PATH
export PATH=$CMAKE_PREFIX/bin:$PATH