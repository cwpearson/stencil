set -x -e

source ci/env.sh

# Install Cmake
mkdir -p $CMAKE_PREFIX
if [[ $TRAVIS_CPU_ARCH == "ppc64le" ]]; then
    wget -SL https://cmake.org/files/v3.13/cmake-3.13.5.tar.gz -O cmake.tar.gz
    tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
    rm cmake.tar.gz
    cd $CMAKE_PREFIX
    ./bootstrap --prefix=$CMAKE_PREFIX
    make install
elif [[ $TRAVIS_CPU_ARCH == "amd64" ]]; then
    wget -SL https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.tar.gz -O cmake.tar.gz
    tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
    rm cmake.tar.gz
fi
cd $HOME

sudo apt-get update 
sudo apt-get install -y --no-install-recommends \
  libopenmpi-dev openmpi-bin

## install CUDA
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

if [[ $TRAVIS_CPU_ARCH == "ppc64le" ]]; then
    CUDA102="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64le/cuda-repo-ubuntu1804_10.2.89-1_ppc64le.deb"
elif [[ $TRAVIS_CPU_ARCH == "amd64" ]]; then
    CUDA102="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb"
fi

wget -SL $CUDA102 -O cuda.deb
sudo dpkg -i cuda.deb
sudo apt-get update 
sudo apt-get install -y --no-install-recommends \
  cuda-toolkit-10-2
