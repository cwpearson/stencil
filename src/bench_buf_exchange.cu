#include <cassert>
#include <chrono>
#include <cmath>

#include "stencil/argparse.hpp"
#include "stencil/cuda_runtime.hpp"
#include "stencil/mat2d.hpp"

double (*ExchangeFunc)(const Mat2D<int64_t> &comm, const int nIters);

double exchange_cuda_memcpy_peer(const Mat2D<int64_t> &comm, const int nIters) {

  // enable peer access
  for (size_t src = 0; src < comm.shape().y; ++src) {
    for (size_t dst = 0; dst < comm.shape().x; ++dst) {
      if (src == dst) {
        continue;
      } else {
        int canAccess;
        CUDA_RUNTIME(cudaDeviceCanAccessPeer(&canAccess, src, dst));
        if (canAccess) {
          CUDA_RUNTIME(cudaSetDevice(src));
          cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0 /*flags*/);
          if (cudaSuccess == err || cudaErrorPeerAccessAlreadyEnabled == err) {
            cudaGetLastError(); // clear the error
          } else if (cudaErrorInvalidDevice == err) {
            cudaGetLastError(); // clear the error
          } else {
            CUDA_RUNTIME(err);
          }
        } else {
        }
      }
      CUDA_RUNTIME(cudaGetLastError());
    }
  }

  size_t nGpus = std::max(comm.shape().x, comm.shape().y);
  Mat2D<cudaStream_t> streams(nGpus, nGpus, nullptr);
  Mat2D<cudaEvent_t> startEvents(nGpus, nGpus, nullptr);
  Mat2D<cudaEvent_t> stopEvents(nGpus, nGpus, nullptr);
  Mat2D<void *> srcBufs(nGpus, nGpus, nullptr);
  Mat2D<void *> dstBufs(nGpus, nGpus, nullptr);
  Mat2D<double> times(nGpus, nGpus, 0);
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      CUDA_RUNTIME(cudaSetDevice(i));
      CUDA_RUNTIME(cudaStreamCreate(&streams.at(i, j)));
      CUDA_RUNTIME(cudaEventCreate(&startEvents.at(i, j)));
      CUDA_RUNTIME(cudaEventCreate(&stopEvents.at(i, j)));
      CUDA_RUNTIME(cudaMalloc(&srcBufs.at(i, j), comm.at(i, j)));

      CUDA_RUNTIME(cudaSetDevice(j));
      CUDA_RUNTIME(cudaMalloc(&dstBufs.at(i, j), comm.at(i, j)));
    }
  }

  std::chrono::duration<double> elapsed = std::chrono::seconds(0);

  for (int n = 0; n < nIters; ++n) {

    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaEventRecord(startEvents.at(i, j), streams.at(i, j)));
        CUDA_RUNTIME(cudaMemcpyPeerAsync(dstBufs.at(i, j), j, srcBufs.at(i, j),
                                         i, comm.at(i, j), streams.at(i, j)));
        CUDA_RUNTIME(cudaEventRecord(stopEvents.at(i, j), streams.at(i, j)));
      }
    }

    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        CUDA_RUNTIME(cudaStreamSynchronize(streams.at(i, j)));
      }
    }
    elapsed += std::chrono::system_clock::now() - start;

    // get time for each transfer
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        float ms;
        CUDA_RUNTIME(cudaEventElapsedTime(&ms, startEvents.at(i, j),
                                          stopEvents.at(i, j)));
        times.at(i, j) += ms / 1000.0;
      }
    }
  }

  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      std::cout << comm.at(i, j) / times.at(i, j) << " ";
    }
    std::cout << "\n";
  }

  return elapsed.count() / double(nIters);
}

int main(int argc, char **argv) {

  // clang-format off
  Mat2D<int64_t> m {
    {10, 1},
    {1, 10},
  };
  // clang-format on

  double time = exchange_cuda_memcpy_peer(m, 30);

  std::cout << time << "\n";
}