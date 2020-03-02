#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>

#include "stencil/argparse.hpp"
#include "stencil/cuda_runtime.hpp"
#include "stencil/mat2d.hpp"

double (*ExchangeFunc)(const Mat2D<int64_t> &comm, const int nIters);

void exchange_cuda_memcpy_peer(double *rtotal, Mat2D<double> *rtimes, const Mat2D<int64_t> &comm, const int nIters) {

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

  // normalize times by nIters
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      times.at(i, j) /= nIters;
    }
  }

  /*
  std::cout << "bw\n";
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      printf("%.4e ", comm.at(i, j) / times.at(i, j));
    }
    std::cout << "\n";
  }
  std::cout << "time\n";
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      printf("%.4e ", times.at(i, j));
    }
    std::cout << "\n";
  }
*/
  // free stuff
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      CUDA_RUNTIME(cudaStreamDestroy(streams.at(i, j)));
      CUDA_RUNTIME(cudaEventDestroy(startEvents.at(i, j)));
      CUDA_RUNTIME(cudaEventDestroy(stopEvents.at(i, j)));
      CUDA_RUNTIME(cudaFree(srcBufs.at(i, j)));
      CUDA_RUNTIME(cudaFree(dstBufs.at(i, j)));
    }
  }

  if (rtotal) *rtotal = elapsed.count() / double(nIters);
  if (rtimes) *rtimes = times;
}

int main(int argc, char **argv) {

  const int64_t K = 1024;
  const int64_t M = K * K;
  const int64_t G = K * K * K;

  Mat2D<int64_t> allToAll1G {
    {G, G, G, G},
    {G, G, G, G},
    {G, G, G, G},
    {G, G, G, G},
  };
  Mat2D<int64_t> allToAll8M {
    {8*M, 8*M, 8*M, 8*M},
    {8*M, 8*M, 8*M, 8*M},
    {8*M, 8*M, 8*M, 8*M},
    {8*M, 8*M, 8*M, 8*M},
  };
  Mat2D<int64_t> stencil512x256x512 {
    {12*M, 12*M, 6*M, 30*K},
    {12*M, 12*M, 30*K, 8*M},
    {6*M, 30*K, 12*M, 12*M},
    {30*K, 6*M, 12*M, 12*M},
  };
  Mat2D<int64_t> local1G {
    {G, G, 0, 0},
    {G, G, 0, 0},
    {0, 0, G, G},
    {0, 0, G, G},
  };
  Mat2D<int64_t> local1Gremote100M {
    {G, G, 100*M, 100*M},
    {G, G, 100*M, 100*M},
    {100*M, 100*M, G, G},
    {100*M, 100*M, G, G},
  };
  // clang-format on

  double time;

  std::cout << "stencil\n";
  exchange_cuda_memcpy_peer(&time, nullptr, stencil512x256x512, 30);
  std::cout << time << "\n";
  std::cout << "All-to-all 8MiB\n";
  exchange_cuda_memcpy_peer(&time, nullptr, allToAll8M, 30);
  std::cout << time << "\n";
  std::cout << "All-to-all 1GiB\n";
  exchange_cuda_memcpy_peer(&time, nullptr, allToAll1G, 30);
  std::cout << time << "\n";
  std::cout << "Local 1GiB\n";
  exchange_cuda_memcpy_peer(&time, nullptr, local1G, 30);
  std::cout << time << "\n";
  std::cout << "Local 1GiB Remote 100M\n";
  exchange_cuda_memcpy_peer(&time, nullptr, local1Gremote100M, 30);
  std::cout << time << "\n";
}
