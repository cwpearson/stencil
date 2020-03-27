#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>

#include "argparse/argparse.hpp"
#include "stencil/cuda_runtime.hpp"
#include "stencil/mat2d.hpp"

__global__ void clock_block(clock_t *d, clock_t clock_count) {
  clock_t start_clock = clock64();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock64() - start_clock;
  }
  if (d) {
    *d = clock_offset;
  }
}

int main(int argc, char **argv) {

  (void)argc;
  (void)argv;

  const int64_t K = 1024;
  const int64_t M = K * K;
  const int64_t G = K * K * K;

  const int nGpus = 4;
  const int nMeasures = 10;
  const int nIters = 50;

  const double period = 0.004;

  // enable peer access
  for (size_t src = 0; src < nGpus; ++src) {
    for (size_t dst = 0; dst < nGpus; ++dst) {
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

  cudaEvent_t latch;
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaEventCreate(&latch));
  Mat2D<cudaStream_t> streams(nGpus, nGpus, nullptr);
  Mat2D<cudaEvent_t> startEvents(nGpus, nGpus, nullptr);
  Mat2D<cudaEvent_t> stopEvents(nGpus, nGpus, nullptr);
  Mat2D<void *> srcBufs(nGpus, nGpus, nullptr);
  Mat2D<void *> dstBufs(nGpus, nGpus, nullptr);
  Mat2D<int64_t> bufSizes(nGpus, nGpus, 0);
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      CUDA_RUNTIME(cudaSetDevice(i));
      CUDA_RUNTIME(cudaStreamCreate(&streams.at(i, j)));
      CUDA_RUNTIME(cudaEventCreate(&startEvents.at(i, j)));
      CUDA_RUNTIME(cudaEventCreate(&stopEvents.at(i, j)));
    }
  }

  // bytes to send
  Mat2D<int64_t> x(nGpus, nGpus, 8 * M);

  // update to x
  Mat2D<int64_t> dx(nGpus, nGpus, 0);

  // time taken for each transfer
  Mat2D<double> y(nGpus, nGpus, 0); // average

  int64_t cycles = 1e5;
  for (int iter = 0; iter < nIters; ++iter) {

    std::cout << "x\n";
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        printf("%ld ", x.at(i, j) / 1024 / 1024);
      }
      std::cout << "\n";
    }

    // set up buffers,
    for (int i = 0; i < nGpus; ++i) {
      for (int j = 0; j < nGpus; ++j) {
        if (bufSizes.at(i, j) < x.at(i, j)) {
          bufSizes.at(i, j) = x.at(i, j) * 2;
          std::cout << "increasing buffer size to " << bufSizes.at(i, j)
                    << "\n";
          CUDA_RUNTIME(cudaFree(srcBufs.at(i, j)));
          CUDA_RUNTIME(cudaFree(dstBufs.at(i, j)));
          CUDA_RUNTIME(cudaSetDevice(i));
          CUDA_RUNTIME(cudaMalloc(&srcBufs.at(i, j), bufSizes.at(i, j)));
          CUDA_RUNTIME(cudaSetDevice(j));
          CUDA_RUNTIME(cudaMalloc(&dstBufs.at(i, j), bufSizes.at(i, j)));
        }
      }
    }

    y = Mat2D<double>(nGpus, nGpus, 0);
    for (int n = 0; n < nMeasures; ++n) {

      // set up copies and wait on latch event.
      // grow the wait time before the latch event until it covers the copy
      // setup time
      while (true) {
        CUDA_RUNTIME(cudaSetDevice(0));
        CUDA_RUNTIME(cudaEventDestroy(latch));
        CUDA_RUNTIME(cudaEventCreate(&latch));
        clock_block<<<1, 1, 0, 0>>>(nullptr, cycles);
        CUDA_RUNTIME(cudaEventRecord(latch, 0));
        for (int i = 0; i < nGpus; ++i) {
          for (int j = 0; j < nGpus; ++j) {
            CUDA_RUNTIME(cudaSetDevice(i));
            CUDA_RUNTIME(cudaStreamWaitEvent(streams.at(i, j), latch, 0));
            CUDA_RUNTIME(
                cudaEventRecord(startEvents.at(i, j), streams.at(i, j)));
            CUDA_RUNTIME(cudaMemcpyPeerAsync(dstBufs.at(i, j), j,
                                             srcBufs.at(i, j), i, x.at(i, j),
                                             streams.at(i, j)));
            CUDA_RUNTIME(
                cudaEventRecord(stopEvents.at(i, j), streams.at(i, j)));
          }
        }

        // make sure kernel was long enough to cover setup of copies
        cudaError_t err = cudaEventQuery(latch);
        if (cudaErrorNotReady == err) {
          cudaGetLastError(); // clear error
          break;
        } else if (cudaSuccess == err) {
          cycles *= 2;
          std::cout << "increasing cycle count to " << cycles << "\n";
          // wait for copies to finish
          for (size_t i = 0; i < nGpus; ++i) {
            for (size_t j = 0; j < nGpus; ++j) {
              CUDA_RUNTIME(cudaStreamSynchronize(streams.at(i, j)));
            }
          }
        } else {
          CUDA_RUNTIME(err);
        }
      }

      // wait for copies to finish
      for (size_t i = 0; i < nGpus; ++i) {
        for (size_t j = 0; j < nGpus; ++j) {
          CUDA_RUNTIME(cudaStreamSynchronize(streams.at(i, j)));
        }
      }

      // get times
      for (size_t i = 0; i < nGpus; ++i) {
        for (size_t j = 0; j < nGpus; ++j) {
          float ms;
          CUDA_RUNTIME(cudaEventElapsedTime(&ms, startEvents.at(i, j),
                                            stopEvents.at(i, j)));
          y.at(i, j) += ms / 1000.0;
        }
      }
    }
    y /= nMeasures;

    std::cout << "y\n";
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        printf("%.4e ", y.at(i, j));
      }
      std::cout << "\n";
    }

    // check if times are all close to target
    bool done = true;
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        if (std::abs(y.at(i, j) - period) > period / 100.0) {
          done = false;
        }
      }
    }
    if (done) {
      break;
    }

    // take a step in moving each time towards the target time
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        double dydx = y.at(i, j) / x.at(i, j);
        dx.at(i, j) = (period - y.at(i, j)) / dydx;
      }
    }

    std::cout << "dx\n";
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        printf("%ld ", dx.at(i, j));
      }
      std::cout << "\n";
    }

    // take a step in moving each time towards the target time
    double gamma = 0.2;
    for (size_t i = 0; i < nGpus; ++i) {
      for (size_t j = 0; j < nGpus; ++j) {
        x.at(i, j) += gamma * dx.at(i, j);
      }
    }
  }

  // print final communication sizes
  for (size_t i = 0; i < nGpus; ++i) {
    for (size_t j = 0; j < nGpus; ++j) {
      std::cout << x.at(i, j) / 1024 / 1024 << " ";
    }
    std::cout << "\n";
  }
}
