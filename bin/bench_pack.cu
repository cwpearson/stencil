#include <chrono>
#include <sstream>

#include <nvToolsExt.h>

#include "argparse/argparse.hpp"
#include "stencil/stencil.hpp"

void bench(size_t *rBytes, double *rPackTime, double *rUnpackTime,
           const Dim3 sz, const Dim3 dir, const int nIters) {


  std::stringstream ss;
  float ms;

  const Dim3 origin(0,0,0);

  LocalDomain ld(sz, origin, 0);
  ld.add_data<float>();
  ld.set_radius(3);

  ld.realize();

  std::vector<Message> msgs;
  msgs.push_back(Message(dir, 0, 0));

  RcStream stream(0);

  DevicePacker packer(stream);
  DeviceUnpacker unpacker(stream);
  packer.prepare(&ld, msgs);
  unpacker.prepare(&ld, msgs);

  if (rBytes)
    *rBytes = packer.size();


  cudaEvent_t startEvent, stopEvent;
  CUDA_RUNTIME(cudaEventCreate(&startEvent));
  CUDA_RUNTIME(cudaEventCreate(&stopEvent));

  ss << dir << " pack";
  nvtxRangePush(ss.str().c_str());
  CUDA_RUNTIME(cudaEventRecord(startEvent, stream));
  for (int n = 0; n < nIters; ++n) {
    packer.pack();
  }
  CUDA_RUNTIME(cudaEventRecord(stopEvent, stream));
  CUDA_RUNTIME(cudaStreamSynchronize(stream));
  nvtxRangePop();
  CUDA_RUNTIME(cudaEventElapsedTime(&ms, startEvent, stopEvent));

  if (rPackTime)
    *rPackTime = double(ms) / 1000 / nIters;

  ss << dir << " unpack";
  nvtxRangePush(ss.str().c_str());
  CUDA_RUNTIME(cudaEventRecord(startEvent, stream));
  for (int n = 0; n < nIters; ++n) {
    unpacker.unpack();
  }
  CUDA_RUNTIME(cudaEventRecord(stopEvent, stream));
  CUDA_RUNTIME(cudaStreamSynchronize(stream));
  nvtxRangePop();
  CUDA_RUNTIME(cudaEventElapsedTime(&ms, startEvent, stopEvent));

  if (rUnpackTime)
    *rUnpackTime = double(ms) / 1000 / nIters;

  CUDA_RUNTIME(cudaEventDestroy(startEvent));
  CUDA_RUNTIME(cudaEventDestroy(stopEvent));
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int nIters = 30;

  argparse::Parser p;
  p.add_option(nIters, "--iters");
  if (!p.parse(argc, argv)) {
    std::cout << p.help();
    exit(EXIT_FAILURE);
  }

  Dim3 ext, dir;
  double packTime, unpackTime;
  size_t bytes;

  ext = Dim3(512, 512, 512);
  dir = Dim3(1, 0, 0);
  bench(&bytes, &packTime, &unpackTime, ext, dir, nIters);
  std::cout << ext << " " << dir << " " << bytes << " " << packTime << " "
            << unpackTime << "\n";

  ext = Dim3(512, 512, 512);
  dir = Dim3(0, 1, 0);
  bench(&bytes, &packTime, &unpackTime, ext, dir, nIters);
  std::cout << ext << " " << dir << " " << bytes << " " << packTime << " "
            << unpackTime << "\n";

  ext = Dim3(512, 512, 512);
  dir = Dim3(0, 0, 1);
  bench(&bytes, &packTime, &unpackTime, ext, dir, nIters);
  std::cout << ext << " " << dir << " " << bytes << " " << packTime << " "
            << unpackTime << "\n";

  return 0;
}
