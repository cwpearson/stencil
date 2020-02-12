#include "catch2/catch.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/packer.cuh"

TEST_CASE("packer", "[packer]") {
  Dim3 arrSz(3, 4, 5);

  LocalDomain ld(arrSz, 0);
  ld.set_radius(2);
  ld.add_data<float>();
  ld.add_data<char>();
  ld.add_data<double>();
  ld.realize();

  LocalDomain dst(arrSz, 0);
  dst.set_radius(2);
  dst.add_data<float>();
  dst.add_data<char>();
  dst.add_data<double>();
  dst.realize();

  std::vector<Message> msgs;
  msgs.push_back(Message(Dim3(-1, -1, -1), 0, 0));
  msgs.push_back(Message(Dim3(1, 1, 1), 0, 0));
  msgs.push_back(Message(Dim3(0, 1, 1), 0, 0));
  msgs.push_back(Message(Dim3(0, 0, 1), 0, 0));
  DevicePacker packer;
  packer.prepare(&ld, msgs);

  DeviceUnpacker unpacker;
  unpacker.prepare(&dst, msgs);

  REQUIRE(packer.size() == unpacker.size());

  packer.pack(0);
  CUDA_RUNTIME(cudaStreamSynchronize(0));

  CUDA_RUNTIME(cudaMemcpy(unpacker.data(), packer.data(), packer.size(),
                          cudaMemcpyDefault));

  unpacker.unpack(0);
  CUDA_RUNTIME(cudaStreamSynchronize(0));
}