#include "catch2/catch.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/packer.cuh"
#include "stencil/rcstream.hpp"

TEST_CASE("packer", "[packer]") {
  Dim3 arrSz(3, 4, 5);
  Dim3 origin(0, 0, 0);

  LocalDomain ld(arrSz, origin, 0);
  ld.set_radius(2);
  ld.add_data<float>();
  ld.add_data<char>();
  ld.add_data<double>();
  ld.realize();

  LocalDomain dst(arrSz, origin, 0);
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

  RcStream stream(0);

  DevicePacker packer(stream);
  packer.prepare(&ld, msgs);

  DeviceUnpacker unpacker(stream);
  unpacker.prepare(&dst, msgs);

  REQUIRE(packer.size() == unpacker.size());

  packer.pack();
  CUDA_RUNTIME(cudaStreamSynchronize(0));

  CUDA_RUNTIME(cudaMemcpy(unpacker.data(), packer.data(), packer.size(),
                          cudaMemcpyDefault));

  unpacker.unpack();
  CUDA_RUNTIME(cudaStreamSynchronize(0));
}

TEST_CASE("packer multi-radius", "[packer]") {
  Dim3 arrSz(3, 4, 5);
  Dim3 origin(0, 0, 0);

  LocalDomain src(arrSz, origin, 0);
  Radius radius = Radius::constant(0);
  radius.dir(1, 0, 0) = 2;
  radius.dir(-1, 0, 0) = 1;
  src.set_radius(radius);
  src.add_data<float>();
  src.add_data<char>();
  src.add_data<double>();
  src.realize();

  LocalDomain dst(arrSz, origin, 0);
  dst.set_radius(radius);
  dst.add_data<float>();
  dst.add_data<char>();
  dst.add_data<double>();
  dst.realize();

  RcStream stream(0);

  INFO("test expected size");
  // +x radius is 2, -x radius is 1
  // send in +x means sending 1x4x5 elements
  // 20 floats = 80
  // 20 char = 100
  // align to double = 104
  // 20 double = 264
  {
    std::vector<Message> msgs;
    msgs.push_back(Message(Dim3(1, 0, 0), 0, 0));
    DevicePacker packer(stream);
    packer.prepare(&src, msgs);

    DeviceUnpacker unpacker(stream);
    unpacker.prepare(&dst, msgs);

    REQUIRE(packer.size() == 264);
    REQUIRE(unpacker.size() == 264);
  }

  INFO("packer and unpacker should match");
  {
    std::vector<Message> msgs;
    msgs.push_back(Message(Dim3(-1, 0,0), 0, 0));
    msgs.push_back(Message(Dim3(1, 0,0), 0, 0));
    DevicePacker packer(stream);
    packer.prepare(&src, msgs);

    DeviceUnpacker unpacker(stream);
    unpacker.prepare(&dst, msgs);

    REQUIRE(packer.size() == unpacker.size());

    packer.pack();
    CUDA_RUNTIME(cudaStreamSynchronize(0));

    CUDA_RUNTIME(cudaMemcpy(unpacker.data(), packer.data(), packer.size(),
                            cudaMemcpyDefault));

    unpacker.unpack();
    CUDA_RUNTIME(cudaStreamSynchronize(0));
  }
}