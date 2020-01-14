#include "catch2/catch.hpp"

#include "stencil/device_allocator.hpp"
#include "stencil/managed_allocator.hpp"

TEMPLATE_TEST_CASE("allocator", "[cuda][template]", int, double) {
  typedef DeviceAllocator<TestType> DA;
  typedef ManagedAllocator<TestType> MA;

  SECTION("device allocator") {
  DA a;
  TestType *p = a.allocate(10);
  REQUIRE(p != nullptr);

  // expect to be able to copy to host
  TestType *host = new TestType[10];
  CUDA_RUNTIME(cudaMemcpy(host, p, 10 * sizeof(TestType), cudaMemcpyDefault));

  a.deallocate(p, 10);
  delete[] host;
  }

  SECTION("managed allocator") {
    MA a;
    TestType *p = a.allocate(10);
    REQUIRE(p != nullptr);
    for (size_t i = 0; i < 10; ++i) {
        p[i] = i;
    }
    for (size_t i = 0; i < 10; ++i) {
        REQUIRE(p[i] == i);
    }
    a.deallocate(p, 10);
    }
}
