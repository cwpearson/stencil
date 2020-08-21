
#include "stencil/mpi.hpp"
#include "stencil/nvml.hpp"
#include "stencil/logging.hpp"

/* some information about the machine we're running on
   nodes indexed as 0..<N
*/
struct Machine {
  std::vector<std::string> hostnames;  // hostname of each node
  std::vector<std::vector<int>> ranks; // ranks on each node
};
  

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  const int rank = mpi::comm_rank(MPI_COMM_WORLD);
  const int size = mpi::comm_size(MPI_COMM_WORLD);

  nvmlReturn_t ret;
  nvml::lazy_init();

  unsigned int deviceCount;
  NVML(nvmlDeviceGetCount_v2(&deviceCount));
  LOG_INFO(deviceCount << " devices");

  for (unsigned int index = 0; index < deviceCount; ++index) {

    nvmlDevice_t device;
    NVML(nvmlDeviceGetHandleByIndex_v2(index, &device))


    char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE];

    NVML(nvmlDeviceGetUUID(device, uuid, sizeof(uuid)));
    LOG_INFO("  " << index << ": " << uuid);
  }

  MPI_Finalize();

  return 0;
}