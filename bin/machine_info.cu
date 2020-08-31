#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"
#include "stencil/machine.hpp"
#include "stencil/mpi.hpp"
#include "stencil/nvml.hpp"

#include <vector>

/* the NVML UUID looks like an array of chars like ascii "GPU-d1810711-f3ef-4529-8662-52609f808deb"
the CUDA device prop UUID is an array of char with bytes that look like d1810711f3ef4529866252609f808deb when printed as
hex return a vector of unsigned char from the nvml ascii string
*/
template <unsigned N> std::vector<unsigned char> parse_nvml_uuid(char uuid[N]) {
  std::vector<unsigned char> ret;
  LOG_DEBUG("parsing NVML uuid " << uuid);

  // scan through characters and start after any non-hex ones
  unsigned start = 0;
  for (unsigned i = 0; i < N && 0 != uuid[i]; ++i) {
    char c = uuid[i];
    if ((c == '-') ||                                                                 // this is part of the UUID string
        (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || (c >= '0' && c <= '9')) { // c is a UUID char
    } else {
      // c is not a UUID char, start parsing afterwards
      start = i + 1;
    }
  }

  for (unsigned i = start; i < N && 0 != uuid[i];) {

    if ('-' == uuid[i]) {
      ++i;
      continue;
    }

    unsigned char val;
    int filled = sscanf(&uuid[i], "%2hhx", &val);
    if (1 != filled) {
      LOG_FATAL("NVML UUID parse error");
    }
    ret.push_back(val);
    i += 2;
  }

  return ret;
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  Machine machine = Machine::build(MPI_COMM_WORLD);

  if (0 == mpi::world_rank()) {
    LOG_INFO("nodes: " << machine.num_nodes());
    for (int rank = 0; rank < machine.num_ranks(); ++rank) {
      LOG_INFO("rank " << rank << ": node " << machine.node_of_rank(rank));
    }
    for (int gpu = 0; gpu < machine.num_gpus(); ++gpu) {
      std::string s;
      s += "gpu ";
      s += std::to_string(gpu);
      s += "/";
      s += std::string(machine.gpu(gpu).uuid());
      s += ": ranks [";
      for (auto r : machine.gpu(gpu).ranks()) {
        s += std::to_string(r) + " ";
      }
      s += ']';
      LOG_INFO(s);
    }
  }
  #if 0
  #endif
#if 0
  const int rank = mpi::comm_rank(MPI_COMM_WORLD);
  const int size = mpi::comm_size(MPI_COMM_WORLD);

  nvmlReturn_t ret;
  nvml::lazy_init();

  unsigned int deviceCount;
  NVML(nvmlDeviceGetCount_v2(&deviceCount));
  LOG_INFO(deviceCount << " NVML devices");

  for (unsigned int index = 0; index < deviceCount; ++index) {

    nvmlDevice_t device;
    NVML(nvmlDeviceGetHandleByIndex_v2(index, &device))

#if NVML_API_VERSION >= 11
    char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE];
#else
    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
#endif

    NVML(nvmlDeviceGetUUID(device, uuid, sizeof(uuid)));
    LOG_INFO("  NVML " << index << ": " << uuid);
    std::vector<unsigned char> rawUuid = parse_nvml_uuid<sizeof(uuid)>(uuid);
    LOG_DEBUG("  NVML " << index << ": " << rawUuid.size());
  }

  int cudaDevCount;
  CUDA_RUNTIME(cudaGetDeviceCount(&cudaDevCount));
  LOG_INFO(cudaDevCount << " CUDA devices");
  for (int index = 0; index < cudaDevCount; ++index) {
    cudaDeviceProp prop;
    CUDA_RUNTIME(cudaGetDeviceProperties(&prop, index));

    // hex str of uuid. 2 hex chars per byte
    char uuidStr[sizeof(prop.uuid.bytes) * 2 + 1] = {};
    
    for (unsigned i = 0; i < sizeof(prop.uuid.bytes); ++i) {
      snprintf(&uuidStr[2*i], 3/*max 2 bytes,+1 NULL*/, "%02x", prop.uuid.bytes[i]);
    }

    LOG_INFO(" CUDA " << index << ": " << uuidStr);
  }
#endif
  MPI_Finalize();

  return 0;
}
