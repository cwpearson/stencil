#include "stencil/machine.hpp"

#include "stencil/cuda_runtime.hpp"
#include "stencil/logging.hpp"
#include "stencil/mpi.hpp"

#include <algorithm>
#include <set>
#include <map>

typedef Machine::Distance Distance;

#if STENCIL_USE_MPI == 1
Machine Machine::build(MPI_Comm comm) {

  const int commSize = mpi::comm_size(comm);

  Machine machine;


  // Assign the same order of hostnames_ in each rank
  {
    // each rank gets host name
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(hostname, &len);

    // distribute names to all ranks
    std::vector<char> commNames(MPI_MAX_PROCESSOR_NAME * commSize);
    MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, commNames.data(), MPI_MAX_PROCESSOR_NAME, MPI_BYTE, comm);

    // uniqify host names
    std::set<std::string> uniqueNames;
    for (int i = 0; i < commSize; ++i) {
      std::string name(&commNames[i * MPI_MAX_PROCESSOR_NAME]);
      uniqueNames.insert(name);
    }
    // assign a canonical ordering of nodes
    for (const std::string &name : uniqueNames) {
      machine.hostnames_.push_back(name);
    }
    LOG_DEBUG("found " << machine.hostnames_.size() << " unique hostname(s)");
  }

  // Find the node of each rank in the machine
  {
    const std::string hostname = mpi::processor_name();
    LOG_DEBUG("my hostname: " << hostname);
    unsigned node = 0;
    {
      bool found = false;
      for (; !found && node < machine.hostnames_.size(); ++node) {
        if (machine.hostnames_[node] == hostname) {
          found = true;
          break;
        }
      }
      if (!found) {
        LOG_FATAL("error determining node id for rank");
      }
    }
    machine.nodeOfRank_.resize(commSize);
    LOG_DEBUG("my node: " << node);
    MPI_Allgather(&node, 1, MPI_INT, machine.nodeOfRank_.data(), 1, MPI_INT, comm);
  }

  // Find the rank of each GPU in the machine
  {

    // get the GPUs visible to this rank
    int cudaDevCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&cudaDevCount));
    LOG_DEBUG(cudaDevCount << " visible CUDA devices");
    std::vector<UUID> uuids(cudaDevCount);

    for (int index = 0; index < cudaDevCount; ++index) {
      cudaDeviceProp prop;
      CUDA_RUNTIME(cudaGetDeviceProperties(&prop, index));
      UUID uuid(prop.uuid.bytes);
      uuids[index] = uuid;
      LOG_DEBUG("device " << index << " uuid=" << uuid);
    }

    // convert uuids to bytes
    std::vector<char> uuidBytes;
    for (const auto &uuid : uuids) {
        uuidBytes.insert(uuidBytes.end(), uuid.bytes_, uuid.bytes_ + sizeof(uuid.bytes_));
    }

    // broadcast GPU UUIDs to all ranks
    std::vector<char> commUUIDbytes(uuidBytes.size() * commSize);
    MPI_Allgather(uuidBytes.data(), uuidBytes.size(), MPI_BYTE, commUUIDbytes.data(),
                  uuidBytes.size(), MPI_BYTE, comm);

    // convert bytes to uuids
    std::vector<UUID> commUUIDs;
    for (size_t i = 0; i < uuids.size() * commSize; ++i) {
        char buf[16];
        std::memcpy(buf, &commUUIDbytes[i*16], 16);
        UUID uuid(buf);
        commUUIDs.push_back(uuid);
    }

    {
      // uniqify UUIDs and track contributing ranks
      std::map<UUID, std::vector<int>> uuidRanks;
      for (size_t i = 0; i < commUUIDs.size(); ++i) {
        // recover UUID from bytes
        UUID &uuid = commUUIDs[i];
        // recieved gpus.size() uuids from each rank, so recover rank from uuid index
        uuidRanks[uuid].push_back(i / uuids.size());
      }

      for (const std::pair<UUID, std::vector<int>> &kv : uuidRanks) {
        machine.gpus_.push_back(GPU(kv.first, kv.second));
      }
    }
            #if 0
    #endif
  }


  return machine;
};
#endif

Distance Machine::gpu_distance(const unsigned srcId, const unsigned dstId) const {}

#if 0

  // 

  // get a communicator for ranks on this node
  MPI_Comm shmComm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmComm);

  // figure out which comm ranks are colocated on this node
  const int ranksPerNode = mpi::comm_size(shmComm);
  std::vector<int> coloRanks(ranksPerNode);
  MPI_Allgather(&commRank, 1, MPI_INT, coloRanks.data(), 1, MPI_INT, shmComm);
  std::sort(coloRanks.begin(), coloRanks.end());

#endif