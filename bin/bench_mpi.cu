#include "stencil/stencil.hpp"

#include "statistics.hpp"

#include <mpi.h>

#include <iostream>
#include <map>
#include <numeric>
#include <vector>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  const int worldRank = mpi::world_rank();
  const int worldSize = mpi::world_size();

  int numNodes;
  {
    MpiTopology topo(MPI_COMM_WORLD);
    numNodes = worldSize / topo.colocated_size();
  }

  // discover which ranks are on which nodes
  Machine machine = Machine::build(MPI_COMM_WORLD);

  // stencil config
  int nIters = 1000;
  int x = 60;
  int y = 60;
  int z = 60;
  Radius radius = Radius::constant(1);
  int gpusPerRank = 1;

  if (argc > 1) {
    x = std::atoi(argv[1]);
    y = x;
    z = x;
  }

  // Determine stencil layout
  std::vector<int> gpus(gpusPerRank);
  std::iota(gpus.begin(), gpus.end(), 0);
  DistributedDomain *dd = new DistributedDomain(x, y, z);
  dd->set_radius(radius);
  dd->add_data<float>();
  dd->set_gpus(gpus);
  dd->do_placement();

  assert(gpus.size() == 1); // or we need to track this per GPU
  std::vector<std::vector<unsigned char>> sendBufs(worldSize);
  std::vector<std::vector<unsigned char>> recvBufs(worldSize);
  std::vector<MPI_Request> sendReqs(worldSize);
  std::vector<MPI_Request> recvReqs(worldSize);

  int selfMessages = 0;
  int coloMessages = 0;
  int nodeMessages = 0;

  const Dim3 myIdx = dd->get_placement()->get_idx(worldRank, 0);

  // figure out all my neighbors
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {

        const Dim3 dir = Dim3(dx, dy, dz);
        if (dir == Dim3(0, 0, 0)) {
          continue;
        }

        const Topology::OptionalNeighbor srcNbr = dd->get_topology().get_neighbor(myIdx, dir);
        const Topology::OptionalNeighbor dstNbr = dd->get_topology().get_neighbor(myIdx, dir * -1);

        if (srcNbr.exists) {
          int srcRank = dd->get_placement()->get_rank(srcNbr.index);
          // the size of the send is equal to our halo
          const Dim3 sz = dd->get_placement()->subdomain_size(myIdx);
          const Dim3 ext = LocalDomain::halo_extent(dir * -1, sz, radius);
          const size_t nBytes = ext.flatten() * sizeof(float);
          recvBufs[srcRank].resize(recvBufs[srcRank].size() + nBytes);
        }

        if (dstNbr.exists) {
          int dstRank = dd->get_placement()->get_rank(srcNbr.index);
          // the size of the send is equal to the neighbor domains halo
          const Dim3 sz = dd->get_placement()->subdomain_size(dstNbr.index);
          const Dim3 ext = LocalDomain::halo_extent(dir * -1, sz, radius);
          const size_t nBytes = ext.flatten() * sizeof(float);
          sendBufs[dstRank].resize(sendBufs[dstRank].size() + nBytes);

          if (dstRank == worldRank) {
            selfMessages++;
          } else if (machine.node_of_rank(worldRank) == machine.node_of_rank(dstRank)) {
            coloMessages++;
          } else {
            nodeMessages++;
          }
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &selfMessages, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &coloMessages, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nodeMessages, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  Statistics statsFirst, statsTotal;

  for (int i = 0; i < nIters; ++i) {

    MPI_Barrier(MPI_COMM_WORLD);

    const double timeStart = MPI_Wtime();
    // do the MPI sends
    for (int dstRank = 0; dstRank < int(sendBufs.size()); ++dstRank) {
      std::vector<unsigned char> &buf = sendBufs[dstRank];
      MPI_Request &req = sendReqs[dstRank];
      MPI_Isend(buf.data(), buf.size(), MPI_BYTE, dstRank, 0, MPI_COMM_WORLD, &req);
      // LOG_DEBUG("Isend   to " << dstRank << " " << buf.size() << "B");
    }
    const double timeIsend = MPI_Wtime();

    // do the MPI recvs
    for (int srcRank = 0; srcRank < int(recvBufs.size()); ++srcRank) {
      std::vector<unsigned char> &buf = sendBufs[srcRank];
      MPI_Request &req = recvReqs[srcRank];
      MPI_Irecv(buf.data(), buf.size(), MPI_BYTE, srcRank, 0, MPI_COMM_WORLD, &req);
      // LOG_DEBUG("Irecv from " << srcRank << " " << buf.size() << "B");
    }
    const double timeIrecv = MPI_Wtime();

    // poll recvs until we find the first that's done
    double timeFirst = 0;
    for (auto &req : recvReqs) {
      int done;
      MPI_Test(&req, &done, MPI_STATUS_IGNORE);
      if (done) {
        timeFirst = MPI_Wtime();
        break;
      }
    }

    // wait for all
    for (auto &req : sendReqs) {
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    for (auto &req : recvReqs) {
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    const double timeDone = MPI_Wtime();

    double durIsend = timeIsend - timeStart;
    double durIrecv = timeIrecv - timeIsend;
    double durFirst = timeFirst - timeIsend; // between last send and first recv
    double durWait = timeDone - timeIrecv;
    double durTotal = timeDone - timeStart;

    MPI_Allreduce(MPI_IN_PLACE, &durIsend, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &durIrecv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &durWait, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &durTotal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // avg time between last send and first recv
    MPI_Allreduce(MPI_IN_PLACE, &durFirst, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    durFirst /= worldSize;

    statsTotal.insert(durTotal);
    statsFirst.insert(durFirst);
  }
  delete dd;

  if (mpi::world_rank() == 0) {
    printf("%d,%d,%d", x, y, z);
    printf(",%d", numNodes);
    printf(",%d,%d,%d", selfMessages, coloMessages, nodeMessages);
    printf(",%e,%e", statsFirst.trimean(), statsTotal.trimean());
    std::cout << "\n";
  }

  MPI_Finalize();
}
