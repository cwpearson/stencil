#!/bin/bash
#BSUB -P csc362
#BSUB -J 6node_bench_mpi 
#BSUB -o 6node_bench_mpi.o%J
#BSUB -e 6node_bench_mpi.e%J
#BSUB -W 01:00
#BSUB -nnodes 6
#BSUB -alloc_flags gpudefault

set -eou pipefail

module load gcc
module load cuda

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/stencil_results
OUT=$SCRATCH/6node_bench_mpi.csv

set -x

# CSV header
echo "x,y,z,nodes,ranks-per-node,self-per-node,colo-per-node,remote-per-node,self,colo,remote,first (s),total (s)" > $OUT

# 6 ranks per node, 1 GPU per rank
for nodes in 1 2 3 4 5 6; do
  for ranks in 1 2 3 6; do
    for X in 15 30 60 100 200 400; do
      gpus=$ranks
      let nrs=$nodes*$ranks
      jsrun --smpiargs="-gpu" -n $nodes -a $ranks -g $ranks -c 42 -r 1 -b packed:7 ../../build/bin/bench-mpi $X $X $X | tee -a $OUT
    done
  done
done
