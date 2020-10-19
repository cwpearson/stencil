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
module load cuda/11.0.3

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/stencil_results
OUT=$SCRATCH/6node_bench_mpi.csv

set -x

# CSV header
echo "x,y,z,nodes,ranks-per-node,self-per-node,colo-per-node,remote-per-node,self,colo,remote,first (s),total (s)" > $OUT

# some strong scaling
# 6 ranks per node, 1 GPU per rank
for nodes in 1 2 3 4 5 6; do
  for ranks in 1 2 3 6; do
      gpus=$ranks
      let nrs=$nodes*$ranks
      if   [ $nrs == 1  ]; then X=750
      elif [ $nrs == 2  ]; then X=945
      elif [ $nrs == 3  ]; then X=1082
      elif [ $nrs == 4  ]; then X=1191
      elif [ $nrs == 5  ]; then X=1282
      elif [ $nrs == 6  ]; then X=1363
      elif [ $nrs == 8  ]; then X=1500
      elif [ $nrs == 9  ]; then X=1560
      elif [ $nrs == 10 ]; then X=1616
      elif [ $nrs == 12 ]; then X=1717
      elif [ $nrs == 15 ]; then X=1850
      elif [ $nrs == 18 ]; then X=1966
      elif [ $nrs == 24 ]; then X=2163
      elif [ $nrs == 30 ]; then X=2330
      elif [ $nrs == 36 ]; then X=2476
      else unset X
      fi
      jsrun --smpiargs="-gpu" -n $nodes -a $ranks -g $ranks -c 42 -r 1 -b packed:7 ../../build/bin/bench-mpi $X $X $X | tee -a $OUT
  done
done

# some weak scaling
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


