#!/bin/bash
#BSUB -P csc362
#BSUB -J 512node_weak_exchange
#BSUB -o 512node_weak_exchange.o%J
#BSUB -e 512node_weak_exchange.e%J
#BSUB -W 1:00
#BSUB -nnodes 512
#BSUB -alloc_flags gpudefault

# should be compiled with -DSTEUP_STATS=ON -DEXCHANGE_STATS=OFF

set -eou pipefail

module load gcc
module load cuda

export X=750
export Y=750
export Z=750
export NUM_ITER=300

OUT=/gpfs/alpine/scratch/cpearson/csc362/stencil_results/512node_weak_exchange.csv

set -x

# CSV header
echo "bin,config,x,y,z,s,MPI (B),Colocated (B),cudaMemcpyPeer (B),direct (B),iters,gpus,nodes,ranks,exchange (s)" > $OUT

# 6 ranks per node, 1 GPU per rank, weak
for nodes in 1 2 4 8 16 32 64 128 256 384 512; do
  ranks=6
  gpus=6 
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged                         | tee -a $OUT  
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo                  | tee -a $OUT
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer           | tee -a $OUT
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer --kernel  | tee -a $OUT
done



