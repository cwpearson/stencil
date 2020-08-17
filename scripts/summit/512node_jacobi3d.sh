#!/bin/bash
#BSUB -P csc362
#BSUB -J 512node_jacobi3d
#BSUB -o 512node_jacobi3d.o%J
#BSUB -e 512node_jacobi3d.e%J
#BSUB -W 1:00
#BSUB -nnodes 512
#BSUB -alloc_flags gpudefault

set -eou pipefail

module load gcc
module load cuda

export X=750
export Y=750
export Z=750
export NUM_ITER=300

export PREFIX=/gpfs/alpine/scratch/cpearson/csc362/stencil_results
export OUT=$PREFIX/512node_jacobi3d.csv


set -x

# CSV header
echo "bin,config,ranks,gpus,x,y,z,staged (B),colo (B),peer (B),kernel (B),min (s),trimean (s)" > $OUT

# six rank per node, 1-6 GPUs, weak
# 1 rank, 1 GPU already done
for nodes in 1 2 4 8 16 32 64 128 256 384 512; do
  gpus=6
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 7 -g $gpus -r 6 -b rs ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged                        | tee -a $OUT 
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 7 -g $gpus -r 6 -b rs ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo                 | tee -a $OUT
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 7 -g $gpus -r 6 -b rs ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo --peer          | tee -a $OUT
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 7 -g $gpus -r 6 -b rs ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo --peer --kernel | tee -a $OUT
done


