#!/bin/bash

#BSUB -W 00:10
#BSUB -nnodes 4
#BSUB -P CSC362
#BSUB -alloc_flags gpudefault
#BSUB -o 4node_pingpong%J.out
#BSUB -e 4node_pingpong%J.err

set -eou pipefail

module load cuda

nodes=4
gpus=6
for ranks in 1 2 4 8 16; do
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -d packed -b packed:21 ../../build/src/pingpong --min 10 --max 25 --iters 20 $ranks
done
