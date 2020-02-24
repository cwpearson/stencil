#!/bin/bash

#BSUB -W 00:05
#BSUB -nnodes 2
#BSUB -P CSC362
#BSUB -alloc_flags gpudefault
#BSUB -o 2node_pingpong%J.out
#BSUB -e 2node_pingpong%J.err

set -eou pipefail

module load cuda

nodes=2
gpus=6
for ranks in 1 2 4 8 16 32; do
  #jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/pingpong $ranks                        
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs ../../build/src/pingpong $ranks                        
done

