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
for ranks in 1 2 4 8 16 20; do
  #jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/pingpong $ranks                        
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -d packed -b packed:1 ../../build/src/pingpong $ranks                        
done

