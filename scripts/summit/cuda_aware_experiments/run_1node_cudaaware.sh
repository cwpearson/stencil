#!/bin/bash

set -eou pipefail

module load cuda

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .
#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

export X=750
export Y=750
export Z=750
export NUM_ITER=30

export SCRATCH=/gpfs/alpine/scratch/merth/csc362

# one rank per node, 6 gpus
gpus=6
ranks=1
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info nvprof -o $SCRATCH/staged_%p.nvvp     ../../../build/src/weak $X $Y $Z $NUM_ITER --staged
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info nvprof -o $SCRATCH/cuda-aware_%p.nvvp ../../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info nvprof -o $SCRATCH/peer_%p.nvvp       ../../../build/src/weak $X $Y $Z $NUM_ITER --peer


