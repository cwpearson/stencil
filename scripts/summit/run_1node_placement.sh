#!/bin/bash

set -eou pipefail

module load cuda

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .
#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

export NUM_ITER=30
export GPUS=6
export RANKS=1

jsrun --smpiargs="-gpu" -n 1 -a $RANKS -c 42 -g $GPUS -r 1 -b rs js_task_info ../../build/src/strong 1440 1452 700 $NUM_ITER --staged --colo --peer --kernel 
jsrun --smpiargs="-gpu" -n 1 -a $RANKS -c 42 -g $GPUS -r 1 -b rs js_task_info ../../build/src/strong 1440 1452 700 $NUM_ITER --naive --staged --colo --peer --kernel 

