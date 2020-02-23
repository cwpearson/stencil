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

export OUT_FILE=weak.out
export ERR_FILE=weak.err

gpus=6
ranks=6
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware
exit 1

# one rank, 6 gpus
ranks=2
gpus=4
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware

exit 1

