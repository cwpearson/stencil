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

# one rank per node, 1-6 GPUs, weak
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

# six rank per node, 1-6 GPUs, weak
for gpus in 1 2 3 4 5 6; do
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

# one rank per node, 1-6 GPUs, strong
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

# six rank per node, 1-6 GPUs, strong
for gpus in 1 2 3 4 5 6; do
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

