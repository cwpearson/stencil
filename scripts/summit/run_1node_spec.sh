#!/bin/bash

#BSUB -W 01:00
#BSUB -nnodes 1
#BSUB -P CSC362
#BSUB -alloc_flags gpudefault
#BSUB -o 1node_spec_%J.out
#BSUB -e 1node_spec_%J.err

set -eou pipefail

module load cuda

export X=750
export Y=750
export Z=750
export NUM_ITER=30

for ranks in 1 6; do
  gpus=6
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo --peer
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo --peer --kernel
done

ranks=2
gpus=6
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged                             
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo                 
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer          
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo 
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo --peer
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo --peer --kernel

for ranks in 1 6; do
  gpus=6
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware --colo 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware --colo --peer
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware --colo --peer --kernel
done

ranks=2
gpus=6
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged                             
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo                 
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo 
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo --peer
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/weak $X $Y $Z $NUM_ITER --cuda-aware --colo --peer --kernel

