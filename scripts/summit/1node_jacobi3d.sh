#!/bin/bash
#BSUB -P csc362
#BSUB -J 1node_jacobi3d
#BSUB -o 1node_jacobi3d.o%J
#BSUB -e 1node_jacobi3d.e%J
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module load gcc
module load cuda
module load nsight-systems/2020.2.1.71

export X=750
export Y=750
export Z=750
export NUM_ITER=60

export PREFIX=/gpfs/alpine/scratch/cpearson/csc362/stencil_results
export OUT=$PREFIX/1node_jacobi3d.csv


set -x

# CSV header
echo "bin,config,ranks,gpus,x,y,z,staged (B),colo (B),peer (B),kernel (B),min (s),trimean (s)" > $OUT

# one rank per node, 1-6 GPUs, weak
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun --smpiargs="-gpu" -n 1 -a 1 -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged                         | tee -a $OUT  
  jsrun --smpiargs="-gpu" -n 1 -a 1 -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo                  | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a 1 -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo --peer           | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a 1 -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo --peer --kernel  | tee -a $OUT
done

# six rank per node, 1-6 GPUs, weak
# 1 rank, 1 GPU already done
for gpus in 2 3 4 5 6; do
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged                        | tee -a $OUT 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo                 | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo --peer          | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/jacobi3d $X $Y $Z -n $NUM_ITER --staged --colo --peer --kernel | tee -a $OUT
done


