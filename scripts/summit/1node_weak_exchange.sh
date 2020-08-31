#!/bin/bash
#BSUB -P csc362
#BSUB -J 1node_weak_exchange
#BSUB -o 1node_weak_exchange.o%J
#BSUB -e 1node_weak_exchange.e%J
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module load gcc
module load cuda

export X=750
export Y=750
export Z=750
export NUM_ITER=30

OUT=/gpfs/alpine/scratch/cpearson/csc362/stencil_results/1node_weak_exchange.csv

set -x

# CSV header
echo "bin,config,naive,x,y,z,s,ldx,ldy,ldz,MPI (B),Colocated (B),cudaMemcpyPeer (B),direct (B),iters,gpus,nodes,ranks,exchange (s)" > $OUT

# one rank per node, 1-6 GPUs, weak
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged                         | tee -a $OUT  
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo                  | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer           | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer --kernel  | tee -a $OUT
done

# six rank per node, 1-6 GPUs, weak
# 1 rank, 1 GPU already done
for gpus in 2 3 4 5 6; do
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged                        | tee -a $OUT 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo                 | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer          | tee -a $OUT
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer --kernel | tee -a $OUT
done


