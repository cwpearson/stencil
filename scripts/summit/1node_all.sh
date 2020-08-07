#!/bin/bash

set -eou pipefail

module load gcc
module load cuda

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .
#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

export X=750
export Y=750
export Z=750
export NUM_ITER=30

set -x

# CSV header
echo bin,config,x,y,z,s,bytes,iters,gpus,nodes,ranks,mpi_topo,node_gpus,peer_en,placement,realize,plan,create,exchange,swap

# one rank per node, 1-6 GPUs, weak
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

# six rank per node, 1-6 GPUs, weak
# 1 rank, 1 GPU already done
for gpus in 2 3 4 5 6; do
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/weak $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

# one rank per node, 1-6 GPUs, strong
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

# six rank per node, 1-6 GPUs, strong
# 1 rank, 1 gpu already done
for gpus in 2 3 4 5 6; do
  ranks=$gpus
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/bin/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
done

