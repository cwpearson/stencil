#!/bin/bash

set -eou pipefail

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .
#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

export DOMAIN_X=750
export DOMAIN_Y=750
export DOMAIN_Z=750
export NUM_ITER=30

export OUT_FILE=weak.out
export ERR_FILE=weak.err

# one rank per node, 6 gpus per node
for nodes in 1 2 4 8 16 32 64; do
  jsrun -n $nodes -a 1 -c 42 -g 6 -r 1 -b rs js_task_info | sort  
done

# one rank per node, 1-6 GPUs
for gpus in 1 2 3 4 5 6; do
  ranks=1
  jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info | sort  
done

# 2 ranks per node, 2/4/6 GPUs
for gpus in  2 4 6; do
  ranks=2
  jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info | sort  
done

# 3 ranks per node, 3/6 GPUs
for gpus in 3 6; do
  ranks=2
  jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info | sort  
done

gpus=4
ranks=4
jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info | sort  

gpus=5
ranks=5
jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info | sort  

gpus=6
ranks=6
jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info | sort  

