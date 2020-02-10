#!/bin/bash

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .

export OMP_NUM_THREADS=1
jsrun -n 6 -r 6 -c 1 -g 1 -a 1 -b rs js_task_info ../../build/src/weak
