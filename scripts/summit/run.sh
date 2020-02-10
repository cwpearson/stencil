#!/bin/bash

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .
#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

export OMP_NUM_THREADS=1
#jsrun -n 1 -r 1 -c 42 -g 6 -p 6 -d packed -b packed:7 js_task_info ../../build/src/weak
jsrun -n 1 -r 1 -c 1 -g 6 -a 1 -b rs js_task_info ../../build/src/weak
