#!/bin/bash

export OMP_NUM_THREADS=1
jsrun -n 1 -r 1 -c 1 -g 6 -a 6 -b rs js_task_info ../../build/src/weak
