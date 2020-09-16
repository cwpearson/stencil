#!/bin/bash
#BSUB -P csc362
#BSUB -J 1node_test
#BSUB -o 1node_test.o%J
#BSUB -e 1node_test.e%J
#BSUB -W 00:05
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module load gcc
module load cuda/10.2.89

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/stencil_results

set -x

ranks=1
gpus=0 
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/test/test_cpu -a

ranks=1
gpus=6 
#jsrun -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info cuda-memcheck --report-api-errors no ../../build/test/test_cuda -a

ranks=2
gpus=2
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info cuda-memcheck --report-api-errors no ../../build/test/test_cuda_mpi -a
#jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info cuda-memcheck --report-api-errors no ../../build/test/test_cuda_mpi "exchange2" -c "r=1,cmp"
jsrun --smpiargs="-gpu" -n 1 -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/test/test_cuda_mpi "exchange2" -c "r=1,da"

