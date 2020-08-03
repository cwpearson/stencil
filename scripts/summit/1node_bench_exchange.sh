#!/bin/bash

#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

module load gcc
module load cuda
#module load openmpi
module load nsight-systems/2020.3.1.71

set -eou pipefail
set -x

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

mkdir -p $SCRATCH/stencil_results
mkdir -p $HOME/stencil_results

export OMP_NUM_THREADS=1

export QDREP=$SCRATCH/stencil_results/1n_1r_%q{OMPI_COMM_WORLD_RANK}
jsrun -n 1 -r 1 -c 1 -g 1 -a 1 -b rs js_task_info \
  nsys profile -t cuda,nvtx -o $QDREP -f true \
  ../../build/bin/bench-exchange \
  > $SCRATCH/stencil_results/1n_1r.txt

export QDREP=$SCRATCH/stencil_results/1n_2r_%q{OMPI_COMM_WORLD_RANK}
jsrun -n 2 -r 2 -c 1 -g 1 -a 1 -b rs js_task_info \
  nsys profile -t cuda,nvtx -o $QDREP -f true \
  ../../build/bin/bench-exchange \
  > $SCRATCH/stencil_results/1n_2r.txt

export QDREP=$SCRATCH/stencil_results/1n_6r_%q{OMPI_COMM_WORLD_RANK}
jsrun -n 6 -r 6 -c 1 -g 1 -a 1 -b rs js_task_info \
  nsys profile -t cuda,nvtx -o $QDREP -f true \
  ../../build/bin/bench-exchange \
  > $SCRATCH/stencil_results/1n_6r.txt


cp $SCRATCH/stencil_results/*.qdrep $HOME/stencil_results
cp $SCRATCH/stencil_results/*.txt $HOME/stencil_results
