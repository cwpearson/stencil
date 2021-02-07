#!/bin/bash
#BSUB -P csc362
#BSUB -J 64node_astaroth
#BSUB -o 64node_astaroth.o%J
#BSUB -e 64node_astaroth.e%J
#BSUB -W 2:00
#BSUB -nnodes 64
#BSUB -alloc_flags gpudefault

# should be compiled with -DSTEUP_STATS=ON -DEXCHANGE_STATS=OFF

set -eou pipefail

module reset
module load gcc
module load cuda/11.0.3
module load nsight-systems/2020.3.1.71

DIR=$HOME/sync_work/stencil_results
OUT=$DIR/astaroth.csv

set -x

mkdir -p $DIR

echo "" > $OUT

echo "nodes,ranks/node,ranks,x,y,z,iter (s),exch (s)" >> $OUT
for nodes in 1; do
  for rpn in 1 2 6; do
    let n=$nodes*$rpn
    echo -n "${nodes},${rpn}," | tee -a $OUT
    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/astaroth/astaroth | tee -a $OUT
    #jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs nsys profile -t cuda,nvtx -f true -o $DIR/astaroth_${nodes}n_${rpn}rpn_${X}x_%q{OMPI_COMM_WORLD_RANK} ../../build/astaroth/astaroth
  done
done
