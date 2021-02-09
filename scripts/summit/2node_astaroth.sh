#!/bin/bash
#BSUB -P csc362
#BSUB -J 2node_astaroth
#BSUB -o 2node_astaroth.o%J
#BSUB -e 2node_astaroth.e%J
#BSUB -W 0:10
#BSUB -nnodes 64
#BSUB -alloc_flags gpudefault

# should be compiled with -DSTEUP_STATS=ON -DEXCHANGE_STATS=OFF

set -eou pipefail

module reset
module load gcc
module load cuda/11.0.3
module load nsight-systems/2020.4.1.144

DIR=/gpfs/alpine/csc362/scratch/cpearson/stencil_results
OUT=$DIR/2node_astaroth.csv

set -x

mkdir -p $DIR
echo "" > $OUT

for flags in "--staged --colo --peer --kernel"; do
  echo $flags >> $OUT
  echo "nodes,ranks/node,ranks,x,y,z,iter (s),exch (s)" >> $OUT
  for nodes in 1 2; do
    for rpn in 1 2 6; do
      let n=$nodes*$rpn
      echo -n "${nodes},${rpn}," | tee -a $OUT
      #jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/astaroth/astaroth $flags | tee -a $OUT
      jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs nsys profile -t cuda,nvtx -f true -o $DIR/astaroth_${nodes}n_${rpn}rpn_scpk_%q{OMPI_COMM_WORLD_RANK} ../../build/astaroth/astaroth $flags
    done
  done
  echo "" >> $OUT
done
