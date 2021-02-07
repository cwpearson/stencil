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

DIR=$HOME/sync_work/stencil_results
OUT=$DIR/astaroth.csv

MPIRUN=$HOME/software/openmpi-4.0.5/bin/mpirun
# MPIRUN=mpirun

set -x

mkdir -p $DIR
echo "" > $OUT

for flags in "--staged" "--staged --colo" "--staged --colo --peer" "--staged --colo --peer --kernel" "--trivial --staged --colo --peer --kernel"; do
  echo $flags >> $OUT
  echo "nodes,ranks/node,ranks,x,y,z,iter (s),exch (s)" | tee -a $OUT
  for nodes in 1; do
    for rpn in 1 2 6; do
      let n=$nodes*$rpn
      echo -n "${nodes},${rpn}," | tee -a $OUT
      $MPIRUN -n $n ../../build/astaroth/astaroth $flags | tee -a $OUT
      #$MPIRUN -n $n nsys profile -t cuda,nvtx -f true -o $DIR/astaroth_${nodes}n_${rpn}rpn_%q{OMPI_COMM_WORLD_RANK} ../../  build/astaroth/astaroth $flags
    done
  done
  echo "" >> $OUT
done