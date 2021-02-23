#!/bin/bash
#BSUB -P csc362
#BSUB -J 512node_astaroth
#BSUB -o 512node_astaroth.o%J
#BSUB -e 512node_astaroth.e%J
#BSUB -W 1:00
#BSUB -nnodes 512
#BSUB -alloc_flags gpudefault

# should be compiled with -DSTEUP_STATS=ON -DEXCHANGE_STATS=OFF

set -eou pipefail

module reset
module load gcc
module load cuda/11.0.3
module load nsight-systems/2020.3.1.71

DIR=/gpfs/alpine/csc362/scratch/cpearson/stencil_results
OUT=$DIR/512node_astaroth.csv

set -x

mkdir -p $DIR
echo "" > $OUT

for flags in "--staged" "--trivial --staged" "--staged --colo" "--staged --colo --peer" "--staged --colo --peer --kernel" "--trivial --staged --colo --peer --kernel"; do
  flags="$flags --no-compute"
  echo $flags >> $OUT
  echo "nodes,ranks/node,ranks,x,y,z,iter (s),exch (s)" >> $OUT
  for nodes in 1 2 4 8 16 32 64 128 256 512; do
    for rpn in 1 2 4 6; do
      let n=$nodes*$rpn
      echo -n "${nodes},${rpn}," | tee -a $OUT
      jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/astaroth/astaroth $flags | tee -a $OUT
      #jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs nsys profile -t cuda,nvtx -f true -o $DIR/astaroth_${nodes}n_${rpn}rpn_${X}x_%q{OMPI_COMM_WORLD_RANK} ../../build/astaroth/astaroth
    done
  done
  echo "" >> $OUT
done
