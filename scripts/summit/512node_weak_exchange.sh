#!/bin/bash
#BSUB -P csc362
#BSUB -J 512node_weak_exchange
#BSUB -o 512node_weak_exchange.o%J
#BSUB -e 512node_weak_exchange.e%J
#BSUB -W 1:00
#BSUB -nnodes 512
#BSUB -alloc_flags gpudefault

# should be compiled with -DSTEUP_STATS=ON -DEXCHANGE_STATS=OFF

set -eou pipefail

module load gcc
module load cuda

export X=750
export Y=750
export Z=750
export NUM_ITER=300


SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/stencil_results
OUT=$SCRATCH/512node_weak_exchange.csv

set -x

# CSV header
echo "bin,config,naive,x,y,z,s,ldx,ldy,ldz,MPI (B),Colocated (B),cudaMemcpyPeer (B),direct (B),iters,gpus,nodes,ranks,exchange (s)" > $OUT

# 6 ranks per node, 1 GPU per rank, weak
for nodes in 1 2 4 8 16 32 64 128 256 384 512; do
  ranks=6
  gpus=6 
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b packed:7 ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged                        --prefix $SCRATCH/${nodes}n_${ranks}r_${gpus}g_s_    | tee -a $OUT  
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b packed:7 ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo                 --prefix $SCRATCH/${nodes}n_${ranks}r_${gpus}g_sc_   | tee -a $OUT
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b packed:7 ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer          --prefix $SCRATCH/${nodes}n_${ranks}r_${gpus}g_scp_  | tee -a $OUT
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b packed:7 ../../build/bin/exchange-weak $X $Y $Z $NUM_ITER --staged --colo --peer --kernel --prefix $SCRATCH/${nodes}n_${ranks}r_${gpus}g_scpk_ | tee -a $OUT
done



