bsub -W 00:30 -nnodes 64 -P CSC362 -alloc_flags gpudefault -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash
