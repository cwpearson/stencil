bsub -W 01:00 -nnodes 1 -P CSC362 -alloc_flags gpudefault -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash
