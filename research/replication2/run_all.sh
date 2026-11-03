#!/bin/bash
# resume-safe: one CSV per dataset (4 rows); rerun skips finished datasets. 2 cores max.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NJOBS=1 MKL_NUM_THREADS=1
mkdir -p results logs
cat jobs.txt | xargs -P2 -L1 bash -c 'if [ -f "$0" ] && [ "$(python3.13 -c "import pandas as pd,sys;print(len(pd.read_csv(sys.argv[1])))" "$0" 2>/dev/null)" = "4" ]; then exit 0; fi; for t in 1 2; do python3.13 bench.py "$0" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> logs/$(basename $0 .csv).log 2>&1 && break; done'
echo ALLDONE > logs/ALLDONE
