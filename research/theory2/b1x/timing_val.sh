#!/bin/bash
# Validation timing (seed 9600, --timing: fits only, no metric computed).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for fam in ring chain intree ER2; do for p in 5 10; do
  /usr/bin/python3 run_b1x.py val val_timing/t_${fam}_${p}.csv.gz $fam $p --timing
done; done
echo ALLDONE
