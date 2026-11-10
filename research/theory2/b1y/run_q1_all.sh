#!/bin/bash
# Q1 test run: 14 main shards (7 families x p in {5, 10}) + dose shards (ring, chain, ER2 x p in {5, 10}; n-dose and rho-dose). 4 parallel workers.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/user/venv312/bin/python
{
for fam in symER ring chain outtree intree ER1 ER2; do for p in 5 10; do echo "main $fam $p"; done; done
for fam in ring chain ER2; do for p in 5 10; do echo "ndose $fam $p"; echo "rhodose $fam $p"; done; done
} | xargs -P 4 -L 1 bash -c 'nice -n 10 '$PY' run_q1.py $0 results/shards/q1/q1_$0_$1_$2.csv $1 $2 >> logs/q1_$0_$1_$2.log 2>&1'
echo done > logs/q1_all.done
