#!/bin/bash
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
P=/usr/bin/python3
$P run_ablation.py results/val unif 3000 3005 base
$P run_ablation.py results/test unif 3005 3013 all
$P run_ablation.py results/laplace laplace 3005 3010 base
$P run_ablation.py results/hetero hetero 3005 3010 base
echo DONE
