#!/bin/bash
# d=110 driver (replaces the d=110 line of run_all.sh after GAM-ridge at d=110 did not finish in 60 min on seed 5000): GAM-ridge is run LAST per dataset.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NJOBS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SPADE_THREADS=1
M=SSC,SSA,SSR_k1.5,SSR_k2.5,BASILWs,SPADE,NOTEARS-linear,var-sortnregress,var-sortnregress-raw
python3.13 -u bench.py results 10 2000 boot 5000 5005 $M >> logs/d110_n2000_boot_b.log 2>&1
echo DONE110 > logs/DONE110
