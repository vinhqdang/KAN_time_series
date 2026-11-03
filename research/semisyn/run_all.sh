#!/bin/bash
# resume-safe sequential driver (1 core). Each (dataset, method) row is skipped if already in results/.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NJOBS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SPADE_THREADS=1
mkdir -p results logs
M=SSC,SSA,SSR_k1.5,SSR_k2.5,BASILWs,SPADE,GAM-ridge,NOTEARS-linear,var-sortnregress,var-sortnregress-raw
python3.13 -u bench.py results 1 2000 boot 3700 3720 $M >> logs/d11_n2000_boot.log 2>&1
python3.13 -u bench.py results 1 2000 gauss 3800 3820 $M >> logs/d11_n2000_gauss.log 2>&1
python3.13 -u bench.py results 1 500 boot 3500 3520 $M >> logs/d11_n500_boot.log 2>&1
python3.13 -u bench.py results 1 500 gauss 3600 3620 $M >> logs/d11_n500_gauss.log 2>&1
python3.13 -u bench.py results 10 2000 boot 5000 5005 $M >> logs/d110_n2000_boot.log 2>&1
echo ALLDONE > logs/ALLDONE
