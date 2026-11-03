#!/bin/bash
# sequential, 1 core. resume-safe (CSV rows keyed by seed+method).
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 NJOBS=1 MKL_NUM_THREADS=1
M=SSC,SSA,SSR_k1.5,BASILWs,GAM-ridge,NAM-SSC,NAM-SSR_k1.5,CAM-like,NOTEARS-MLP,DAGMA-nonlinear,DCE-like,NOTEARS-MLP-gc,GraN-DAG
mkdir -p logs
python3.13 ext_bench.py E1_d10_unif 10 1000 unif 4000 4010 $M 2>>logs/E1.err | tee -a logs/E1.out
python3.13 ext_bench.py E3_d10_laplace 10 1000 laplace 4000 4010 $M 2>>logs/E3.err | tee -a logs/E3.out
python3.13 ext_bench.py E2_d20_unif 20 1000 unif 4000 4010 $M 2>>logs/E2.err | tee -a logs/E2.out
