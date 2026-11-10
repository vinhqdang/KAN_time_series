"""Seed and RNG-stream registry for the b1y studies, and the registry of every earlier stream (for the disjointness test).

Graph/parameter stream:  default_rng([401, family_index, p, seed])           (same graph for every arm, T, n, rho_cap)
Data stream:             default_rng([402, family_index, p, seed, T, arm_index, n, rho_index, 17])
Validation (debugging, timing, hyper-parameter tuning only): validation seeds below; never reported as evidence for
any hypothesis, except that Q2 tunes DAGMA/NOTEARS hyper-parameters on them (rule in PLAN.md).
"""
import numpy as np

KEY_GRAPH, KEY_DATA = 401, 402
KEY_Q5_GRAPH, KEY_Q5_DATA = 403, 404          # Q5 baselines use their own streams (fresh seeds)

SEEDS = dict(
    q1_test=list(range(4000, 4030)),           # 30 graphs per (family, p): Q1 main arms
    q1_dose=list(range(4200, 4220)),           # 20 graphs per (family, p): n-dose and rho-dose
    q2_test=list(range(4300, 4312)),           # 12 graphs per (family, p)
    q5_test=list(range(4100, 4112)),           # 12 graphs per (family, p)
    q1_val=list(range(9800, 9805)),
    q2_val=list(range(9810, 9813)),            # 3 graphs per (family, p): hyper-parameter tuning
    q5_val=[9820],
)

FAMILY_INDEX = ["symER", "ring", "chain", "outtree", "intree", "ER1", "ER2"]   # same order as b1/common.py FAMILIES


def graph_rng(fam, p, seed, key=KEY_GRAPH):
    return np.random.default_rng([key, FAMILY_INDEX.index(fam), p, seed])


def data_rng(fam, p, seed, T, arm_index, n, rho_index=0, key=KEY_DATA):
    return np.random.default_rng([key, FAMILY_INDEX.index(fam), p, seed, T, arm_index, n, rho_index, 17])


# ---------------------------------------------------------------------------------------------- earlier streams
# b1/run_b1.py   : [fi, p, s] and [fi, p, s, T, gen, regime, 7], fi in 0..6, s in 1000..1029 (test), 9000..9004 (val)
# b1nl           : PARAM_KEY NL=101, SGN=SGNU=102; DATA_KEY NL=201, SGN=202, SGNU=203; seeds 2000..2024 (NL, SGN),
#                  2000..2009 (SGNU), 9100..9102 (val)
# b1x/run_b1x.py : [301, fi, p, s], [302, fi, p, s, T, gen, 13]; seeds 3000..3011 (test), 9600 (val)
EARLIER = dict(
    b1=dict(prefixes=set(range(0, 7)), seeds=set(range(1000, 1030)) | set(range(9000, 9005))),
    b1nl=dict(prefixes={101, 102, 201, 202, 203}, seeds=set(range(2000, 2025)) | set(range(9100, 9103))),
    b1x=dict(prefixes={301, 302}, seeds=set(range(3000, 3012)) | {9600}),
)
