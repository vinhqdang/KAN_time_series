import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_pilot import *
from src.synthetic import generate_instantaneous_anm
a = [];b = [];c = []
for s in range(1000, 1020):
    X, A, f, sd = gen_iscm(1500, 10, s); a.append(varsortability(X, A))
    X, A, f, sd = gen_iscm(1500, 10, s, standardize=False); b.append(varsortability(X, A))
    X, A = generate_instantaneous_anm(1500, 10, s); c.append(varsortability(X, A))
print("varsortability d=10: iSCM %.3f | same generator unstandardised %.3f | src.synthetic %.3f" % (np.mean(a), np.mean(b), np.mean(c)))
