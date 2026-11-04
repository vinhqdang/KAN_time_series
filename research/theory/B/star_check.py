"""Exact star / fork / collider checks for Candidate B (contemporaneous summary SEM, arbitrary positive noise variances)."""
import numpy as np
rng = np.random.default_rng(7)
def r2(S):
    P = np.linalg.inv(S); return 1 - 1/(np.diag(S)*np.diag(P))
bad_in = bad_out = 0; N = 20000; lines = []
for rep in range(N):
    m = rng.integers(2, 8); om = rng.uniform(0.1, 5, m + 1); b = rng.choice([-1,1], m) * rng.uniform(0.05, 3, m)
    # out-star: node 0 -> leaves 1..m
    B = np.zeros((m+1, m+1)); B[1:, 0] = b
    T = np.linalg.inv(np.eye(m+1) - B); S = T @ np.diag(om) @ T.T; r = r2(S)
    bad_out += (r[0] < r[1:]).sum() > 0          # any leaf with R2 above root => at least one correct edge
    # in-star: leaves 1..m -> node 0
    B = np.zeros((m+1, m+1)); B[0, 1:] = b
    T = np.linalg.inv(np.eye(m+1) - B); S = T @ np.diag(om) @ T.T; r = r2(S)
    bad_in += (r[1:] > r[0]).sum() > 0           # any parent with R2 above sink => a misordered edge
lines.append(f"out-star (m in 2..7, random weights/noise, {N} draws): draws with ANY correctly ordered edge = {bad_out}  (theory: 0)")
lines.append(f"in-star  (same): draws with ANY misordered edge = {bad_in}  (theory: 0)")
# m=1 tie
om = np.array([1.0, 2.0]); b = 1.3; B = np.array([[0,0],[b,0]]); T = np.linalg.inv(np.eye(2)-B); S = T@np.diag(om)@T.T
lines.append(f"m=1: R2 = {r2(S)} (tie)")
print("\n".join(lines)); open("results_B_star.txt","w").write("\n".join(lines))
