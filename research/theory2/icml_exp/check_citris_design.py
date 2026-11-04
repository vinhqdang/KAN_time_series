"""Design check (inputs only, no sensor column read): regenerate the citris_1 actuator protocol from the
published generator (lt_crl_benchmark_v1/generators/citris_1.py, seed 1297781987260068, same draw order) and
compare with the experimenter-set columns of citris_1.csv.  Usage: check_citris_design.py <citris_1.csv>"""
import sys, numpy as np, pandas as pd

def reflect(x, lo, hi):
    return 2 * lo - x if x < lo else (2 * hi - x if x > hi else x)

N, burn = 100000, 1000
rng = np.random.default_rng(1297781987260068)
no_int = 0.35
flag = rng.choice(np.arange(6), replace=True, p=[no_int] + [(1 - no_int) / 5] * 5, size=N)
flag = np.concatenate([np.zeros(burn), flag])
Z = np.zeros((N + burn, 5))
for i in range(1, N + burn):
    r, g, b, p1, p2 = Z[i - 1]; f = flag[i]
    red = rng.uniform(0, 255) if f == 1 else r + rng.uniform(-50, 50)
    Z[i, 0] = reflect(red, 0, 255)
    green = rng.uniform(0, 255) if f == 2 else g + (r - g) / 2 + rng.uniform(-50, 50)
    Z[i, 1] = reflect(green, 0, 255)
    blue = rng.uniform(0, 255) if f == 3 else b + (g - b) / 4 + rng.uniform(-50, 50)
    Z[i, 2] = reflect(blue, 0, 255)
    pol_1 = rng.uniform(-90, 90) if f == 4 else p1 + rng.uniform(-10, 10)
    Z[i, 3] = reflect(pol_1, -90, 90)
    if f == 5:
        pol_2 = rng.uniform(-90, 90)
    else:
        sign = 1 if blue > red else -1          # unreflected current-step values, as in the generator
        pol_2 = p2 + sign * (p1 - p2) / 4 + rng.uniform(-5, 5)
    Z[i, 4] = reflect(pol_2, -90, 90)
Z = Z[burn:]; flag = flag[burn:]
df = pd.read_csv(sys.argv[1], usecols=["flag", "red", "green", "blue", "pol_1", "pol_2"])
print("flag equal:", np.mean(df.flag.values == flag))
for k, c in enumerate(["red", "green", "blue"]):
    print(c, "equal int():", np.mean(df[c].values == Z[:, k].astype(int)))
for k, c in enumerate(["pol_1", "pol_2"]):
    print(c, "max |csv - protocol(0.1)|:", np.max(np.abs(df[c].values - np.round(Z[:, 3 + k], 1))))
