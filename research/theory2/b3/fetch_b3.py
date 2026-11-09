"""Fetch the B3 held-out csv members by HTTP range requests (TLS verification on; ../icml_exp/remote_zip.py), check
the zip CRC-32 (zipfile does this on extraction) against the value fixed in PLAN.md, and record SHA-256, row and
column counts.  Then (design check, inputs only) regenerate the ar_1_uniform_* actuator protocols from the published,
seeded generator and compare with the recorded red/green/blue/pol_1/pol_2 columns.  No sensor column is read.
Usage: python3 -I fetch_b3.py <out_dir> <report.json>"""
import sys, os, io, json, zipfile, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icml_exp"))
from remote_zip import HTTPRangeFile

OUT, REP = sys.argv[1:3]
URL = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/"
MEMBERS = {  # member -> CRC-32 from the zip central directory (PLAN.md section 1)
    "lt_camera_walks_v1": {"ar_1_uniform_ref": "b153930d", "ar_1_uniform_pol_1_90": "894684f7",
                           "ar_1_uniform_pol_12_30": "dcab420f"},
    "lt_camera_v1": {"scm_5_red": "a3d5a406", "scm_5_blue": "08eb6c50", "scm_5_green": "158bdeb3",
                     "scm_4_green": "de1c4f3d", "scm_4_blue": "efc3eb05", "scm_4_red": "9fdbe1ed"}}

def sample_AR_1_uniform(n, coef, random_state):          # copied verbatim from generators/actuators_ar.py
    rng = np.random.default_rng(random_state)
    innovations = rng.uniform(-1, 1, size=n)
    x = np.zeros_like(innovations); x[0] = innovations[0]
    for i in range(1, n): x[i] = x[i - 1] * coef + innovations[i]
    x -= x.min(); x /= x.max(); return (2 * x) - 1

SETTINGS = {"ref": {"red": (0.5, 128, 120), "green": (0.6, 128, 120), "blue": (0.7, 128, 120),
                    "pol_1": (0.8, 0, 30), "pol_2": (0.9, 0, 30)},
            "pol_1_90": {"red": (0.5, 128, 120), "green": (0.6, 128, 120), "blue": (0.7, 128, 120),
                         "pol_1": (0.8, 0, 30), "pol_2": (0.9, 90, 30)},
            "pol_12_30": {"red": (0.5, 128, 120), "green": (0.6, 128, 120), "blue": (0.7, 128, 120),
                          "pol_1": (0.8, 30, 30), "pol_2": (0.9, 30, 30)}}

def regenerate(name):
    seed = list(SETTINGS).index(name); out = {}
    for i, (var, (coef, mean, scale)) in enumerate(SETTINGS[name].items()):
        x = sample_AR_1_uniform(30000, coef, 100 * seed + i + 1728296067) * scale + mean
        out[var] = np.array([int(v) for v in x], float) if var in ("red", "green", "blue") else \
            np.array([float("%0.1f" % v) for v in x])
    return out

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rep = {}
    for ds, mem in MEMBERS.items():
        z = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(URL + ds + ".zip"), buffer_size=1 << 20))
        for exp, crc in mem.items():
            dst = os.path.join(OUT, exp + ".csv")
            info = z.getinfo(f"{ds}/{exp}/{exp}.csv")
            assert "%08x" % info.CRC == crc, (exp, "%08x" % info.CRC)
            if not os.path.exists(dst):
                with z.open(info) as src, open(dst + ".part", "wb") as f:   # zipfile verifies CRC-32 at EOF
                    while True:
                        b = src.read(1 << 22)
                        if not b: break
                        f.write(b)
                os.replace(dst + ".part", dst)
            h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
            df = pd.read_csv(dst, usecols=lambda c: c in ("red", "green", "blue", "pol_1", "pol_2"))
            ncol = len(pd.read_csv(dst, nrows=1).columns)
            rep[exp] = {"dataset": ds, "crc32": crc, "sha256": h, "rows": len(df), "columns": ncol}
            if exp.startswith("ar_1"):
                g = regenerate(exp.replace("ar_1_uniform_", ""))
                rep[exp]["design_check_max_abs_diff"] = {v: float(np.max(np.abs(df[v].to_numpy(float) - g[v])))
                                                         for v in g}
            print(exp, rep[exp], flush=True)
    json.dump(rep, open(REP, "w"), indent=1)
