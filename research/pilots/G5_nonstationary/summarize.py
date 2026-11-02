import json,numpy as np
R=json.load(open("g5_results.json"))["test"]
ms=list(next(iter(R.values())).keys())
out=["| scenario | method | AUROC mean+-sd | AUPRC mean+-sd |","|---|---|---|---|"]
for scn in ["I1","I1C","ST"]:
    for m in ms:
        a=[R[f"{scn}|{s}"][m]["auroc"] for s in range(200,210)]; p=[R[f"{scn}|{s}"][m]["auprc"] for s in range(200,210)]
        out.append(f"| {scn} | {m} | {np.mean(a):.3f} +- {np.std(a):.3f} | {np.mean(p):.3f} +- {np.std(p):.3f} |")
out+=["","Paired AUPRC differences (mean +- sd over 10 seeds; mean/(sd/sqrt10) in brackets)"]
for scn in ["I1","I1C","ST"]:
    for a,b in [("SPADE-diff","SPADE-levels"),("ECM-SPADE","SPADE-diff"),("ECM-SPADE","ECM-SPADE(no-placebo)"),("ECM-SPADE","PCMCI-diff"),("ECM-SPADE","VARLasso-diff")]:
        d=np.array([R[f"{scn}|{s}"][a]["auprc"]-R[f"{scn}|{s}"][b]["auprc"] for s in range(200,210)])
        out.append(f"{scn}: {a} - {b}: {d.mean():+.3f} +- {d.std():.3f} [t={d.mean()/(d.std(ddof=1)/np.sqrt(10)+1e-12):.1f}]")
open("g5_results.md","w").write("\n".join(out)); print("\n".join(out))
