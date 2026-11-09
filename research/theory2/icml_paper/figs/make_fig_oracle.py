"""Paper version of b1x/results/fig_oracle (same data: b1x/results/oracle_b3.json, cells)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
here = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(here, "..", "..", "b1x", "results", "oracle_b3.json")))["cells"]
cols = dict(symER="#1b9e77", ring="#66a61e", chain="#d95f02", outtree="#7570b3", intree="#e7298a",
            ER1="#a6761d", ER2="#666666")
names = dict(symER="symER", ring="ring", chain="chain", outtree="out-tree", intree="in-tree", ER1="ER(1)", ER2="ER(2)")
SC = {"symER", "ring"}
plt.rcParams.update({"font.size": 7})
fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.85), sharey=True)
for k, (pred, lab) in enumerate((("gap", r"exact gap $s_\rho(G_T)-s_\rho^W$"), ("prob", "detectability oracle"))):
    for fam in cols:
        c = [r for r in C if r["family"] == fam]
        ax[k].scatter([r[pred] for r in c], [r["obs"] for r in c], s=[3 + 1.2 * r["T"] for r in c], color=cols[fam],
                      marker="s" if fam in SC else "o", alpha=0.85, label=names[fam], linewidths=0)
    ax[k].plot([0, 0.52], [0, 0.52], "k:", lw=0.6)
    ax[k].set_xlabel(lab, fontsize=6.5)
    x = [r[pred] for r in C]; y = [r["obs"] for r in C]
    ax[k].set_title(f"Spearman {stats.spearmanr(x, y).statistic:.2f}, $R^2$ {stats.pearsonr(x, y).statistic**2:.2f}", fontsize=6.5)
    ax[k].tick_params(labelsize=6)
ax[0].set_ylabel("observed gain (rel OA)", fontsize=6.5)
ax[0].legend(fontsize=4.8, frameon=False, ncol=2, loc="upper left", handletextpad=0.1, columnspacing=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(here, "fig_oracle.pdf"))
