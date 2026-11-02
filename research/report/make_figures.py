"""Figures for the progress report, all drawn from the CSV/JSON files in experimental_results/ and research/pilots/."""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.join(os.path.dirname(__file__), "..", ".."); ER = f"{ROOT}/experimental_results"; PL = f"{ROOT}/research/pilots"
OUT = os.path.join(os.path.dirname(__file__), "fig"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"SPADE": "#d95f02", "good": "#1b9e77", "bad": "#7570b3", "grey": "#9a9a9a", "blue": "#1f78b4"}
def save(name): plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=170); plt.close()

# F1 timeline
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
steps = [("1. Bài SPADE\n(đã đăng, Array)", "#cfe8ff"), ("2. Revision 2\nthêm dữ liệu thật\nphi tài chính", "#cfe8ff"),
         ("3. 'Mở rộng' ngẫu hứng\nX1-X5\n(chưa phải kết quả)", "#ffe1c4"), ("4. Scoping review\n130 bài đã xác minh", "#d9f2d9"),
         ("5. Năm pilot\nG1-G5", "#d9f2d9"), ("6. Bây giờ:\nchọn hướng,\nthiết kế lại", "#fff3b0")]
for i, (t, col) in enumerate(steps):
    x = 0.15 + i * 1.65
    ax.add_patch(FancyBboxPatch((x, 0.9), 1.45, 1.6, boxstyle="round,pad=0.03", fc=col, ec="#555"))
    ax.text(x + 0.725, 1.7, t, ha="center", va="center", fontsize=8.6)
    if i < len(steps) - 1: ax.add_patch(FancyArrowPatch((x + 1.48, 1.7), (x + 1.65, 1.7), arrowstyle="-|>", mutation_scale=12))
ax.text(5, 0.35, "xanh dương: đã xong/đã công bố   cam: làm nhanh, chưa đủ chặt   xanh lá: làm lại có hệ thống   vàng: cần quyết định", ha="center", fontsize=8.5, color="#444")
save("f1_timeline")

# F2 SPADE schematic
fig, ax = plt.subplots(figsize=(9.5, 3.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, t, fc="#eef"): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#444")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)
for k, y in enumerate([3.0, 1.9, 0.8]): box(0.2, y, 1.2, 0.7, f"nguyên nhân\nứng viên $x_{k+1}$", "#fff3e0")
xs = np.linspace(-2, 2, 50)
for k, (y, f) in enumerate(zip([3.0, 1.9, 0.8], [np.tanh(1.5 * xs), 0.3 * (xs ** 2 - 1), np.sin(1.5 * xs) * 0.7])):
    axin = fig.add_axes([0.30, 0.13 + (2 - k) * 0.25, 0.17, 0.15]); axin.plot(xs, f, color=C["SPADE"], lw=2); axin.set_xticks([]); axin.set_yticks([]); axin.set_title(f"hàm spline $\\phi_{{{k+1}}}$", fontsize=7)
    ax.add_patch(FancyArrowPatch((1.4, y + 0.35), (2.7, y + 0.35), arrowstyle="-|>", mutation_scale=10))
    ax.add_patch(FancyArrowPatch((4.7, y + 0.35), (6.3, 1.95), arrowstyle="-|>", mutation_scale=10))
box(6.3, 1.5, 1.4, 0.9, "cộng lại:\n$\\hat x_e=b+\\sum_j\\phi_j(x_j)$", "#e3f2fd"); ax.add_patch(FancyArrowPatch((7.7, 1.95), (8.5, 1.95), arrowstyle="-|>", mutation_scale=10)); box(8.5, 1.6, 1.3, 0.7, "biến kết quả\n$x_e$", "#e8f5e9")
ax.text(5.0, 4.02, "Mỗi biến chỉ được dự đoán qua các hàm một biến của các biến cha (không có mạng trộn ở giữa)", ha="center", fontsize=9, weight="bold")
ax.text(7.0, 0.6, "Phạt group-lasso: cắt cả cạnh (đưa cả nhóm hệ số về 0)\nRàng buộc acyclicity (NOTEARS): đồ thị không có vòng", ha="center", fontsize=8.5, color="#333")
save("f2_spade")

# F3 published headline
r = pd.read_csv(f"{ER}/instdag_raw.csv"); order = ["SPADE", "NoGAM", "SCORE", "NOTEARS-linear", "GOLEM", "DAGMA-nonlinear", "NOTEARS-MLP", "GraN-DAG", "DAGMA-linear"]
g = r.groupby(["method", "d"]).auroc.mean().unstack(); g = g.loc[[m for m in order if m in g.index]]
fig, ax = plt.subplots(figsize=(9.5, 3.8)); w = 0.26
for i, d in enumerate([6, 10, 20]): ax.bar(np.arange(len(g)) + (i - 1) * w, g[d].values, w, label=f"d={d}", color=["#fdbf6f", "#fb9a99", "#cab2d6"][i] if False else ["#a6cee3", "#1f78b4", "#08306b"][i])
ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=30, ha="right"); ax.set_ylabel("AUROC (khôi phục đồ thị)"); ax.axhline(0.5, color="k", lw=0.6, ls=":"); ax.legend(ncol=3, frameon=False)
ax.set_title("Kết quả chính của bài đã đăng (đồ thị DAG tức thời phi tuyến, n=1500)", fontsize=10); save("f3_published")

# F4 real data
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
d1 = pd.read_csv(f"{ER}/realworld_bike_summary.csv").set_index("method"); d2 = pd.read_csv(f"{ER}/realworld_bike_hourly_primary_summary.csv").set_index("method"); d3 = pd.read_csv(f"{ER}/realworld_beijing_primary_summary.csv").set_index("method")
sel = ["SPADE", "NOTEARS-linear", "DAGMA-nonlinear", "NoGAM", "SCORE", "GOLEM"]
for ax, df, t in zip(axs, [d1, d2, d3], ["Bike theo ngày (n=730)", "Bike theo giờ, mẫu 2000", "Beijing PM2.5, mẫu 2000"]):
    v = df.loc[sel, "bk_auroc_mean"]; e = df.loc[sel, "bk_auroc_std"]
    ax.bar(range(len(sel)), v, yerr=e, color=[C["SPADE"] if s == "SPADE" else C["grey"] for s in sel], capsize=2); ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=40, ha="right", fontsize=8); ax.set_title(t, fontsize=9); ax.set_ylim(0, 1)
axs[0].set_ylabel("BK-AUROC (khớp kiến thức miền)"); save("f4_realworld")

# F5 X1 + X4
v = pd.read_csv(f"{ER}/x1_validation_raw.csv"); f = pd.read_csv(f"{ER}/x1_final_raw.csv")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
gv = v.groupby(["d", "variant"]).auroc.mean().unstack()
gv[["base", "adaptive", "refit", "boot", "adaptive_boot"]].plot(kind="bar", ax=axs[0], color=["#999", "#a6cee3", "#fb9a99", "#b2df8a", C["SPADE"]], width=0.8); axs[0].set_ylim(0.8, 1.0); axs[0].set_title("X1: chọn biến thể (seed validation)", fontsize=9); axs[0].legend(fontsize=6, ncol=2); axs[0].set_ylabel("AUROC"); axs[0].tick_params(axis="x", rotation=0)
ff = f[f.part == "fresh"].groupby(["d", "method"]).auroc.mean().unstack(); ff.plot(kind="bar", ax=axs[1], color=["#999", C["SPADE"]]); axs[1].set_ylim(0.8, 1.04); axs[1].legend(ncol=2, fontsize=7, loc="upper center"); axs[1].set_title("X1: seed mới (45 lần chạy)", fontsize=9); axs[1].tick_params(axis="x", rotation=0)
rows = [pd.read_csv(p) for d in (50, 100, 200) for p in sorted(glob.glob(f"{ER}/x4_final_d{d}*.csv"))]; x4 = pd.concat(rows); g4 = x4.groupby(["d", "method"]).auroc.mean().unstack()
for m, c, lab in [("SPADE (lg=0.01)", C["grey"], "SPADE λ=0.01 cố định"), ("SPADE-scaled (lg=c/d,K=15)", C["SPADE"], "SPADE-scaled"), ("NOTEARS-linear", C["blue"], "NOTEARS-linear"), ("DAGMA-linear", "#33a02c", "DAGMA-linear")]:
    if m in g4: axs[2].plot(g4.index, g4[m], "o-", color=c, label=lab)
axs[2].set_xlabel("d (số biến)"); axs[2].set_xticks([50, 100, 200]); axs[2].set_ylim(0.4, 1.02); axs[2].legend(fontsize=7, loc="lower left"); axs[2].set_title("X4: quy mô (chỉ có baseline tuyến tính)", fontsize=9); save("f5_extension_x1x4")

# F6 X2 + X3 + X5
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6))
s = pd.read_csv(f"{ER}/sachs_raw.csv"); s = s[(s.status == "ok") & (s.part == "primary")]; gs = s.groupby("method").auroc.agg(["mean", "std"]).sort_values("mean", ascending=False)
gs.index = [i.replace("SPADE-adaptive_boot", "SPADE-AB") for i in gs.index]
axs[0].barh(gs.index[::-1], gs["mean"][::-1], xerr=gs["std"][::-1], color=[C["SPADE"] if "SPADE" in i else C["grey"] for i in gs.index[::-1]]); axs[0].axvline(0.5, color="k", lw=0.6, ls=":"); axs[0].set_title("X2: Sachs (18 cạnh thật), AUROC", fontsize=9)
t = pd.read_csv(f"{ER}/x3_fidelity_raw.csv"); t["nl"] = t.linear_oracle_r2 < 0.5; tt = t[t.found & t.nl].groupby("method").agg(r2=("r2", "mean"), n=("r2", "size"))
tt = tt.loc[["MLP-edge", "SPADE", "SPADE-2stage"]]; axs[1].bar(tt.index, tt.r2, color=[C["grey"], C["SPADE"], "#e6ab02"]); [axs[1].text(i, tt.r2.iloc[i] + 0.02, f"n={tt.n.iloc[i]}", ha="center", fontsize=8) for i in range(3)]
axs[1].set_ylim(0, 1); axs[1].set_title("X3: R² hình dạng, cạnh phi tuyến mạnh", fontsize=9)
f5 = pd.read_csv(f"{ER}/x5_final_raw.csv"); b5 = pd.read_csv(f"{ER}/x5b_beijing_raw.csv")
a = f5.groupby("model").mse.mean().sort_values(); a = a[a.index.isin(["Naive (last value)", "SPADE-fast (res+gl)", "TSMixer+RevIN", "PatchTST+RevIN", "CD-KAN", "LSTM+RevIN"])]
axs[2].barh(a.index[::-1], a.values[::-1], color=["#e6ab02" if "Naive" in i else (C["SPADE"] if "SPADE" in i or "CD-KAN" in i else C["grey"]) for i in a.index[::-1]]); axs[2].set_title("X5: dự báo tài chính, MSE (thấp = tốt)", fontsize=9); save("f6_x2x3x5")

# F7 G1
d10 = pd.read_csv(f"{PL}/G1_debiased_selection/raw_test_d10.csv"); d20 = pd.read_csv(f"{PL}/G1_debiased_selection/raw_test_d20.csv")
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5)); meths = ["base", "adaptive", "refit_s0.001", "dss_s0.001"]; labs = ["base\n(group-lasso)", "adaptive", "refit", "DSS\n(đầy đủ)"]
for ax, col, ttl in zip(axs, ["slope", "curve_r2", "auprc"], ["Hệ số hiệu chỉnh biên độ (1 = đúng)", "R² đường cong học được", "AUPRC cạnh"]):
    vals = [[d[d.method == m][col].mean() for m in meths] for d in (d10, d20)]
    for i, (d, c) in enumerate(zip(["d=10", "d=20"], ["#a6cee3", "#1f78b4"])): ax.bar(np.arange(4) + (i - 0.5) * 0.38, vals[i], 0.38, color=c, label=d)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8); ax.set_title(ttl, fontsize=9)
    if col == "slope": ax.axhline(1, color="k", lw=0.6, ls=":"); ax.legend(frameon=False)
save("f7_g1")

# F8 G2 + G3
g2 = pd.read_csv(f"{PL}/G2_shape_benchmark/summary_pooled.csv").set_index("method")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6)); m2 = ["GAM-ridge", "SPADE", "MLP-edge", "Linear", "NT-MLP"]; m2 = [m for m in m2 if m in g2.index]
axs[0].bar(np.arange(len(m2)) - 0.2, g2.loc[m2, "AUROC"], 0.4, label="AUROC cạnh", color="#a6cee3"); axs[0].bar(np.arange(len(m2)) + 0.2, g2.loc[m2, "SF-F1"], 0.4, label="SF-F1 (cạnh tìm được VÀ đúng hình dạng)", color=C["SPADE"])
axs[0].set_xticks(range(len(m2))); axs[0].set_xticklabels(m2, fontsize=8); axs[0].legend(fontsize=7, frameon=False); axs[0].set_title("G2: GAM-ridge (không DAG, không sparsity) ngang SPADE", fontsize=9)
frames = [pd.read_csv(p) for p in glob.glob(f"{PL}/G3_scale/results/test_*_s800_*.csv")]; g3 = pd.concat(frames); g3 = g3[g3.gen == "std"]
pt = g3.groupby(["d", "method"]).auroc.mean().unstack()
for m, c in [("SPOT", C["SPADE"]), ("SPOT-oracle-order", C["good"]), ("NOTEARS-linear", C["blue"])]:
    if m in pt: axs[1].plot(pt.index, pt[m], "o-", label=m.replace("SPOT-oracle-order", "SPOT với thứ tự ĐÚNG (chẩn đoán)"), color=c)
axs[1].set_xticks([50, 100, 200]); axs[1].set_ylim(0.4, 1.05); axs[1].legend(fontsize=7, frameon=False); axs[1].set_xlabel("d"); axs[1].set_title("G3: lỗi nằm ở bước ước lượng thứ tự", fontsize=9); save("f8_g2g3")

# F9 G4 + G5
A = pd.read_csv(f"{PL}/G4_benchmark_audit/results_A_raw.csv"); B = pd.read_csv(f"{PL}/G4_benchmark_audit/results_B_std.csv")
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6)); show = [("var-sortnregress", C["grey"]), ("NOTEARS-linear", C["blue"]), ("SPADE", C["SPADE"])]
for ax, D, ttl in [(axs[0], A, "G4: bộ sinh của dự án (sortable ~0.9)"), (axs[1], B, "G4: bộ sinh đã chuẩn hóa từng biến")]:
    gg = D.groupby(["d", "method"]).auroc.mean().unstack()
    for m, c in show:
        if m in gg: ax.plot(gg.index, gg[m], "o-", color=c, label=m.replace("var-sortnregress", "chỉ sắp theo phương sai"))
    ax.set_xticks(sorted(D.d.unique())); ax.set_ylim(0.45, 1.02); ax.set_title(ttl, fontsize=9); ax.set_xlabel("d")
axs[0].legend(fontsize=7, frameon=False); axs[0].set_ylabel("AUROC")
j = json.load(open(f"{PL}/G5_nonstationary/g5_results.json"))["test"]; rows = []
for k, v in j.items():
    sc = k.split("|")[0]
    for m, mm in v.items(): rows.append(dict(sc=sc, method=m, auprc=mm["auprc"]))
g5 = pd.DataFrame(rows).groupby(["sc", "method"]).auprc.mean().unstack(); cols = [c for c in ["SPADE-levels", "SPADE-diff", "ECM-SPADE"] if c in g5]
g5.loc[["I1", "I1C", "ST"], cols].plot(kind="bar", ax=axs[2], color=["#bbb", "#a6cee3", C["SPADE"]]); axs[2].set_title("G5: dữ liệu mức hỏng khi chuỗi I(1)", fontsize=9); axs[2].set_ylabel("AUPRC"); axs[2].tick_params(axis="x", rotation=0); axs[2].legend(fontsize=7, frameon=False); save("f9_g4g5")

# F10 decision matrix
fig, ax = plt.subplots(figsize=(10, 3.9)); ax.axis("off")
rowsT = [["G1", "Thiên lệch biên độ/chọn cạnh", "Adaptive weighting sửa biên độ (20/20 đồ thị)", "Refit + stability làm AUPRC giảm", "CẬN BIÊN"],
         ["G2", "Đo độ trung thực hình dạng", "Metric rẻ; 'tìm được' ≠ 'đúng hình dạng'", "GAM-ridge ngang SPADE; rank gần như không đổi", "CẬN BIÊN"],
         ["G3", "Khám phá cộng ở d lớn", "Với thứ tự đúng: AUROC 0.93-1.00", "Thứ tự ước lượng ~ngẫu nhiên", "BỎ (dạng này)"],
         ["G4", "Kiểm toán benchmark", "sortnregress ngang SPADE; nhiễu ±0.05", "Ý tưởng đã có (Reisach)", "PHƯƠNG PHÁP"],
         ["G5", "Chuỗi thời gian phi dừng", "Dữ liệu mức hỏng khi I(1)", "Công thức sửa chỉ +0.02", "CẬN BIÊN"]]
tb = ax.table(cellText=rowsT, colLabels=["", "Hướng", "Điều đứng vững", "Điều không", "Kết luận"], loc="center", cellLoc="left", colWidths=[0.04, 0.2, 0.3, 0.3, 0.13]); tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 2.0)
cmap = {"CẬN BIÊN": "#fff3b0", "BỎ (dạng này)": "#f8c8c8", "PHƯƠNG PHÁP": "#cfe8ff"}
for i, rr in enumerate(rowsT, start=1): tb[(i, 4)].set_facecolor(cmap[rr[4]])
for c in range(5): tb[(0, c)].set_facecolor("#ddd")
save("f10_decision"); print("figures:", sorted(os.listdir(OUT)))
