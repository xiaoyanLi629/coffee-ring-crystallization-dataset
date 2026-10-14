"""Figures for the Scientific Data manuscript that are not already produced by the analysis modules.

Fig. 1  experimental design (5 Tables x 5x5 CaCl2-MgSO4 grid) and one droplet per detector mode in every modality.
Fig. 2  acquisition schedule: optical photographs and SEM/EDS sessions per day, by Table and detector mode.
Outputs manuscript/figures/{fig1_overview.png, fig2_schedule.png}
"""
import os
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/manuscript/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
meta = pd.read_csv(ROOT + "/metadata/samples.csv")
g = pd.read_csv(ROOT + "/analysis/out_g/derived.csv")[["path", "date_optical"]]
meta = meta.merge(g, on="path")
NA = [0.25, 0.5, 0.75, 1.0, 2.0]; CA = [0.5, 0.75, 1.0, 1.5, 2.0]; MG = [0.25, 0.5, 0.75, 1.0, 2.0]
MAPS = [("Ca", "Ca"), ("S", "S"), ("Cl", "Cl"), ("Na", "Na"), ("Mg", "Mg")]


def eds(path, e):
    a = cv2.imread(f"{ROOT}/experiment/data/{path}/{e}.jpeg").astype(np.float32).sum(2) / 765
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    return np.clip(a / (np.percentile(a, 99.7) + 1e-6), 0, 1)


def crop_optical(path):
    o = cv2.imread(f"{ROOT}/experiment/data/{path}/image.jpg")[:, :, ::-1]
    gr = cv2.GaussianBlur(o.max(2).astype(np.float32), (0, 0), 8)
    ys, xs = np.nonzero(gr > np.percentile(gr, 80))
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h = int(0.32 * o.shape[0])
    c = o[max(cy - h, 0):cy + h, max(cx - int(h * 4 / 3), 0):cx + int(h * 4 / 3)].astype(np.float32)
    return np.clip(c / np.percentile(c, 99.5), 0, 1)


fig = plt.figure(figsize=(7.2, 4.0))
gs = fig.add_gridspec(3, 7, height_ratios=[1.25, 1, 1], hspace=0.22, wspace=0.06)
# (a) design
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(-1.5, 36); ax.set_ylim(-1.3, 7.2); ax.axis("off")
for t, na in enumerate(NA):
    x0 = 1 + t * 7
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((x0 + i, j), .9, .9, color=plt.cm.viridis(0.15 + 0.17 * t), alpha=.35 + .1 * (i + j) / 4))
    ax.text(x0 + 2.45, 5.35, f"Table {t + 1}\nNaHCO$_3$ {na} mM", ha="center", va="bottom")
    if t == 0:
        ax.text(x0 + 2.45, -0.75, "CaCl$_2$ 0.5 → 2.0 mM", ha="center", va="top", fontsize=7)
        ax.text(x0 - 0.35, 2.45, "MgSO$_4$\n0.25 → 2.0", ha="right", va="center", fontsize=7)
ax.text(-1.5, 7.1, "a", fontweight="bold", fontsize=10, va="top")
# (b,c) one droplet per detector mode, same CaCl2-MgSO4 grid cell (Sample 13)
for r, (path, lab) in enumerate([("Table_2/Sample_13/2_rep", "b"), ("Table_4/Sample_13/2_rep", "c")]):
    row = meta[meta.path == path].iloc[0]
    panels = [("optical", crop_optical(path), None),
              (f"SEM ({row.sem_signal})", cv2.imread(f"{ROOT}/experiment/data/{path}/sem.jpg", 0), "gray")]
    panels += [(f"EDS {e}", eds(path, f), "magma") for e, f in MAPS]
    for k, (t, im, cm) in enumerate(panels):
        a = fig.add_subplot(gs[r + 1, k])
        a.imshow(im, cmap=cm); a.set_xticks([]); a.set_yticks([])
        a.set_title(t, fontsize=7, pad=2)
        if k == 0:
            a.set_ylabel(f"T{row.table}_S{row['sample']}_R{row.replicate}", fontsize=7)
            a.text(-0.2, 1.12, lab, transform=a.transAxes, fontweight="bold", fontsize=10)
fig.savefig(OUT + "/fig1_overview.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Fig. 2 schedule
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=False)
cols = {t: plt.cm.viridis(0.15 + 0.17 * (t - 1)) for t in range(1, 6)}
for a, col, title in [(ax[0], "date_optical", "Optical photographs"), (ax[1], "session", "SEM/EDS sessions")]:
    d = meta.copy(); d["day"] = pd.to_datetime(d[col])
    days = sorted(d.day.unique())
    lab = [pd.Timestamp(x).strftime("%m-%d") if pd.Timestamp(x).year == 2021 else pd.Timestamp(x).strftime("%Y-%m-%d") for x in days]
    for i, day in enumerate(days):
        bottom = 0
        dd = d[d.day == day]
        for t in range(1, 6):
            for mode, hatch in (("BEC", ""), ("SEI", "////")):
                n = int(((dd.table == t) & (dd.sem_signal == mode)).sum()) if col == "session" else int((dd.table == t).sum()) * (mode == "BEC")
                if n:
                    a.bar(i, n, bottom=bottom, color=cols[t], hatch=hatch, edgecolor="k", lw=.4)
                    bottom += n
    a.set_xticks(range(len(days))); a.set_xticklabels(lab, rotation=0, fontsize=6.5)
    a.set_ylabel("droplets"); a.set_title(title, fontsize=8, loc="left")
hs = [Rectangle((0, 0), 1, 1, color=cols[t]) for t in range(1, 6)] + [Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch=""), Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="////")]
ax[1].legend(hs, [f"Table {t}" for t in range(1, 6)] + ["BEC", "SEI"], ncol=7, fontsize=6.5, loc="upper center", bbox_to_anchor=(.5, -0.25), frameon=False)
for a, l in zip(ax, "ab"):
    a.text(-0.07, 1.05, l, transform=a.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "/fig2_schedule.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("ms figs done")
