"""Module H: reference benchmarks with simple, fully reproducible models (standardise -> [PCA] -> ridge).

H1 image -> concentration (log2 NaHCO3, CaCl2, MgSO4), inputs: EDS hand-crafted stretch-insensitive features, SEM
   hand-crafted, SEM ResNet18 embedding, optical hand-crafted, optical embedding, geometry only, detector-mode only.
   Protocols: P1 recipe-grouped 5-fold (metadata cv_test_fold), P2 leave-one-session-out, P3 cross-detector
   (train BEC -> test SEI and reverse; NaHCO3 restricted to Tables 2-3 where both detectors exist).
H2 SEM -> EDS (droplet level): predict the 7 element d values from SEM inputs, vs. an oracle that sees the true
   concentrations; plus within-recipe (replicate-to-replicate) correspondence with recipe means removed.
Outputs analysis/out_h/{summary.json, h1_table.csv, h2_table.csv, fig_h1.png}
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = os.environ.get("CRD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root; override with CRD_ROOT
OUT = ROOT + "/analysis/out_h"
os.makedirs(OUT, exist_ok=True)
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
SALT = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]
SEMF = ["sem_z_in_mean", "sem_z_in_sd", "sem_hf_in", "sem_hf_rim", "sem_hf_centre", "sem_bright_frac", "sem_dark_frac", "sem_rim_width"]
OPTF = ["opt_in_d", "opt_rim_d", "opt_centre_d", "opt_bright_frac", "opt_hf"]

f = pd.read_csv(ROOT + "/analysis/out_def/features.csv")
g = pd.read_csv(ROOT + "/analysis/out_g/derived.csv")
df = f.merge(g[["path", "date_optical", "opt_mask_method", "opt_diam_px", "opt_w", "eds_diam_px_half"] + OPTF], on="path")
df = df[df.usable_eds & df.usable_sem & (df.mask_method != "failed")].reset_index(drop=True)
df["opt_diam_rel"] = df.opt_diam_px / df.opt_w
df.loc[df.opt_mask_method == "failed", OPTF + ["opt_diam_rel"]] = np.nan
df["recipe"] = df.table.astype(str) + "_" + df["sample"].astype(str)
df["bec"] = (df.sem_signal == "BEC").astype(float)
Y = np.log2(df[SALT].values)

R = np.load(ROOT + "/analysis/out_def/radial.npz"); ri = {p: i for i, p in enumerate(R["paths"])}
C = np.load(ROOT + "/analysis/out_def/coloc.npz"); ci = {p: i for i, p in enumerate(C["paths"])}
E = np.load(ROOT + "/derived/emb_resnet18.npz"); ei = {p: i for i, p in enumerate(E["paths"])}
radial_shape = np.stack([R["radial_shape"][ri[p]].ravel() for p in df.path])
sem_radial = np.stack([R["sem_radial"][ri[p]] for p in df.path])
coloc_exc = np.stack([C["coloc"][ci[p]][:, 0] - C["coloc"][ci[p]][:, 1] for p in df.path])
eds_hand = np.c_[df[[f"{e}_{k}" for e in EL for k in ("d", "cov", "rbar", "gini")]].values, coloc_exc, radial_shape]
INPUTS = {
    "EDS (hand-crafted)": (eds_hand, None),
    "SEM (hand-crafted)": (np.c_[df[SEMF].values, sem_radial], None),
    "SEM (ResNet18)": (np.stack([E["sem"][ei[p]] for p in df.path]), 64),
    "Optical (hand-crafted)": (df[OPTF + ["opt_diam_rel"]].values, None),
    "Optical (ResNet18)": (np.stack([E["opt"][ei[p]] for p in df.path]), 64),
    "Geometry (EDS + optical diameter)": (np.c_[df.eds_diam_px_half, df.mask_frac, df.opt_diam_rel], None),
    "Detector mode only": (df[["bec"]].values, None),
}
INPUTS["EDS + SEM (ResNet18)"] = (np.c_[eds_hand, INPUTS["SEM (ResNet18)"][0]], 96)


def model(npc, p, n=10 ** 9):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if npc and p > npc:
        steps.append(PCA(min(npc, n // 2), random_state=0))
    return make_pipeline(*steps, RidgeCV(alphas=np.logspace(-2, 5, 29)))


def run(X, y, npc, splits):
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        ok = tr[~np.isnan(y[tr])]
        pred[te] = model(npc, X.shape[1], len(ok)).fit(X[ok], y[ok]).predict(X[te])
    return pred


def score(y, p, rec=None):
    m = ~np.isnan(p) & ~np.isnan(y)
    out = {"R2": round(float(r2_score(y[m], p[m])), 3), "rho": round(float(stats.spearmanr(y[m], p[m]).correlation), 3), "n": int(m.sum())}
    if rec is not None:
        t = pd.DataFrame({"y": y[m], "p": p[m], "r": rec[m]}).groupby("r").mean()
        out["R2_recipe"] = round(float(r2_score(t.y, t.p)), 3)
    return out


idx = np.arange(len(df))
P1 = [(idx[df.cv_test_fold != k], idx[df.cv_test_fold == k]) for k in sorted(df.cv_test_fold.unique())]
P2 = [(idx[df.session != s], idx[df.session == s]) for s in sorted(df.session.unique())]
P2o = [(idx[df.date_optical != s], idx[df.date_optical == s]) for s in sorted(df.date_optical.unique())]
bec, sei = idx[df.bec == 1], idx[df.bec == 0]
t23 = df.table.isin([2, 3]).values
rows = []
for name, (X, npc) in INPUTS.items():
    for j, s in enumerate(SALT):
        y = Y[:, j]
        for prot, sp in (("P1 recipe CV", P1), ("P2 leave-one-session-out", P2), ("P2o leave-one-optical-date-out", P2o)):
            rows.append({"input": name, "target": s, "protocol": prot, **score(y, run(X, y, npc, sp), df.recipe.values)})
        if name == "Detector mode only":
            continue
        for a, b, lab in ((bec, sei, "P3 train BEC -> test SEI"), (sei, bec, "P3 train SEI -> test BEC")):
            if s == "NaHCO3_mM":
                a, b = a[t23[a]], b[t23[b]]
            p = np.full(len(y), np.nan); p[b] = model(npc, X.shape[1], len(a)).fit(X[a][~np.isnan(y[a])], y[a][~np.isnan(y[a])]).predict(X[b])
            rows.append({"input": name, "target": s, "protocol": lab, **score(y, p, df.recipe.values)})
h1 = pd.DataFrame(rows)
h1.to_csv(OUT + "/h1_table.csv", index=False)

# ---------------- H2 SEM -> EDS (element d)
D = df[[f"{e}_d" for e in EL]].values
rows = []
srcs = {"SEM (hand-crafted)": INPUTS["SEM (hand-crafted)"], "SEM (ResNet18)": INPUTS["SEM (ResNet18)"],
        "Concentrations (oracle)": (Y, None), "Concentrations + session (oracle)": (np.c_[Y, pd.get_dummies(df.session).values.astype(float)], None)}
for name, (X, npc) in srcs.items():
    for k, e in enumerate(EL):
        for prot, sp in (("P1 recipe CV", P1), ("P2 leave-one-session-out", P2)):
            if "session" in name and prot.startswith("P2"):
                continue
            rows.append({"input": name, "element": e, "protocol": prot, **score(D[:, k], run(X, D[:, k], npc, sp))})
# within recipe: residuals from recipe means on both sides (replicate-to-replicate correspondence)
def demean(A):
    return A - pd.DataFrame(A).groupby(df.recipe.values).transform("mean").values
for name in ("SEM (hand-crafted)", "SEM (ResNet18)"):
    X, npc = srcs[name]
    Xr = demean(np.nan_to_num(X, nan=np.nanmedian(X)))
    for k, e in enumerate(EL):
        rows.append({"input": name, "element": e, "protocol": "within-recipe (P1 folds)", **score(demean(D)[:, k], run(Xr, demean(D)[:, k], npc, P1))})
h2 = pd.DataFrame(rows)
h2.to_csv(OUT + "/h2_table.csv", index=False)

summ = {"n": int(len(df)),
        "H1": {f"{r.input} | {r.target} | {r.protocol}": {k: r[k] for k in ("R2", "rho", "R2_recipe", "n")} for _, r in h1.iterrows()},
        "H2": {f"{r.input} | {r.element} | {r.protocol}": {k: r[k] for k in ("R2", "rho", "n")} for _, r in h2.iterrows()}}
json.dump(summ, open(OUT + "/summary.json", "w"), indent=1, default=float)

# figure: R2 by input x protocol per salt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
names = list(INPUTS)
for a, s in zip(ax, SALT):
    for k, (prot, c) in enumerate([("P1 recipe CV", "C0"), ("P2 leave-one-session-out", "C1"), ("P3 train BEC -> test SEI", "C2"), ("P3 train SEI -> test BEC", "C3")]):
        v = [h1[(h1.input == n) & (h1.target == s) & (h1.protocol == prot)].R2.values for n in names]
        v = [x[0] if len(x) else np.nan for x in v]
        a.barh(np.arange(len(names)) + (k - 1.5) * .2, np.clip(v, -0.5, 1), height=.2, color=c, label=prot)
    a.axvline(0, c="k", lw=.6); a.set_title(s.replace("_mM", "") + " (log2)"); a.set_xlim(-.5, 1); a.set_xlabel("R² (clipped at -0.5)")
ax[0].set_yticks(np.arange(len(names))); ax[0].set_yticklabels(names); ax[0].invert_yaxis(); ax[2].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "/fig_h1.png", dpi=150); plt.close(fig)
print("H done")
