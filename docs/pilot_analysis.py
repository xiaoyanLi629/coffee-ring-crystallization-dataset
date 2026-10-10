"""Pilot statistics on pilot/features.csv. Writes pilot/summary.json and diagnostic figures."""
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/pilot"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
df = pd.read_csv(ROOT + "/features.csv")
df["recipe"] = df.table.astype(str) + "_" + df["sample"].astype(str)
df["NaHCO3"], df["CaCl2"], df["MgSO4"] = df.C_mM, df.Ca_mM, df.S_mM
SALTS = ["NaHCO3", "CaCl2", "MgSO4"]
OWN = {"C": "NaHCO3", "Na": "NaHCO3", "Ca": "CaCl2", "Cl": "CaCl2", "Mg": "MgSO4", "S": "MgSO4", "O": None}
out = {}

# A. mask QC
mf = df.mask_frac
out["mask"] = {"median": mf.median(), "p5": mf.quantile(.05), "p95": mf.quantile(.95),
               "n_lt_0.05": int((mf < .05).sum()), "n_gt_0.8": int((mf > .8).sum()),
               "suspect": df.loc[(mf < .05) | (mf > .8), "rep"].tolist()[:30]}

# B. dose response: standardised multiple regression of each EDS feature on the three salts
def betas(sub, y):
    X = sub[SALTS].values.astype(float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Y = sub[y].values.astype(float); Y = (Y - Y.mean()) / (Y.std() + 1e-9)
    A = np.c_[X, np.ones(len(X))]
    b, *_ = np.linalg.lstsq(A, Y, rcond=None)
    r2 = 1 - ((Y - A @ b) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return dict(zip(SALTS, np.round(b[:3], 3).tolist())) | {"R2": round(float(r2), 3), "n": len(sub)}

dose = {}
for feat in ["contrast", "total"]:
    for e in EL:
        y = f"{e}_{feat}"
        rec = df.groupby("recipe").agg({y: "mean", **{s: "first" for s in SALTS}}).reset_index()
        dose[y] = {"droplet_all": betas(df, y), "recipe_mean": betas(rec, y),
                   "BEC": betas(df[df.sem_signal == "BEC"], y), "SEI": betas(df[df.sem_signal == "SEI"], y)}
        if OWN[e]:
            dose[y]["spearman_own"] = round(float(stats.spearmanr(df[y], df[OWN[e]]).correlation), 3)
            # within-table spearman with own salt (only CaCl2/MgSO4 vary within a table)
            if OWN[e] != "NaHCO3":
                dose[y]["spearman_own_within_table"] = {int(t): round(float(stats.spearmanr(g[y], g[OWN[e]]).correlation), 3)
                                                       for t, g in df.groupby("table")}
out["dose"] = dose

# C. shortcut: can acquisition "style" predict the Table (= NaHCO3 level)?
gkf = GroupKFold(5)
groups = df.recipe
def clf_score(cols, target):
    X = df[cols].values; y = df[target].values
    p = cross_val_predict(RandomForestClassifier(500, random_state=0, n_jobs=-1), X, y, cv=gkf, groups=groups)
    return round(float(balanced_accuracy_score(y, p)), 3)
style = [c for c in df.columns if c.startswith("semstyle_")]
eds_bg = [f"{e}_out" for e in EL] + [f"{e}_outsd" for e in EL]
eds_chem = [f"{e}_contrast" for e in EL] + [f"{e}_total" for e in EL]
opt = ["opt_mean", "opt_std", "opt_b", "opt_g", "opt_r", "opt_p99"]
out["shortcut_table_balacc(chance 0.2)"] = {
    "SEM style only": clf_score(style, "table"),
    "EDS background only (outside droplet)": clf_score(eds_bg, "table"),
    "EDS export size only": clf_score(["eds_w", "eds_h"], "table"),
    "EDS in-droplet chemistry": clf_score(eds_chem, "table"),
    "optical global stats": clf_score(opt, "table"),
}
df["mode_bin"] = (df.sem_signal == "SEI").astype(int)
out["shortcut_mode_balacc(chance 0.5)"] = {"SEM style only": clf_score(style, "mode_bin")}

# regression of each salt from different feature families (recipe-grouped)
def reg_score(cols, target, sub=None):
    d = df if sub is None else df[sub]
    p = cross_val_predict(RandomForestRegressor(500, random_state=0, n_jobs=-1), d[cols].values, d[target].values,
                          cv=GroupKFold(5), groups=d.recipe)
    return round(float(r2_score(d[target].values, p)), 3)
out["salt_R2"] = {s: {"SEM style": reg_score(style, s), "EDS chemistry": reg_score(eds_chem, s),
                      "EDS background": reg_score(eds_bg, s), "optical": reg_score(opt, s)} for s in SALTS}
# within detector mode
out["salt_R2_within_mode"] = {m: {s: {"SEM style": reg_score(style, s, df.sem_signal == m),
                                      "EDS chemistry": reg_score(eds_chem, s, df.sem_signal == m)} for s in SALTS}
                              for m in ["BEC", "SEI"]}

# D. replicate consistency: ICC(1) from one-way ANOVA over recipes
def icc1(y):
    g = df.groupby("recipe")[y]
    k = g.size().mean(); grand = df[y].mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (g.ngroups - 1)
    msw = ((df[y] - g.transform("mean")) ** 2).sum() / (len(df) - g.ngroups)
    return round(float((msb - msw) / (msb + (k - 1) * msw)), 3)
icc_feats = ["drop_d_norm50", "drop_ecc", "mask_frac", "sem_contrast", "opt_mean"] + \
            [f"{e}_contrast" for e in EL] + [f"{e}_rim_ratio" for e in EL] + [f"{e}_out" for e in EL]
out["icc1"] = {f: icc1(f) for f in icc_feats}

# E. geometry
out["geometry"] = {
    "drop_d_norm50 median/IQR": [round(df.drop_d_norm50.median(), 3), round(df.drop_d_norm50.quantile(.25), 3), round(df.drop_d_norm50.quantile(.75), 3)],
    "spearman d_frac vs mag": round(float(stats.spearmanr(df.drop_d_frac, df.sem_mag).correlation), 3),
    "spearman d_norm50 vs mag": round(float(stats.spearmanr(df.drop_d_norm50, df.sem_mag).correlation), 3),
    "d_norm50 by table": df.groupby("table").drop_d_norm50.median().round(3).to_dict(),
    "d_norm50 by mode": df.groupby("sem_signal").drop_d_norm50.median().round(3).to_dict(),
    "betas d_norm50 on salts": betas(df, "drop_d_norm50"),
}

# F. radial structure
rz = np.load(ROOT + "/radial.npz", allow_pickle=True)
R = rz["radial"]  # N x 7 x 10
out["rim_ratio_median"] = {e: round(float(np.nanmedian(df[f"{e}_rim_ratio"])), 2) for e in EL}
out["rim_ratio_vs_salts"] = {e: betas(df.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"{e}_rim_ratio"]), f"{e}_rim_ratio") for e in EL}

# G. EDS background by table/mode/date (session effects)
out["eds_background_by_table"] = {e: df.groupby("table")[f"{e}_out"].median().round(4).to_dict() for e in EL}
out["eds_background_by_mode"] = {e: df.groupby("sem_signal")[f"{e}_out"].median().round(4).to_dict() for e in EL}
out["eds_d_median"] = {e: round(float(df[f"{e}_d"].median()), 2) for e in EL}
out["eds_frac_droplets_d_gt_1"] = {e: round(float((df[f"{e}_d"] > 1).mean()), 3) for e in EL}

json.dump(out, open(ROOT + "/summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# figures
fig, ax = plt.subplots(2, 7, figsize=(24, 7))
for k, e in enumerate(EL):
    s = OWN[e] or "MgSO4"
    for m, c in [("BEC", "tab:blue"), ("SEI", "tab:orange")]:
        g = df[df.sem_signal == m].groupby(s)[f"{e}_contrast"]
        ax[0, k].errorbar(g.mean().index, g.mean().values, g.sem().values, marker="o", color=c, label=m)
    ax[0, k].set_title(f"{e} contrast vs {s}"); ax[0, k].legend(fontsize=7)
    prof = np.nanmedian(R[:, k, :], 0); q1 = np.nanpercentile(R[:, k, :], 25, 0); q3 = np.nanpercentile(R[:, k, :], 75, 0)
    x = (np.arange(10) + .5) / 10
    ax[1, k].plot(x, prof, "k-o"); ax[1, k].fill_between(x, q1, q3, alpha=.3)
    ax[1, k].set_title(f"{e} radial (0 centre, 1 edge)")
plt.tight_layout(); plt.savefig(ROOT + "/fig_dose_radial.png", dpi=80)

masks = rz["masks"]; idx = np.random.RandomState(0).choice(len(df), 24, replace=False)
fig, ax = plt.subplots(4, 6, figsize=(18, 10))
for a, i in zip(ax.ravel(), idx):
    a.imshow(masks[i], cmap="gray"); a.set_title(df.rep[i], fontsize=7); a.axis("off")
plt.tight_layout(); plt.savefig(ROOT + "/fig_masks.png", dpi=70)
print("analysis done")
