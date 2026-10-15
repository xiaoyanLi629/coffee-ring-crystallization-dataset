"""Modules D (chemical consistency), E (deposit spatial structure), F (reproducibility) on analysis/out_def/*.
All chemistry effects are reported with and without calendar-day (session) fixed effects."""
import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_def"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
OWN = {"C": "NaHCO3_mM", "Na": "NaHCO3_mM", "Ca": "CaCl2_mM", "Cl": "CaCl2_mM", "Mg": "MgSO4_mM", "S": "MgSO4_mM", "O": "MgSO4_mM"}
SALTS = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]
df = pd.read_csv(OUT + "/features.csv")
rz = np.load(OUT + "/radial.npz", allow_pickle=True); cz = np.load(OUT + "/coloc.npz", allow_pickle=True)
keep = (df.usable_eds & df.usable_sem & (df.mask_method != "failed")).values
df = df[keep].reset_index(drop=True)
RAD, SHAPE, SEMR = rz["radial"][keep], rz["radial_shape"][keep], rz["sem_radial"][keep]
COL = cz["coloc"][keep]; PAIRS = [tuple(p) for p in cz["pairs"]]
df["recipe"] = df.table.astype(str) + "_" + df["sample"].astype(str)
for s in SALTS:
    df["log_" + s] = np.log(df[s])
df["log_CaSO4_product"] = np.log(df.CaCl2_mM * df.MgSO4_mM)
df["log_tds"] = np.log(df.tds_mg_L)
df["drop_size_rel"] = np.sqrt(df.mask_frac) * 50.0 / df.sem_mag  # relative physical diameter
S_FE = pd.get_dummies(df.session, drop_first=True).values.astype(float)
res = {"n": int(len(df)), "mask_method": df.mask_method.value_counts().to_dict()}


def ols(y, X, fe=False, sub=None):
    """standardised betas and SEs for the columns of X (DataFrame); optional session fixed effects"""
    idx = np.arange(len(df)) if sub is None else np.where(sub)[0]
    yy = y.iloc[idx].astype(float).values
    ok = np.isfinite(yy); idx, yy = idx[ok], yy[ok]
    Xv = X.iloc[idx].astype(float).values
    Xv = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-12); yy = (yy - yy.mean()) / (yy.std() + 1e-12)
    A = np.c_[Xv, np.ones(len(yy))]
    if fe:
        F = pd.get_dummies(df.session.iloc[idx], drop_first=True).values.astype(float)
        A = np.c_[A, F]
    b, *_ = np.linalg.lstsq(A, yy, rcond=None)
    e = yy - A @ b; dof = max(len(yy) - np.linalg.matrix_rank(A), 1)
    cov = np.linalg.pinv(A.T @ A) * (e ** 2).sum() / dof
    k = X.shape[1]
    return {c: [round(float(b[i]), 3), round(float(np.sqrt(cov[i, i])), 3)] for i, c in enumerate(X.columns)}


LS = df[["log_" + s for s in SALTS]]

# ---------------- D1 dose response with stretch-insensitive measures ----------------
d1 = {}
for e in EL:
    d1[e] = {}
    for m in ["cov", "d", "contrast"]:
        y = df[f"{e}_{m}"]
        d1[e][m] = {"no_FE": ols(y, LS), "session_FE": ols(y, LS, fe=True)}
        # within-session spearman with own salt (only meaningful for CaCl2 / MgSO4, which vary within a day)
        if OWN[e] != "NaHCO3_mM":
            rhos = [stats.spearmanr(g[f"{e}_{m}"], g[OWN[e]]).correlation for _, g in df.groupby("session") if g[OWN[e]].nunique() > 2]
            d1[e][m]["within_session_rho_median"] = round(float(np.nanmedian(rhos)), 3)
res["D1_dose"] = d1

# ---------------- D2 co-localisation ----------------
pn = [f"{EL[a]}-{EL[b]}" for a, b in PAIRS]
d2 = {}
for j, p in enumerate(pn):
    obs, ctl = COL[:, j, 0], COL[:, j, 1]
    d2[p] = {"obs_median": round(float(np.nanmedian(obs)), 3), "ctrl_median": round(float(np.nanmedian(ctl)), 3),
             "excess_median": round(float(np.nanmedian(obs - ctl)), 3), "frac_obs_gt_ctrl": round(float(np.nanmean(obs > ctl)), 3)}
res["D2_coloc"] = dict(sorted(d2.items(), key=lambda kv: -kv[1]["excess_median"]))
for j, p in enumerate(pn):
    df[f"coloc_{p}"] = COL[:, j, 0] - COL[:, j, 1]
res["D2_coloc_vs_salts_FE"] = {p: ols(df[f"coloc_{p}"], LS, fe=True) for p in ["Na-Cl", "Ca-S", "Ca-O", "O-S", "Mg-S", "Mg-O", "C-Ca", "C-O", "Na-S", "Cl-Mg"] if f"coloc_{p}" in df}

# ---------------- D3 crystal-level evidence from the modelling project ----------------
MP = "/root/autodl-fs/coffee-ring-water-chemistry/results"
for name in ["crystal_unit_test.json", "crystal_phases_v2.json", "crystal_phases.json"]:
    p = os.path.join(MP, name)
    if os.path.exists(p):
        try:
            j = json.load(open(p))
            res[f"D3_{name}"] = {k: j[k] for k in list(j)[:12]} if isinstance(j, dict) else str(j)[:2000]
        except Exception as ex:
            res[f"D3_{name}"] = f"unreadable: {ex}"

# ---------------- E1/E2 radial structure ----------------
shape_med = np.nanmedian(SHAPE, 0)  # 7 x NB
res["E1_radial_shape_median"] = {e: np.round(shape_med[k], 2).tolist() for k, e in enumerate(EL)}
res["E2_rbar"] = {e: {"median": round(float(df[f"{e}_rbar"].median()), 3), "IQR": np.round(df[f"{e}_rbar"].quantile([.25, .75]).values, 3).tolist(),
                      "rim_index_median": round(float(df[f"{e}_rim_index"].median()), 2)} for e in EL}
res["E2_rbar_order_outer_to_inner"] = sorted(EL, key=lambda e: -df[f"{e}_rbar"].median())
# paired test: per droplet, is Ca outer than Mg, S outer than Cl, etc.
pairs_e = [("Ca", "Mg"), ("S", "Mg"), ("Ca", "Cl"), ("S", "Cl"), ("Ca", "Na"), ("Na", "Mg"), ("Cl", "Mg"), ("Ca", "S")]
res["E2_paired_outer"] = {f"{a}>{b}": {"frac_droplets": round(float((df[f"{a}_rbar"] > df[f"{b}_rbar"]).mean()), 3),
                                       "wilcoxon_p": float(stats.wilcoxon(df[f"{a}_rbar"], df[f"{b}_rbar"]).pvalue)} for a, b in pairs_e}

# ---------------- E3 composition dependence of the radial structure ----------------
X3 = df[["log_NaHCO3_mM", "log_CaCl2_mM", "log_MgSO4_mM"]]
res["E3_rbar_vs_salts"] = {e: {"no_FE": ols(df[f"{e}_rbar"], X3), "session_FE": ols(df[f"{e}_rbar"], X3, fe=True)} for e in EL}
res["E3_gini_vs_salts_FE"] = {e: ols(df[f"{e}_gini"], X3, fe=True) for e in EL}

# ---------------- E4 deposit size ----------------
res["E4_size"] = {"vs_salts": {"no_FE": ols(df.drop_size_rel, X3), "session_FE": ols(df.drop_size_rel, X3, fe=True)},
                  "vs_tds": {"no_FE": ols(df.drop_size_rel, df[["log_tds"]]), "session_FE": ols(df.drop_size_rel, df[["log_tds"]], fe=True)},
                  "spearman_size_tds": round(float(stats.spearmanr(df.drop_size_rel, df.tds_mg_L).correlation), 3)}

# ---------------- E5 SEM morphology per detector mode ----------------
semf = ["sem_z_in_mean", "sem_z_in_sd", "sem_hf_in", "sem_hf_rim", "sem_hf_centre", "sem_bright_frac", "sem_dark_frac", "sem_rim_width"]
res["E5_sem_by_mode_FE"] = {mode: {f: ols(df[f], X3, fe=True, sub=(df.sem_signal == mode).values) for f in semf} for mode in ["BEC", "SEI"]}

# ---------------- F reproducibility ----------------
nest = df.groupby("recipe").session.nunique()
res["F0_recipes_spanning_multiple_sessions"] = int((nest > 1).sum())


def var_components(y):
    g = df.assign(y=y).dropna(subset=["y"])
    resid = g.groupby("recipe").y.var(ddof=1).mean()
    rec_means = g.groupby(["session", "recipe"]).y.mean().reset_index()
    k = g.groupby("recipe").size().mean()
    within_sess = rec_means.groupby("session").y.var(ddof=1).mean() - resid / k
    sess = g.groupby("session").y.mean().var(ddof=1) - (within_sess + resid / k) / rec_means.groupby("session").size().mean()
    v = np.clip([sess, within_sess, resid], 0, None); tot = v.sum() + 1e-12
    return {"session": round(float(v[0] / tot), 3), "recipe_within_session": round(float(v[1] / tot), 3), "replicate": round(float(v[2] / tot), 3)}


feats_f = [f"{e}_{m}" for e in EL for m in ["cov", "d", "rbar", "rim_index", "gini"]] + semf + ["drop_size_rel"]
vc = {f: var_components(df[f].replace([np.inf, -np.inf], np.nan)) for f in feats_f}
pd.DataFrame(vc).T.to_csv(OUT + "/variance_components.csv")
res["F1_variance_components"] = vc

# F2 replicate agreement of radial-profile shapes (alignment free)
flat = SHAPE.reshape(len(df), -1)
ok = np.isfinite(flat).all(1)
C = np.corrcoef(np.nan_to_num(flat))
rec = df.recipe.values; ses = df.session.values
same_rec, same_ses_diff_rec, diff_ses = [], [], []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if not (ok[i] and ok[j]):
            continue
        if rec[i] == rec[j]:
            same_rec.append(C[i, j])
        elif ses[i] == ses[j]:
            same_ses_diff_rec.append(C[i, j])
        else:
            diff_ses.append(C[i, j])
res["F2_radial_shape_corr"] = {"same_recipe": round(float(np.median(same_rec)), 3), "same_day_other_recipe": round(float(np.median(same_ses_diff_rec)), 3),
                               "other_day": round(float(np.median(diff_ses)), 3), "n_pairs": [len(same_rec), len(same_ses_diff_rec), len(diff_ses)]}
per_el = {}
for k, e in enumerate(EL):
    Ce = np.corrcoef(np.nan_to_num(SHAPE[:, k, :]))
    a, b = [], []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            (a if rec[i] == rec[j] else b if ses[i] == ses[j] else []).append(Ce[i, j])
    per_el[e] = [round(float(np.median(a)), 3), round(float(np.median(b)), 3)]
res["F2_per_element_[same_recipe, same_day_other_recipe]"] = per_el

json.dump(res, open(OUT + "/summary.json", "w"), indent=1, default=float)

# ---------------- figures ----------------
x = (np.arange(12) + .5) / 12
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for k, e in enumerate(EL):
    ax[0].plot(x, shape_med[k], "-o", ms=3, label=e)
ax[0].set_xlabel("normalised radius (0 centre, 1 edge)"); ax[0].set_ylabel("background-subtracted signal / droplet mean"); ax[0].legend(); ax[0].set_title("E1 radial shape (median over droplets)")
order = res["E2_rbar_order_outer_to_inner"]
ax[1].boxplot([df[f"{e}_rbar"] for e in order], labels=order, showfliers=False); ax[1].set_title("E2 intensity-weighted mean radius")
plt.tight_layout(); plt.savefig(OUT + "/fig_radial.png", dpi=90)

fig, ax = plt.subplots(figsize=(12, 4.5))
items = list(res["D2_coloc"].items())
ax.bar(range(len(items)), [v["obs_median"] for _, v in items], label="observed", alpha=.8)
ax.bar(range(len(items)), [v["ctrl_median"] for _, v in items], label="shifted control", alpha=.6)
ax.set_xticks(range(len(items))); ax.set_xticklabels([k for k, _ in items], rotation=60); ax.legend(); ax.set_title("D2 pixel co-localisation inside the droplet (median r)")
plt.tight_layout(); plt.savefig(OUT + "/fig_coloc.png", dpi=90)

fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
for a_, e in zip(ax.ravel(), ["Ca", "Cl", "Mg", "O", "S"]):
    for sess, g in df.groupby("session"):
        if g[OWN[e]].nunique() > 2:
            gg = g.groupby(OWN[e])[f"{e}_cov"].mean()
            a_.plot(gg.index, gg.values, "-", alpha=.5)
    a_.set_title(f"{e} coverage vs {OWN[e].replace('_mM', '')} (mM), one line per day", fontsize=10); a_.set_xscale("log")
plt.tight_layout(); plt.savefig(OUT + "/fig_dose_by_day.png", dpi=80)
print("stats done")
