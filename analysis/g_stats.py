"""Module G (stats): provenance, SEM-EDS registration, SEM grey vs composition by detector, detector-mode bridge,
Mg/Al substrate leakage, optical modality.  Inputs analysis/out_g/derived.csv, analysis/out_def/{features.csv,radial.npz}.
Outputs analysis/out_g/{summary.json, fig_sem_vs_eds.png, fig_mode_bridge.png, fig_mg_al.png, fig_optical.png}
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_g"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
Z = {"C": 6, "O": 8, "Na": 11, "Mg": 12, "S": 16, "Cl": 17, "Ca": 20}
SEMF = ["sem_z_in_mean", "sem_z_in_sd", "sem_hf_in", "sem_hf_rim", "sem_bright_frac", "sem_dark_frac", "sem_rim_width"]
SALT = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]

g = pd.read_csv(OUT + "/derived.csv")
f = pd.read_csv(ROOT + "/analysis/out_def/features.csv")
df = f.merge(g[[c for c in g.columns if c not in f.columns or c == "path"]], on="path")
df = df[df.usable_eds & df.usable_sem & (df.mask_method != "failed")].copy()
for s in SALT:
    df["l" + s] = np.log2(df[s])
df["bec"] = (df.sem_signal == "BEC").astype(float)
res = {"n": int(len(df))}


def zs(x):
    return (x - x.mean()) / x.std()


def ols(y, X, names):
    X = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    dof = len(y) - X.shape[1]
    cov = np.linalg.pinv(X.T @ X) * (e @ e) / dof
    se = np.sqrt(np.diag(cov))
    return {n: [round(float(b[i + 1]), 3), round(float(se[i + 1]), 3)] for i, n in enumerate(names)}


def dummies(s):
    return pd.get_dummies(s, drop_first=True).values.astype(float)


# ---------------- G0 provenance
gg = g.copy()
res["G0"] = {"optical_dates": gg.date_optical.value_counts().sort_index().to_dict(),
             "optical_date_by_table": pd.crosstab(gg.date_optical, gg.table).to_dict(),
             "optical_res_by_date": pd.crosstab(gg.date_optical, gg.opt_w.astype(str) + "x" + gg.opt_h.astype(str)).to_dict(),
             "optical_before_sem": int((pd.to_datetime(gg.date_optical) < pd.to_datetime(gg.session)).sum()),
             "processed_dates": gg.date_processed.value_counts().to_dict()}

# ---------------- G1 SEM-EDS registration (modelling project's registration, re-summarised)
reg = json.load(open("/root/autodl-fs/coffee-ring-water-chemistry/data/derived/eds_registration.json"))
rd = pd.DataFrame(reg["droplets"])
res["G1"] = {"n": int(len(rd)), "n_fallback": int(rd.fallback.sum()),
             "scale_quantiles": np.round(rd.s.quantile([.05, .5, .95]).values, 4).tolist(),
             "shift_px_eds_grid_median_abs": [float(rd.tx.abs().median()), float(rd.ty.abs().median())],
             "ncc_identity_median": round(float(rd.ncc_identity.median()), 3), "ncc_registered_median": round(float(rd.ncc_registered.median()), 3),
             "ncc_second_peak_median": round(float(rd.ncc_second_peak.median()), 3),
             "frac_peak_margin_gt_0.05": round(float(((rd.ncc_registered - rd.ncc_second_peak) > .05).mean()), 3)}

# ---------------- G2 SEM grey vs composition, by detector
g2 = {}
for mode, d in df.groupby("sem_signal"):
    g2[mode] = {"n": int(len(d)), "R2_median": round(float(d.sem_on_eds_R2.median()), 3)}
    for e in EL:
        exc = d[f"semr_{e}"] - d[f"semr_{e}_ctrl"]
        g2[mode][e] = {"beta_median": round(float(d[f"semb_{e}"].median()), 3), "beta_frac_pos": round(float((d[f"semb_{e}"] > 0).mean()), 3),
                       "r_median": round(float(d[f"semr_{e}"].median()), 3), "excess_median": round(float(exc.median()), 3),
                       "excess_frac_pos": round(float((exc > 0).mean()), 3)}
    g2[mode]["spearman_beta_vs_Z"] = round(float(stats.spearmanr([g2[mode][e]["beta_median"] for e in EL], [Z[e] for e in EL]).correlation), 3)
res["G2"] = g2
# recipe-level: does BEC grey level track composition? per mode, mean sem_z_in_mean vs mean atomic-number proxy (mass-weighted)
MW = {"NaCl": (58.44, {"Na": 22.99, "Cl": 35.45}), "CaSO4": (136.14, {"Ca": 40.08, "S": 32.06, "O": 64.0}),
      "MgSO4": (120.37, {"Mg": 24.31, "S": 32.06, "O": 64.0}), "NaHCO3": (84.0, {"Na": 22.99, "H": 1.0, "C": 12.0, "O": 48.0})}

# ---------------- G3 detector-mode bridge
g3 = {}
sem = df[SEMF].apply(zs)
# (a) how well SEM features identify the mode, leave-one-session-out
sess = df.session.values
acc = []
for s in np.unique(sess):
    tr, te = sess != s, sess == s
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(sem[tr], df.bec[tr])
    acc.append((clf.predict(sem[te]) == df.bec[te]).mean() * te.sum())
g3["mode_from_sem_features_LOSO_acc"] = round(float(np.sum(acc) / len(df)), 3)
# (b) shortcut: how much of NaHCO3 is explained by the mode label alone
g3["shortcut_R2_lNaHCO3_from_mode"] = round(float(np.corrcoef(df.bec, df.lNaHCO3_mM)[0, 1] ** 2), 3)
# accuracy of the rule "BEC -> NaHCO3 <= 0.75 mM, SEI -> NaHCO3 >= 1 mM"
g3["shortcut_lowhigh_NaHCO3_acc_from_mode"] = round(float(((df.bec == 1) == (df.NaHCO3_mM <= 0.75)).mean()), 3)
# (c) mode effect at fixed NaHCO3: Tables 2 and 3 only (both modes present), recipe salts and table as covariates
b = df[df.table.isin([2, 3])]
g3["bridge_n"] = {str(k): int(v) for k, v in b.groupby(["table", "sem_signal"]).size().items()}
g3["bridge_sessions"] = {f"{k[0]}_{k[1]}": v for k, v in b.groupby(["table", "sem_signal"]).session.unique().apply(list).items()}
g3["mode_effect_T2T3"] = {}
g3["NaHCO3_within_mode_sessionFE"] = {}
g3["NaHCO3_pooled_sessionFE"] = {}
for c in SEMF:
    y = zs(df[c])
    yb = y[b.index]
    X = np.c_[b.bec, b.lCaCl2_mM, b.lMgSO4_mM, (b.table == 3).astype(float)]
    g3["mode_effect_T2T3"][c] = ols(yb.values, X, ["BEC", "lCa", "lMg", "T3"])["BEC"]
    for mode, d in df.groupby("sem_signal"):
        yy = y[d.index].values
        X = np.c_[d.lNaHCO3_mM, d.lCaCl2_mM, d.lMgSO4_mM, dummies(d.session)]
        g3["NaHCO3_within_mode_sessionFE"].setdefault(c, {})[mode] = ols(yy, X, ["lNa", "lCa", "lMg"] + [f"s{i}" for i in range(X.shape[1] - 3)])["lNa"]
    X = np.c_[df.lNaHCO3_mM, df.lCaCl2_mM, df.lMgSO4_mM, dummies(df.session)]
    g3["NaHCO3_pooled_sessionFE"][c] = ols(y.values, X, ["lNa", "lCa", "lMg"] + [f"s{i}" for i in range(X.shape[1] - 3)])["lNa"]
# which sessions identify NaHCO3 within a mode
g3["two_table_sessions"] = {s: {"tables": sorted(d.table.unique().tolist()), "mode": d.sem_signal.iloc[0]} for s, d in df.groupby("session") if d.table.nunique() > 1}
res["G3"] = g3

# ---------------- G4 Mg (and O) signal from the Al substrate
R = np.load(ROOT + "/analysis/out_def/radial.npz")
ridx = {p: i for i, p in enumerate(R["paths"])}
rad = np.stack([R["radial"][ridx[p]] for p in df.path])  # (n, 7, 12) in-bin mean minus outside mean
g4 = {}
for k, e in enumerate(EL):
    rim = np.nanmean(rad[:, k, -2:], 1); cen = np.nanmean(rad[:, k, :6], 1)
    g4[e] = {}
    for lvl, idx in df.groupby("MgSO4_mM").groups.items():
        ii = df.index.get_indexer(idx)
        g4[e][str(lvl)] = {"d_median": round(float(df.loc[idx, f"{e}_d"].median()), 3),
                           "frac_d_lt0": round(float((df.loc[idx, f"{e}_d"] < 0).mean()), 3),
                           "rim_minus_out_median": round(float(np.nanmedian(rim[ii])), 4), "frac_rim_below_substrate": round(float((rim[ii] < 0).mean()), 3),
                           "centre_minus_out_median": round(float(np.nanmedian(cen[ii])), 4)}
kMg, kS, kCa = EL.index("Mg"), EL.index("S"), EL.index("Ca")
mg_rim = np.nanmean(rad[:, kMg, -2:], 1)
load_rim = np.nanmean(rad[:, kS, -2:], 1) / (np.nanstd(rad[:, kS, -2:]) + 1e-9) + np.nanmean(rad[:, kCa, -2:], 1) / (np.nanstd(rad[:, kCa, -2:]) + 1e-9)
g4["rho_Mg_rim_vs_CaS_rim_by_MgSO4"] = {str(l): round(float(stats.spearmanr(mg_rim[df.MgSO4_mM.values == l], load_rim[df.MgSO4_mM.values == l]).correlation), 3)
                                       for l in sorted(df.MgSO4_mM.unique())}
for e in ["Mg", "O"]:
    X = np.c_[df.lNaHCO3_mM, df.lCaCl2_mM, df.lMgSO4_mM, dummies(df.session)]
    g4[f"{e}_d_sessionFE"] = {k: v for k, v in ols(zs(df[f"{e}_d"]).values, X, ["lNa", "lCa", "lMg"] + [f"s{i}" for i in range(X.shape[1] - 3)]).items() if not k.startswith("s")}
res["G4"] = g4

# ---------------- G5 optical
o = df[df.opt_mask_method != "failed"].copy()
o["opt_session"] = o.date_optical
o["ratio"] = o.opt_diam_px / o.eds_diam_px_half
g5 = {"mask_methods": g.opt_mask_method.value_counts().to_dict(), "n": int(len(o))}
for res_, d in o.groupby(o.opt_w):
    g5[f"diam_ratio_opt{res_}"] = {"median": round(float(d.ratio.median()), 3), "cv": round(float(d.ratio.std() / d.ratio.mean()), 3),
                                    "pearson_opt_vs_eds_diam": round(float(np.corrcoef(d.opt_diam_px, d.eds_diam_px_half)[0, 1]), 3), "n": int(len(d))}
ok = o.optreg_ncc_ctrl.notna()
dif = o.optreg_ncc[ok] - o.optreg_ncc_ctrl[ok]
g5["registration"] = {"ncc_median": round(float(o.optreg_ncc.median()), 3), "ctrl_median": round(float(o.optreg_ncc_ctrl.median()), 3),
                      "frac_beats_ctrl": round(float((dif > 0).mean()), 3), "frac_beats_ctrl_by_0.05": round(float((dif > .05).mean()), 3),
                      "wilcoxon_p": float(stats.wilcoxon(dif).pvalue), "flip_frac_among_clear": round(float(o.optreg_flip[ok][dif > .05].mean()), 3),
                      "by_optical_date": {k: round(float(((d.optreg_ncc - d.optreg_ncc_ctrl) > .05).mean()), 3) for k, d in o[ok].groupby("opt_session")}}
OPTF = ["opt_diam_px", "opt_in_d", "opt_rim_d", "opt_centre_d", "opt_bright_frac", "opt_hf"]
o["opt_diam_mm_rel"] = o.opt_diam_px / o.opt_w  # resolution-free size proxy
OPTF[0] = "opt_diam_mm_rel"
g5["salt_effects_optical_dateFE"] = {}
for c in OPTF:
    X = np.c_[o.lNaHCO3_mM, o.lCaCl2_mM, o.lMgSO4_mM, dummies(o.opt_session)]
    g5["salt_effects_optical_dateFE"][c] = {k: v for k, v in ols(zs(o[c]).values, X, ["lNa", "lCa", "lMg"] + [f"s{i}" for i in range(X.shape[1] - 3)]).items() if not k.startswith("s")}
# variance share of optical date vs recipe (one-way, descriptive)
vs = {}
for c in OPTF:
    y = zs(o[c]); rec = o.table.astype(str) + "_" + o["sample"].astype(str)
    vs[c] = {"eta2_optical_date": round(float(y.groupby(o.opt_session).transform("mean").var() / y.var()), 3),
             "eta2_recipe": round(float(y.groupby(rec).transform("mean").var() / y.var()), 3)}
g5["eta2"] = vs
res["G5"] = g5

json.dump(res, open(OUT + "/summary.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))

# ---------------- figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, mode in zip(ax, ["BEC", "SEI"]):
    d = df[df.sem_signal == mode]
    order = sorted(EL, key=lambda e: Z[e])
    a.boxplot([d[f"semb_{e}"] for e in order], showfliers=False)
    a.set_xticklabels([f"{e}\nZ={Z[e]}" for e in order]); a.axhline(0, c="k", lw=.6)
    a.set_title(f"{mode} (n={len(d)}): SEM grey ~ element maps"); a.set_ylabel("standardised coefficient")
fig.tight_layout(); fig.savefig(OUT + "/fig_sem_vs_eds.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.8))
xs = np.arange(len(SEMF))
mb = [g3["mode_effect_T2T3"][c] for c in SEMF]
nb = [g3["NaHCO3_within_mode_sessionFE"][c].get("BEC", [np.nan, np.nan]) for c in SEMF]
ns = [g3["NaHCO3_within_mode_sessionFE"][c].get("SEI", [np.nan, np.nan]) for c in SEMF]
for off, vals, lab in [(-.25, mb, "BEC vs SEI (Tables 2-3)"), (0, nb, "log2 NaHCO3, within BEC"), (.25, ns, "log2 NaHCO3, within SEI")]:
    ax.errorbar(xs + off, [v[0] for v in vals], yerr=[1.96 * v[1] for v in vals], fmt="o", capsize=3, label=lab)
ax.axhline(0, c="k", lw=.6); ax.set_xticks(xs); ax.set_xticklabels([c.replace("sem_", "") for c in SEMF], rotation=30)
ax.set_ylabel("effect (SD units)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT + "/fig_mode_bridge.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for e, c in [("Mg", "C0"), ("O", "C1"), ("S", "C2")]:
    k = EL.index(e)
    for lvl, ls in [(0.25, "--"), (2.0, "-")]:
        sel = df.MgSO4_mM.values == lvl
        ax[0].plot((np.arange(12) + .5) / 12, np.nanmedian(rad[sel, k], 0), ls, c=c, label=f"{e}, MgSO4={lvl}")
ax[0].axhline(0, c="k", lw=.6); ax[0].set_xlabel("relative radius (0 centre, 1 edge)"); ax[0].set_ylabel("map level minus substrate")
ax[0].legend(fontsize=7)
lv = sorted(df.MgSO4_mM.unique())
ax[1].plot(lv, [g4["Mg"][str(l)]["frac_rim_below_substrate"] for l in lv], "o-", label="Mg")
ax[1].plot(lv, [g4["O"][str(l)]["frac_rim_below_substrate"] for l in lv], "s-", label="O")
ax[1].set_xlabel("MgSO4 (mM)"); ax[1].set_ylabel("fraction of droplets, rim below substrate"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT + "/fig_mg_al.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for w, d in o.groupby("opt_w"):
    ax[0].scatter(d.eds_diam_px_half, d.opt_diam_px, s=6, label=f"optical {w} px wide")
ax[0].set_xlabel("EDS droplet diameter (half-grid px)"); ax[0].set_ylabel("optical droplet diameter (px)"); ax[0].legend(fontsize=8)
ax[1].hist(o.optreg_ncc[ok], bins=30, alpha=.6, label="same droplet"); ax[1].hist(o.optreg_ncc_ctrl[ok], bins=30, alpha=.6, label="other replicate")
ax[1].set_xlabel("optical-SEM gradient NCC (best rotation/mirror)"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT + "/fig_optical.png", dpi=150); plt.close(fig)
print("G stats done")
