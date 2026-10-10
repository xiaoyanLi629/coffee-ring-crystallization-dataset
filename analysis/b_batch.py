"""Module B: acquisition batch effects.

B2 variance decomposition (chemistry vs session), B3 NaHCO3 effect identified within mixed sessions,
B4 shortcut test: recipe-grouped CV vs leave-one-session-out, raw vs normalised SEM features, permutation null.
Inputs: metadata/samples.csv, pilot/features.csv, pilot/radial.npz (masks). Outputs: analysis/out_b/.
"""
import os, json
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_b"
os.makedirs(OUT, exist_ok=True)
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
SALTS = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]

meta = pd.read_csv(ROOT + "/metadata/samples.csv")
pil = pd.read_csv(ROOT + "/pilot/features.csv")
pil["droplet_id"] = [f"T{a}_S{b}_R{c}" for a, b, c in zip(pil.table, pil["sample"], pil.replicate)]
keep = [c for c in pil.columns if c not in meta.columns or c == "droplet_id"]
df = meta.merge(pil[keep], on="droplet_id")
rz = np.load(ROOT + "/pilot/radial.npz", allow_pickle=True)
mask_by_rep = dict(zip(rz["reps"], rz["masks"]))
df["recipe"] = df.table.astype(str) + "_" + df["sample"].astype(str)
df = df[df.usable_eds & df.usable_sem].reset_index(drop=True)


# ---------- normalised SEM features (substrate-referenced) ----------
def sem_norm(path):
    sem = cv2.imread(f"{ROOT}/experiment/data/{path}/sem.jpg", cv2.IMREAD_GRAYSCALE)
    sm = cv2.resize(sem, (640, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(mask_by_rep[path].astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = ~cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    med = np.median(sm[out]); iqr = np.subtract(*np.percentile(sm[out], [75, 25])) + 1e-3
    z = (sm - med) / iqr
    f = {}
    for q in [5, 25, 50, 75, 95]:
        f[f"semn_in_p{q}"] = float(np.percentile(z[m], q))
    hp = z - cv2.GaussianBlur(z, (0, 0), 3)
    f["semn_in_hf"] = float(hp[m].std())
    f["semn_in_bright_frac"] = float((z[m] > 2).mean())
    f["semn_in_dark_frac"] = float((z[m] < -2).mean())
    # rim vs centre (relative)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    r = 1 - dist / (dist.max() + 1e-6)
    f["semn_rim_minus_centre"] = float(z[m & (r > .8)].mean() - z[m & (r < .5)].mean())
    return f


with Pool(48) as p:
    sn = p.map(sem_norm, df.path.tolist(), chunksize=4)
df = pd.concat([df, pd.DataFrame(sn)], axis=1)

FAM = {
    "SEM style (raw, whole image)": [c for c in df.columns if c.startswith("semstyle_")],
    "SEM in-droplet, substrate-normalised": [c for c in df.columns if c.startswith("semn_")],
    "EDS in-droplet chemistry": [f"{e}_contrast" for e in EL] + [f"{e}_total" for e in EL],
    "EDS relative (rim ratio, d)": [f"{e}_rim_ratio" for e in EL] + [f"{e}_d" for e in EL],
    "EDS background (outside droplet)": [f"{e}_out" for e in EL] + [f"{e}_outsd" for e in EL],
    "Optical global": ["opt_mean", "opt_std", "opt_b", "opt_g", "opt_r", "opt_p99"],
    "Geometry": ["drop_d_frac", "drop_ecc", "mask_frac"],
}
for k, cols in FAM.items():
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = df[cols].fillna(df[cols].median())
res = {"n": len(df)}


# ---------- B2 variance decomposition ----------
def dummies(cols):
    return pd.get_dummies(df[cols].astype(str), drop_first=True).values.astype(float)


def r2_of(X, y):
    X = np.c_[X, np.ones(len(y))]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


Xc = dummies(SALTS)            # chemistry: 3 salts x 5 levels, main effects
Xs = dummies(["session"])      # 12 calendar days
dec = {}
for fam, cols in FAM.items():
    for c in cols:
        y = df[c].values.astype(float)
        rc, rs, rcs = r2_of(Xc, y), r2_of(Xs, y), r2_of(np.c_[Xc, Xs], y)
        dec[c] = dict(family=fam, R2_chem=rc, R2_session=rs, R2_both=rcs,
                      unique_chem=rcs - rs, unique_session=rcs - rc, shared=rc + rs - rcs)
dec = pd.DataFrame(dec).T
dec.to_csv(OUT + "/variance_decomposition.csv")
res["variance_decomposition_family_median"] = dec.groupby("family")[["R2_chem", "R2_session", "unique_chem", "unique_session", "shared"]].median().astype(float).round(3).to_dict(orient="index")

# ---------- B3 NaHCO3 effect with and without session fixed effects ----------
df["log_NaHCO3"] = np.log(df.NaHCO3_mM)
base = np.c_[df.log_NaHCO3, dummies(["CaCl2_mM", "MgSO4_mM"])]
b3 = {}
for c in FAM["EDS in-droplet chemistry"] + FAM["SEM in-droplet, substrate-normalised"] + FAM["EDS background (outside droplet)"] + ["opt_mean", "drop_d_frac"]:
    y = (df[c] - df[c].mean()) / (df[c].std() + 1e-12)
    out = {}
    for name, X in [("no_session", base), ("session_FE", np.c_[base, Xs])]:
        A = np.c_[X, np.ones(len(y))]
        b, *_ = np.linalg.lstsq(A, y.values, rcond=None)
        resid = y.values - A @ b
        dof = len(y) - A.shape[1]
        cov = np.linalg.pinv(A.T @ A) * (resid ** 2).sum() / dof
        out[name] = [round(float(b[0]), 3), round(float(np.sqrt(cov[0, 0])), 3)]
    b3[c] = out
res["NaHCO3_effect_per_log_unit_[beta,se]"] = b3
mixed = df[df.session.isin(["2021-08-16", "2021-08-19", "2021-08-23"])]
res["mixed_sessions"] = mixed.groupby(["session", "table"]).size().astype(int).to_dict()
res["mixed_sessions"] = {f"{a} T{b}": v for (a, b), v in res["mixed_sessions"].items()}

# ---------- B4 predictive tests ----------
def cv_r2(cols, target, cv, groups):
    p = cross_val_predict(RandomForestRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1),
                          df[cols].values, df[target].values, cv=cv, groups=groups)
    return round(float(r2_score(df[target], p)), 3)


b4 = {}
for fam, cols in FAM.items():
    b4[fam] = {}
    for t in SALTS:
        b4[fam][t] = {"recipe_5fold": cv_r2(cols, t, GroupKFold(5), df.recipe),
                      "leave_one_session_out": cv_r2(cols, t, LeaveOneGroupOut(), df.session)}
res["salt_R2_recipeCV_vs_LOSO"] = b4

# shortcut: Table from features, with permutation null (labels permuted at recipe level within CaCl2 x MgSO4 cell)
def table_acc(cols, y):
    p = cross_val_predict(RandomForestClassifier(400, random_state=0, n_jobs=-1), df[cols].values, y, cv=GroupKFold(5), groups=df.recipe)
    return balanced_accuracy_score(y, p)


rng = np.random.RandomState(0)
cell = df.CaCl2_mM.astype(str) + "_" + df.MgSO4_mM.astype(str)
sc = {}
for fam in ["SEM style (raw, whole image)", "SEM in-droplet, substrate-normalised", "EDS background (outside droplet)", "EDS in-droplet chemistry", "EDS relative (rim ratio, d)", "Optical global"]:
    obs = table_acc(FAM[fam], df.table.values)
    null = []
    for _ in range(20):
        rec_tab = df.groupby("recipe").agg(table=("table", "first"), cell=("CaCl2_mM", lambda x: 0)).reset_index()
        perm = {}
        for cval, g in df.groupby(cell):
            recs = g.recipe.unique(); tabs = [df.loc[df.recipe == r, "table"].iat[0] for r in recs]
            for r, t in zip(recs, rng.permutation(tabs)):
                perm[r] = t
        null.append(table_acc(FAM[fam], df.recipe.map(perm).values))
    sc[fam] = {"balanced_acc": round(float(obs), 3), "perm_null_mean": round(float(np.mean(null)), 3), "perm_null_max": round(float(np.max(null)), 3)}
res["table_from_features_balanced_acc"] = sc
res["mode_from_SEM_norm_balacc"] = round(float(balanced_accuracy_score(df.sem_signal,
    cross_val_predict(RandomForestClassifier(400, random_state=0, n_jobs=-1), df[FAM["SEM in-droplet, substrate-normalised"]].values,
                      df.sem_signal.values, cv=GroupKFold(5), groups=df.recipe))), 3)

json.dump(res, open(OUT + "/summary.json", "w"), indent=1, default=float)
df.to_csv(OUT + "/features_with_semnorm.csv", index=False)

# ---------- figures ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
order = sorted(df.session.unique())
for k, e in enumerate(["Mg", "Na", "S"]):
    for t, g in df.groupby("table"):
        gg = g.groupby("session")[f"{e}_out"].median()
        ax[k].plot([order.index(s) for s in gg.index], gg.values, "o-", label=f"T{t}")
    ax[k].set_xticks(range(len(order))); ax[k].set_xticklabels([s[5:] for s in order], rotation=60)
    ax[k].set_title(f"{e} background outside droplet, by day"); ax[k].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "/fig_background_by_day.png", dpi=80)

d2 = dec.copy(); d2["fam"] = d2.family
fig, ax = plt.subplots(figsize=(10, 6))
for i, (fam, g) in enumerate(d2.groupby("fam")):
    ax.scatter(g.unique_chem.astype(float), g.unique_session.astype(float), label=fam, s=25)
ax.plot([0, .8], [0, .8], "k:"); ax.set_xlabel("unique R² chemistry (given session)"); ax.set_ylabel("unique R² session (given chemistry)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT + "/fig_variance_decomposition.png", dpi=80)
print("B done")
